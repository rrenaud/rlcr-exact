"""What fits on one 24 GB GPU: HF sampling throughput and LoRA train-step memory for Qwen2.5 base models.

Writes one JSON line per measurement to $GPUQ_RUN_DIR/probe.jsonl (or ./probe.jsonl).
"""
import gc, json, os, sys, time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, sys.argv[1])  # RLCR checkout, for its system prompts
from system_prompts import get_sys_prompt

OUT = os.path.join(os.environ.get("GPUQ_RUN_DIR", "."), "probe.jsonl")
MODELS = [("Qwen/Qwen2.5-0.5B", "bf16"), ("Qwen/Qwen2.5-1.5B", "bf16"), ("Qwen/Qwen2.5-3B", "bf16"),
          ("Qwen/Qwen2.5-7B", "bf16"), ("Qwen/Qwen2.5-7B", "nf4")]
# Training shapes: (task, prompt tokens, completion tokens), near the p50-p90 of RLCR's data.
SHAPES = [("hotpot", 1600, 320), ("math", 300, 1024), ("math-long", 300, 4096)]
GROUP = 16


def log(**rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def prompts(tok):
    out = {}
    for task, name, sp in [("hotpot", "mehuldamani/hotpot_qa", "tabc_long"), ("math", "mehuldamani/big-math-digits", "tabc")]:
        ex = load_dataset(name, split="test")[0]
        text = tok.apply_chat_template([{"role": "system", "content": get_sys_prompt(sp)},
                                        {"role": "user", "content": f"\n\nPROBLEM: {ex['problem']}\n\n"}],
                                       tokenize=False, add_generation_prompt=True)
        out[task] = tok(text, return_tensors="pt").input_ids
    return out


def load(name, quant):
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(name, **kw)


def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def gen_probe(model, tok, ids, task, new_tokens):
    reset()
    batch = ids.repeat(GROUP, 1).cuda()
    torch.cuda.synchronize(); t = time.time()
    try:
        with torch.no_grad():
            model.generate(batch, attention_mask=torch.ones_like(batch), do_sample=True, temperature=0.7,
                           max_new_tokens=new_tokens, min_new_tokens=new_tokens, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); dt = time.time() - t
        return dict(ok=True, seconds=round(dt, 2), tok_per_s=round(GROUP * new_tokens / dt, 1),
                    peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    except torch.OutOfMemoryError:
        return dict(ok=False, oom=True)


def train_probe(model, task, n_prompt, n_comp, micro):
    """One LoRA forward+backward on `micro` sequences; log-probs only over the completion tokens."""
    reset()
    seq = torch.randint(0, 150000, (micro, n_prompt + n_comp), device="cuda")
    torch.cuda.synchronize(); t = time.time()
    try:
        hidden = model.base_model.model.model(input_ids=seq, use_cache=False).last_hidden_state[:, n_prompt - 1:-1]
        logits = model.base_model.model.lm_head(hidden).float()
        logp = torch.log_softmax(logits, -1).gather(-1, seq[:, n_prompt:, None]).squeeze(-1)
        logp.mean().backward()
        torch.cuda.synchronize(); dt = time.time() - t
        model.zero_grad(set_to_none=True)
        return dict(ok=True, seconds=round(dt, 3), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        return dict(ok=False, oom=True)


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    ps = prompts(tok)
    log(event="prompts", **{k: v.shape[1] for k, v in ps.items()}, device=torch.cuda.get_device_name(0))
    for name, quant in MODELS:
        reset()
        try:
            model = load(name, quant)
        except torch.OutOfMemoryError:
            log(model=name, quant=quant, event="load", ok=False, oom=True); continue
        log(model=name, quant=quant, event="load", weights_gb=round(torch.cuda.memory_allocated() / 2**30, 2))
        for task, new in [("hotpot", 320), ("math", 1024)]:
            log(model=name, quant=quant, event="generate", task=task, group=GROUP, prompt=ps[task].shape[1],
                new_tokens=new, **gen_probe(model, tok, ps[task], task, new))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
        model.train()
        for task, n_prompt, n_comp in SHAPES:
            for micro in [1, 2, 4, 8]:
                r = train_probe(model, task, n_prompt, n_comp, micro)
                log(model=name, quant=quant, event="train", task=task, prompt=n_prompt, completion=n_comp,
                    micro=micro, lora_r=16, **r)
                if not r["ok"]:
                    break
        del model
    log(event="done")


if __name__ == "__main__":
    main()
