"""Prototype: one copy of the base weights serves both vLLM generation and LoRA training (vLLM env, one GPU).

vLLM runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0). An HF Qwen2 is built on the meta device and each of its
frozen parameters is bound to a view of vLLM's tensor (q/k/v are row slices of vLLM's fused qkv_proj, gate/up of
gate_up_proj), so the trainer allocates only the PEFT LoRA and its Adam state. After each update the adapter is saved
to /dev/shm and handed to vLLM as a new LoRARequest.

Checks, in order (each prints one JSON line):
  1. bind: every HF parameter bound, shapes match, a few tensors equal the checkpoint's; GPU memory added by binding + LoRA
  2. logits: HF-on-vLLM-weights log-probs of a cached rollout vs vLLM's prompt_logprobs for the same tokens
  3. update: a few Adam steps on one target string, then vLLM with the saved adapter gives it higher log-prob
  4. cycle: generate 4 questions x 8 rollouts with the adapter (T 0.7), one LoRA forward/backward over them, adapter
     save + reload; times and peak memory. Also 32 questions x 8 at once (the iterated-refresh batch).
usage: shared_vllm_proto.py [--kv-gb 3] [--questions 4]   (vLLM env)
"""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import argparse, glob, json, shutil, time

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.lora.request import LoRARequest

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--rollouts", default="/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137/rollouts_hotpot_fresh_train.jsonl")
ap.add_argument("--kv-gb", type=float, default=3.0)
ap.add_argument("--questions", type=int, default=4)
ap.add_argument("--micro-batch", type=int, default=2)
ap.add_argument("--max-new", type=int, default=512)
a = ap.parse_args()
dev = torch.device("cuda")
GB = 2 ** 30


def out(**k):
    print(json.dumps(k), flush=True)


def mem():
    free, total = torch.cuda.mem_get_info()
    return dict(used_gb=round((total - free) / GB, 2), torch_alloc_gb=round(torch.cuda.memory_allocated() / GB, 2))


# ---- vLLM, in-process ----
t = time.time()
llm = LLM(model=a.model, enable_lora=True, max_lora_rank=8, max_loras=2, max_model_len=4096, seed=0,
          kv_cache_memory_bytes=int(a.kv_gb * GB), enable_prefix_caching=True)
vmodel = llm.apply_model(lambda m: m)[0]  # in-process: the module itself
out(event="vllm_loaded", seconds=round(time.time() - t, 1), model_class=type(vmodel).__name__, **mem())
vparams = {n.replace(".base_layer", ""): p.data for n, p in vmodel.named_parameters()}  # plain tensors: vLLM's Parameter subclasses cannot be re-wrapped
out(event="vllm_params", n=len(vparams), sample={n: list(p.shape) for n, p in list(vparams.items())[:9]},
    inference_tensor=any(p.is_inference() for p in vparams.values()))

# ---- HF model on meta, bound to vLLM's tensors ----
cfg = AutoConfig.from_pretrained(a.model)
H, nh, nkv, I = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size
hd = H // nh
before = torch.cuda.memory_allocated()
with torch.device("meta"):
    hf = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16, attn_implementation="sdpa")


def views(name):
    """HF parameter name -> the vLLM tensor view it should be."""
    if name == "lm_head.weight":
        return vparams["lm_head.weight"][: cfg.vocab_size]
    if name == "model.embed_tokens.weight":
        return vparams["model.embed_tokens.weight"][: cfg.vocab_size]
    for proj, (lo, hi) in (("q_proj", (0, nh * hd)), ("k_proj", (nh * hd, (nh + nkv) * hd)),
                           ("v_proj", ((nh + nkv) * hd, (nh + 2 * nkv) * hd))):
        if f".self_attn.{proj}." in name:
            return vparams[name.replace(proj, "qkv_proj")][lo:hi]
    for proj, (lo, hi) in (("gate_proj", (0, I)), ("up_proj", (I, 2 * I))):
        if f".mlp.{proj}." in name:
            return vparams[name.replace(proj, "gate_up_proj")][lo:hi]
    return vparams[name]


bound = 0
for name, p in list(hf.named_parameters()):
    v = views(name)
    assert tuple(v.shape) == tuple(p.shape), (name, v.shape, p.shape)
    mod = hf.get_submodule(name.rsplit(".", 1)[0])
    mod._parameters[name.rsplit(".", 1)[1]] = torch.nn.Parameter(v, requires_grad=False)
    bound += 1
with torch.device(dev):
    hf.model.rotary_emb = type(hf.model.rotary_emb)(config=cfg)  # non-persistent buffers were on meta
assert not any(b.is_meta for b in hf.buffers()) and not any(p.is_meta for p in hf.parameters())

# a few tensors against the checkpoint itself
from huggingface_hub import snapshot_download
ckpt = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"])
check = ["model.layers.20.self_attn.k_proj.weight", "model.layers.20.self_attn.q_proj.bias", "model.layers.3.mlp.up_proj.weight",
         "model.embed_tokens.weight", "lm_head.weight"]
equal = {}
idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
sd = dict(hf.named_parameters())
for n in check:
    with safe_open(os.path.join(ckpt, idx[n]), "pt", device="cuda") as f:
        equal[n] = bool(torch.equal(f.get_tensor(n), sd[n]))
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
hf.enable_input_require_grads()
peft_cfg = LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      layers_to_transform=list(range(14, cfg.num_hidden_layers)))
model = get_peft_model(hf, peft_cfg)
for n, p in model.named_parameters():
    if "lora_" in n and p.device.type != "cuda":
        raise SystemExit(f"LoRA param {n} on {p.device}")
trainable = [p for p in model.parameters() if p.requires_grad]
torch.cuda.empty_cache()
out(event="bind", bound=bound, equal_to_checkpoint=equal, trainable_params=sum(p.numel() for p in trainable),
    added_by_bind_and_lora_gb=round((torch.cuda.memory_allocated() - before) / GB, 3), **mem())

# ---- logits: HF path vs vLLM prompt_logprobs ----
rows = [json.loads(l) for _, l in zip(range(max(32, a.questions)), open(a.rollouts))]
r0 = rows[0]
ids = r0["prompt_ids"] + r0["completion_ids"][0]
model.eval()
with torch.no_grad(), model.disable_adapter():
    lp_hf = torch.log_softmax(model(torch.tensor([ids], device=dev)).logits[0, :-1].float(), -1)
lp_hf = lp_hf.gather(1, torch.tensor(ids[1:], device=dev)[:, None])[:, 0].cpu()
res = llm.generate(TokensPrompt(prompt_token_ids=ids), SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
lp_v = torch.tensor([next(iter(d.values())).logprob if d else float("nan") for d in res.prompt_logprobs[1:]])
comp = slice(len(r0["prompt_ids"]) - 1, None)  # the completion tokens
diff = (lp_hf - lp_v).abs()
out(event="logits", tokens=len(ids), max_abs_diff=round(diff.max().item(), 4), mean_abs_diff=round(diff.mean().item(), 5),
    completion_sum_logp_hf=round(lp_hf[comp].sum().item(), 3), completion_sum_logp_vllm=round(lp_v[comp].sum().item(), 3))

# ---- update: train toward a target continuation, check vLLM sees it ----
opt = torch.optim.Adam(trainable, lr=2e-3)
prefix = r0["prompt_ids"] + r0["completion_ids"][0][:-8]
tok = llm.get_tokenizer()
target = tok.encode(" zebra zebra zebra", add_special_tokens=False)


def target_logp_vllm(lora):
    res = llm.generate(TokensPrompt(prompt_token_ids=prefix + target), SamplingParams(max_tokens=1, prompt_logprobs=0),
                       lora_request=lora, use_tqdm=False)[0]
    return sum(next(iter(d.values())).logprob for d in res.prompt_logprobs[len(prefix):])


shm = "/dev/shm/rrenaud_shared_vllm_proto"
shutil.rmtree(shm, ignore_errors=True)
version = [0]


def publish():
    version[0] += 1
    path = f"{shm}/v{version[0]}"
    t = time.time(); model.save_pretrained(path); s = time.time() - t
    return LoRARequest(f"v{version[0]}", version[0], path), s


base_lp = target_logp_vllm(None)
model.train()
x = torch.tensor([prefix + target], device=dev)
for _ in range(5):
    logits = model(x).logits[0, len(prefix) - 1:-1].float()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=dev))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
lora, save_s = publish()
t = time.time(); new_lp = target_logp_vllm(lora); first_s = time.time() - t
with torch.no_grad():
    model.eval()
    lp_train = -torch.nn.functional.cross_entropy(model(x).logits[0, len(prefix) - 1:-1].float(), torch.tensor(target, device=dev), reduction="sum").item()
out(event="update", target_logp_base_vllm=round(base_lp, 3), target_logp_adapter_vllm=round(new_lp, 3),
    target_logp_adapter_hf=round(lp_train, 3), save_s=round(save_s, 3), first_use_s=round(first_s, 2), **mem())

# ---- cycle timing ----
def generate(qs, lora):
    t = time.time()
    res = llm.generate([TokensPrompt(prompt_token_ids=q["prompt_ids"]) for q in qs],
                       SamplingParams(n=8, temperature=0.7, max_tokens=a.max_new), lora_request=lora, use_tqdm=False)
    return res, time.time() - t


torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
res, gen_s = generate(rows[: a.questions], lora)
seqs = [q["prompt_ids"] + list(o.token_ids) for q, r in zip(rows, res) for o in r.outputs]
model.train()
t = time.time()
for i in range(0, len(seqs), a.micro_batch):  # plain (unpacked) forward/backward on every rollout: an upper bound
    part = seqs[i: i + a.micro_batch]
    L = max(map(len, part))
    x = torch.full((len(part), L), tok.pad_token_id or 0, device=dev)
    m = torch.zeros((len(part), L), dtype=torch.long, device=dev)
    for j, s in enumerate(part):
        x[j, : len(s)] = torch.tensor(s); m[j, : len(s)] = 1
    # stand-in loss on the last tokens; only those logits are computed (as train_frozen scores only the confidence)
    logits = model(input_ids=x, attention_mask=m, logits_to_keep=9).logits[:, :-1].float()
    loss = logits.logsumexp(-1).mean()
    loss.backward()
opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize(); train_s = time.time() - t
lora, save_s = publish()
t = time.time(); _ = target_logp_vllm(lora); reload_s = time.time() - t
out(event="cycle", questions=a.questions, rollouts=len(seqs), mean_new_tokens=round(sum(len(o.token_ids) for r in res for o in r.outputs) / len(seqs), 1),
    gen_s=round(gen_s, 2), train_fwd_bwd_s=round(train_s, 2), save_s=round(save_s, 3), reload_and_score_s=round(reload_s, 2),
    peak_torch_alloc_gb=round(torch.cuda.max_memory_allocated() / GB, 2), **mem())
res, gen_s = generate(rows[:32], lora)
out(event="refresh_batch", questions=32, rollouts=32 * 8, gen_s=round(gen_s, 2), **mem())
shutil.rmtree(shm, ignore_errors=True)
import psutil
for ch in psutil.Process().children(recursive=True):
    ch.kill()
os._exit(0)
