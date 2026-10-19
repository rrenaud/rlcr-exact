"""Math generation check, generation side: greedy full rollouts (<think> ... </confidence>, then until the model stops
or --max-new) of a math split under one scripts/math_prompts.py system prompt, with an optional confidence LoRA,
graded per row.

The pilot adapters were trained on rows whose prompt is that system prompt (math_pilot_prep.sh), so generation uses
it too. Labels: `correct` is format-free, like the training labels (RLCR's math-verify on the text through the
first </answer> plus a canonical analysis/confidence tail; 0 without </answer>); `strict` is RLCR's own label on the
whole completion (zero when the tabc format check fails, e.g. text after </confidence>). `stop` = the completion
ends right after </confidence>.

usage: gen_math.py --prompt NAME --data SPLIT.jsonl --out OUT.jsonl [--adapter DIR] [--max-new 4096]   (vLLM env)
"""
import argparse, json, os, re, sys, time

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import math_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-correctness")
ap.add_argument("--prompt", required=True, choices=math_prompts.NAMES)
ap.add_argument("--data", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--max-new", type=int, default=4096)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--util", type=float, default=0.85)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.rlcr)
from reward_fns import accuracy_reward

CANON = "\n<analysis>\nx\n</analysis>\n<confidence>\n0.5\n</confidence>"  # as regen_analysis.py
qs = [json.loads(l) for l in open(a.data)][: a.limit]
llm = LLM(model=a.model, gpu_memory_utilization=a.util, max_model_len=8192, seed=a.seed, enable_prefix_caching=True,
          enable_lora=a.adapter is not None, max_lora_rank=64)
tok = llm.get_tokenizer()
sys_prompt = math_prompts.build(a.prompt)
texts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                  {"role": "user", "content": f"\n\nPROBLEM: {q['problem']}\n\n"}],
                                 tokenize=False, add_generation_prompt=True) for q in qs]
params = SamplingParams(temperature=0.0, max_tokens=a.max_new, seed=a.seed,
                        stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")])
lora = LoRARequest("policy", 1, a.adapter) if a.adapter else None
t = time.time()
res = llm.generate(texts, params, lora_request=lora, use_tqdm=False)
dt = time.time() - t


def grade(texts, qs):
    return accuracy_reward("tabc", [[{"role": "assistant", "content": x}] for x in texts], [q["answer"] for q in qs],
                           [q["source"] for q in qs])


outs = [r.outputs[0] for r in res]
gens = [o.text for o in outs]
heads = [g[: g.index("</answer>") + len("</answer>")] if "</answer>" in g else None for g in gens]
correct = grade([h + CANON if h else "" for h in heads], qs)
strict = grade(gens, qs)
with open(a.out, "w") as f:
    for q, o, g, h, c, s in zip(qs, outs, gens, heads, correct, strict):
        ans = re.findall(r"<answer>(.*?)</answer>", h, re.S) if h else []
        conf = re.findall(r"<confidence>(.*?)</confidence>", g, re.S)
        try:
            cv = float(conf[-1]) if conf else None
        except ValueError:
            cv = None
        f.write(json.dumps(dict(id=q["id"], answer=q["answer"], prompt=a.prompt, adapter=a.adapter, text=g,
                                gen_tokens=len(o.token_ids), finish=o.finish_reason,
                                truncated=o.finish_reason == "length", has_answer=h is not None,
                                pred=ans[0].strip() if ans else None, correct=float(c), strict=float(s),
                                confidence=cv if cv is not None and 0 <= cv <= 1 else None,
                                stop="</confidence>" in g and g.split("</confidence>")[-1].strip() == "")) + "\n")
n = len(qs)
print(json.dumps(dict(event="done", data=a.data, prompt=a.prompt, adapter=a.adapter, n=n, seconds=round(dt, 1),
                      accuracy=sum(correct) / n, strict=sum(strict) / n, mean_gen_tokens=sum(len(o.token_ids) for o in outs) / n,
                      truncated=sum(o.finish_reason == "length" for o in outs) / n)), flush=True)

# vLLM's engine shutdown can hang holding the GPU: kill the engine subprocesses, then exit.
import psutil
children = psutil.Process().children(recursive=True)
for ch in children:
    ch.kill()
psutil.wait_procs(children, timeout=30)
os._exit(0)
