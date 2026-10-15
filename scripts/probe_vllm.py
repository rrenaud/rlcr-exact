"""vLLM rollout cost and base-model behaviour on RLCR's tasks: one GRPO-sized batch per task.

usage: probe_vllm.py RLCR_DIR MODEL [GPU_MEM_FRACTION]
Appends one JSON line per task to $GPUQ_RUN_DIR/vllm.jsonl and saves sample completions next to it.
"""
import collections, json, os, sys, time

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

sys.path.insert(0, sys.argv[1])
from reward_fns import accuracy_reward, format_reward
from system_prompts import get_sys_prompt

MODEL = sys.argv[2]
MEM = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
OUT_DIR = os.environ.get("GPUQ_RUN_DIR", ".")
PROMPTS, GROUP = 32, 16
TASKS = [("hotpot", "mehuldamani/hotpot_qa", "tabc_long", 1536), ("math", "mehuldamani/big-math-digits", "tabc", 4096)]

llm = LLM(MODEL, gpu_memory_utilization=MEM, max_model_len=6144, enable_prefix_caching=True, seed=0)
tok = llm.get_tokenizer()

for task, name, sp, max_tokens in TASKS:
    ds = load_dataset(name, split="train").shuffle(seed=0).select(range(PROMPTS))
    convs = [[{"role": "system", "content": get_sys_prompt(sp)},
              {"role": "user", "content": f"\n\nPROBLEM: {ex['problem']}\n\n"}] for ex in ds]
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
    params = SamplingParams(n=GROUP, temperature=0.7, max_tokens=max_tokens, seed=0)
    t = time.time()
    results = llm.generate(texts, params)
    dt = time.time() - t

    outs = [o for r in results for o in r.outputs]
    lens = np.array([len(o.token_ids) for o in outs])
    completions = [[{"role": "assistant", "content": o.text}] for o in outs]
    answers = [ex["answer"] for ex in ds for _ in range(GROUP)]
    sources = [ex["source"] for ex in ds for _ in range(GROUP)]
    fmt = np.array(format_reward("tabc", completions))
    acc = np.array([accuracy_reward("tabc", [c], [a], [s])[0] for c, a, s in zip(completions, answers, sources)])
    confs = collections.Counter()
    for o in outs:
        if "<confidence>" in o.text:
            confs[o.text.rsplit("<confidence>", 1)[1].split("</confidence>")[0].strip()[:12]] += 1

    rec = dict(model=MODEL, task=task, prompts=PROMPTS, group=GROUP, mem_fraction=MEM, seconds=round(dt, 1),
               gen_tok_per_s=round(lens.sum() / dt, 1), rollouts_per_s=round(len(outs) / dt, 2),
               len_p50=int(np.median(lens)), len_p90=int(np.percentile(lens, 90)), len_max=int(lens.max()),
               truncated=round(float((lens >= max_tokens).mean()), 3), format_ok=round(float(fmt.mean()), 3),
               accuracy=round(float(acc.mean()), 3), confidence_strings=confs.most_common(12))
    print(json.dumps(rec), flush=True)
    with open(os.path.join(OUT_DIR, "vllm.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(os.path.join(OUT_DIR, f"samples_{MODEL.split('/')[-1]}_{task}.json"), "w") as f:
        json.dump([o.text for o in outs[:: GROUP][:8]], f, indent=1)

# vLLM's engine shutdown can hang with the GPU still held; results are on disk, so kill the
# engine subprocesses ourselves and skip teardown (os._exit alone orphans them, still holding memory).
import psutil
sys.stdout.flush()
children = psutil.Process().children(recursive=True)
for c in children:
    c.kill()
psutil.wait_procs(children, timeout=30)
os._exit(0)
