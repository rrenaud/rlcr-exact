"""Analysis-only regeneration: prefill each cached rollout's <think>...</answer> under a new system prompt and
generate the <analysis> and <confidence> (stop at </confidence>), for each prompt in scripts/math_prompts.py.

Answers are cut at the first </answer> of the cached completion (token ids kept as generated). Labels are
prompt-independent: RLCR's math-verify grader on the prefix plus a canonical well-formed analysis/confidence.
Rows whose completion has no </answer> are dropped (counted).

usage: regen_analysis.py --model M --rollouts R.jsonl --split dev|train --temperature T [--per-question k]
                         [--limit n] [--prompts a,b,...] --out OUT.jsonl      (vLLM env, gpuq payload)
"""
import argparse, json, os, sys, time

from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import math_prompts

ap = argparse.ArgumentParser()
ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-correctness")
ap.add_argument("--rollouts", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--per-question", type=int, default=1, help="cached rollouts per question to use as prefixes")
ap.add_argument("--limit", type=int, default=None, help="questions")
ap.add_argument("--prompts", default=",".join(math_prompts.NAMES))
ap.add_argument("--max-new", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--util", type=float, default=0.9)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.rlcr)
from reward_fns import accuracy_reward

llm = LLM(model=a.model, gpu_memory_utilization=a.util, max_model_len=8192, seed=a.seed, enable_prefix_caching=True)
tok = llm.get_tokenizer()
CANON = "\n<analysis>\nx\n</analysis>\n<confidence>\n0.5\n</confidence>"


def cut_at_answer(ids):
    """Shortest prefix of ids whose decoded text contains </answer>, or None."""
    text = ""
    for i, t in enumerate(ids):
        text += tok.decode([t])
        if "</answer>" in text:
            return i + 1
    return None


rows, dropped = [], 0
for qi, line in enumerate(open(a.rollouts)):
    if a.limit is not None and qi >= a.limit:
        break
    q = json.loads(line)
    for j, ids in enumerate(q["completion_ids"][: a.per_question]):
        c = cut_at_answer(ids)
        if c is None:
            dropped += 1
            continue
        prefix = tok.decode(ids[:c])
        label = accuracy_reward("tabc", [[{"role": "assistant", "content": prefix + CANON}]], [q["answer"]], [q["source"]])[0]
        rows.append(dict(id=q["id"], j=j, answer=q["answer"], problem=q["problem"], prefix_ids=ids[:c], correct=label))
print(json.dumps(dict(event="rows", split=a.split, rows=len(rows), dropped_no_answer=dropped,
                      accuracy=sum(r["correct"] for r in rows) / max(len(rows), 1))), flush=True)

params = SamplingParams(temperature=a.temperature, max_tokens=a.max_new, seed=a.seed, stop=["</confidence>"],
                        include_stop_str_in_output=True, stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")])
with open(a.out, "a") as f:
    for name in a.prompts.split(","):
        sys_prompt = math_prompts.build(name)
        heads = {}
        inputs = []
        for r in rows:
            if r["problem"] not in heads:
                heads[r["problem"]] = tok.encode(tok.apply_chat_template(
                    [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"\n\nPROBLEM: {r['problem']}\n\n"}],
                    tokenize=False, add_generation_prompt=True), add_special_tokens=False)
            inputs.append(dict(prompt_token_ids=list(heads[r["problem"]]) + r["prefix_ids"]))
        t = time.time()
        res = llm.generate(inputs, params, use_tqdm=False)
        dt = time.time() - t
        for r, x, o in zip(rows, inputs, res):
            out = o.outputs[0]
            f.write(json.dumps(dict(prompt=name, split=a.split, id=r["id"], j=r["j"], correct=r["correct"],
                                    temperature=a.temperature, prompt_ids=x["prompt_token_ids"],
                                    prefix_len=len(r["prefix_ids"]), gen_ids=list(out.token_ids), gen=out.text,
                                    finish=out.finish_reason, stop=str(out.stop_reason))) + "\n")
        f.flush()
        print(json.dumps(dict(event="prompt", name=name, rows=len(rows), seconds=round(dt, 1),
                              mean_gen_tokens=round(sum(len(o.outputs[0].token_ids) for o in res) / len(res), 1),
                              hit_cap=sum(o.outputs[0].finish_reason == "length" for o in res))), flush=True)

# vLLM's engine shutdown can hang holding the GPU: kill the engine subprocesses, then exit.
import psutil
children = psutil.Process().children(recursive=True)
for ch in children:
    ch.kill()
psutil.wait_procs(children, timeout=30)
os._exit(0)
