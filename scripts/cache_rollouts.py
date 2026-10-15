"""Cache rollouts of a released RLCR checkpoint, logged like ../rlcr-reward-conf-weighted-tt-scaling/rollout_vllm.py.

OUT.jsonl has one line per question: text, prompt/completion token ids, each sampled token's log-prob, lengths,
truncation and the producing policy, plus RLCR's format and correctness labels for each output. OUT.topk.jsonl.gz
holds the top-K log-probs at every token of the last <answer>...</answer> and <confidence>...</confidence>
contents (logprob_spans.span_topk, imported from that work stream so the format stays identical). Log-probs are
vLLM's raw_logprobs: log_softmax of the raw logits at temperature 1.

Generation runs in chunks of questions so the top-K buffers stay small, and resumes from questions already in OUT.

usage: cache_rollouts.py --rlcr DIR --data SPLIT.jsonl --k 8 --temperature 0.7 --out FILE [--limit N]
"""
import argparse, gzip, json, os, sys, time

from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rlcr-reward-conf-weighted-tt-scaling"))
from logprob_spans import make_token_bytes, sampled_logprobs, span_topk

ap = argparse.ArgumentParser()
ap.add_argument("--rlcr", required=True)
ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-brier-base-tabc")
ap.add_argument("--data", required=True)
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--max-new", type=int, default=4096)  # RLCR's math max_completion_length and eval max_tokens
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--topk", type=int, default=64)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--chunk", type=int, default=250, help="questions per generate call")
ap.add_argument("--util", type=float, default=0.9)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.rlcr)
from reward_fns import accuracy_reward, format_reward
from system_prompts import get_sys_prompt

rows = [json.loads(l) for l in open(a.data)][: a.limit]
split = os.path.basename(a.data).removesuffix(".jsonl")
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
todo = [(i, r) for i, r in enumerate(rows) if r["id"] not in done]

llm = LLM(model=a.model, gpu_memory_utilization=a.util, max_model_len=8192, seed=a.seed,
          enable_prefix_caching=True, max_logprobs=max(a.topk, 1))
tok = llm.get_tokenizer()
token_bytes = make_token_bytes(tok)
sys_prompt = get_sys_prompt("tabc")
params = SamplingParams(n=a.k, temperature=a.temperature, top_p=1.0, max_tokens=a.max_new, seed=a.seed,
                        stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")], logprobs=a.topk, flat_logprobs=True)
policy = dict(base=a.model, adapter=None, adapter_sha256_16=None, iter=None, seed=a.seed, temperature=a.temperature,
              logprobs_mode="raw_logprobs")
print(json.dumps(dict(event="start", split=split, questions=len(rows), todo=len(todo), k=a.k,
                      temperature=a.temperature, topk=a.topk)), flush=True)

for c in range(0, len(todo), a.chunk):
    chunk = todo[c : c + a.chunk]
    texts = [tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                      {"role": "user", "content": f"\n\nPROBLEM: {r['problem']}\n\n"}],
                                     tokenize=False, add_generation_prompt=True) for _, r in chunk]
    t = time.time()
    res = llm.generate(texts, params, use_tqdm=False)
    dt, ntok = time.time() - t, sum(len(o.token_ids) for r in res for o in r.outputs)
    with open(a.out, "a") as f, gzip.open(a.out + ".topk.jsonl.gz", "at") as topf:
        for (i, ex), r in zip(chunk, res):
            completions = [[{"role": "assistant", "content": o.text}] for o in r.outputs]
            f.write(json.dumps(dict(
                idx=i, id=ex["id"], task="math", split=split, problem=ex["problem"], answer=ex["answer"],
                source=ex["source"], llama8b_solve_rate=ex["llama8b_solve_rate"], policy=policy,
                outputs=[o.text for o in r.outputs], lengths=[len(o.token_ids) for o in r.outputs],
                truncated=[o.finish_reason == "length" for o in r.outputs],
                prompt_ids=list(r.prompt_token_ids), completion_ids=[list(o.token_ids) for o in r.outputs],
                sampled_logprobs=[sampled_logprobs(list(o.token_ids), o.logprobs) for o in r.outputs],
                format_ok=format_reward("tabc", completions),
                # RLCR's correctness label: math-verify on the last <answer>, zero when the format check fails.
                correct=accuracy_reward("tabc", completions, [ex["answer"]] * len(completions),
                                        [ex["source"]] * len(completions)))) + "\n")
            for j, o in enumerate(r.outputs):
                ids = list(o.token_ids)
                topf.write(json.dumps(dict(idx=i, id=ex["id"], j=j,
                                           answer=span_topk(o.text, ids, o.logprobs, "answer", a.topk, token_bytes),
                                           confidence=span_topk(o.text, ids, o.logprobs, "confidence", a.topk, token_bytes)))
                           + "\n")
    print(json.dumps(dict(event="chunk", split=split, done=c + len(chunk), of=len(todo), seconds=round(dt, 1),
                          gen_tok_per_s=round(ntok / dt, 1))), flush=True)
print(json.dumps(dict(event="done", split=split, out=a.out)), flush=True)

# vLLM's engine shutdown can hang holding the GPU: kill the engine subprocesses, then exit.
import psutil
children = psutil.Process().children(recursive=True)
for ch in children:
    ch.kill()
psutil.wait_procs(children, timeout=30)
os._exit(0)
