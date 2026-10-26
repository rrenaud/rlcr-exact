"""k rollouts per question for test-time-scaling analysis, from the prompt token ids of existing cache files.

Reads cache lines (prompt_ids, answer, id) from each input, samples k completions with vLLM at the given temperature
and writes cache-format lines (outputs, completion_ids, format_ok, correct, sampled token log-probs) to OUT, inputs
concatenated with `split` set to the input's name. format_ok is a regex stand-in for RLCR's format reward (an
<answer> and a final <confidence> block that ends the text); correct is exact match of the last <answer>, gated on it.

usage: tts_gen.py IN.jsonl [IN.jsonl ...] --out OUT.jsonl [--k 8] [--temperature 0.7] [--limit N]
"""
import argparse, json, os, re, sys

from vllm import LLM, SamplingParams, TokensPrompt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rlcr_exact.grading import em_correct

FORMAT = re.compile(r"<answer>.*?</answer>.*<confidence>[^<]*</confidence>\s*$", re.DOTALL)

ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max-new", type=int, default=2048)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=None, help="questions per input")
a = ap.parse_args()

qs = []
for path in a.inputs:
    split = os.path.basename(path).removesuffix(".jsonl").rsplit("_", 1)[-1]
    for line in list(open(path))[: a.limit]:
        q = json.loads(line)
        qs.append(dict(id=q["id"], split=split, problem=q["problem"], answer=q["answer"], prompt_ids=q["prompt_ids"]))

llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=8192, seed=a.seed, enable_prefix_caching=True)
params = SamplingParams(n=1, temperature=a.temperature, top_p=1.0, max_tokens=a.max_new, logprobs=0)
outs = llm.generate([TokensPrompt(prompt_token_ids=q["prompt_ids"]) for q in qs for _ in range(a.k)], params)
with open(a.out, "w") as f:
    for i, q in enumerate(qs):
        o = [c for r in outs[i * a.k: (i + 1) * a.k] for c in r.outputs]
        texts = [c.text for c in o]
        fmt = [int(bool(FORMAT.search(t))) for t in texts]
        q |= dict(outputs=texts, completion_ids=[list(c.token_ids) for c in o],
                  lengths=[len(c.token_ids) for c in o], truncated=[c.finish_reason == "length" for c in o],
                  sampled_logprobs=[[round(next(iter(d.values())).logprob, 5) for d in c.logprobs] for c in o],
                  format_ok=fmt, correct=[float(ok and em_correct(t, q["answer"])) for ok, t in zip(fmt, texts)],
                  policy=dict(base=a.model, temperature=a.temperature, seed=a.seed))
        f.write(json.dumps(q) + "\n")
print(f"{len(qs)} questions x {a.k}: format {sum(sum(q['format_ok']) for q in qs) / (a.k * len(qs)):.3f}, "
      f"correct {sum(sum(q['correct']) for q in qs) / (a.k * len(qs)):.3f}")
