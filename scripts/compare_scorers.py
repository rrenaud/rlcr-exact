"""Trie vs digit scoring of the confidence on the untrained checkpoint, over cached greedy prefixes.

For each row: the trie's 228 candidate probabilities summed into the digit scorer's 12 levels (a candidate
"0.d..." goes to level 0.d, "1..." to 1.0, "0</confidence>" to the bare "0") vs the digit scorer's level
probabilities; both scorers' greedy levels vs the confidence vLLM actually generated; both scorers' eval
metrics (train_frozen.evaluate); time per row and peak memory.

usage: compare_scorers.py ROLLOUTS_JSONL [--rows N] [--digit-batch B]
Writes $GPUQ_RUN_DIR/compare_scorers.json.
"""
import argparse, json, os, re, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_frozen import evaluate, load_rows, make_scorer, make_token_bytes
from rlcr_exact.confidence import make_candidates

ap = argparse.ArgumentParser()
ap.add_argument("rollouts")
ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-brier-base-tabc")
ap.add_argument("--rows", type=int, default=300)
ap.add_argument("--digit-batch", type=int, default=8)
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
rows = load_rows(a.rollouts, make_token_bytes(tok), 4608)[0][: a.rows]
# The confidence vLLM generated for each (greedy, one output per question) row.
sampled = {q["id"]: re.findall(r"<confidence>(.*?)</confidence>", q["outputs"][0], re.S)[-1]
           for q in map(json.loads, open(a.rollouts)) if q["format_ok"][0]}


def level_of(s):
    """Digit-scorer level index of a confidence string: 0..9 for 0.d..., 10 for 1..., 11 for a bare "0"."""
    if s.startswith("1"):
        return 10
    if s == "0":
        return 11
    return int(s[2]) if re.fullmatch(r"0\.\d.*", s) else None


trie_values, trie_score, trie_info = make_scorer("trie", tok)
digit_values, digit_score, digit_info = make_scorer("digit", tok)
trie_strings = [s.split("<")[0] for s in make_candidates(tok).strings]  # "0.85</confidence><end>" -> "0.85"
trie_to_level = torch.zeros(len(trie_strings), len(digit_values))
for j, s in enumerate(trie_strings):
    trie_to_level[j, level_of(s)] = 1.0

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
prefixes = [r[1] for r in rows]


def timed(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t, torch.cuda.max_memory_allocated() / 2**30


def run(score, batch):
    """All rows through `score` in batches: ([rows, K] logps, greedy pick per row)."""
    outs = [score(model, prefixes[i: i + batch], greedy=True) for i in range(0, len(prefixes), batch)]
    return torch.cat([o[0] for o in outs]), sum((o[1] for o in outs), [])


with torch.no_grad():
    (t_logps, t_picks), t_sec, t_mem = timed(lambda: run(trie_score, 1))
    (d_logps, d_picks), d_sec, d_mem = timed(lambda: run(digit_score, a.digit_batch))
    _, d1_sec, d1_mem = timed(lambda: run(digit_score, 1))

t_levels = t_logps.exp().float().cpu() @ trie_to_level
d_levels = d_logps.exp().float().cpu()
diff = (t_levels - d_levels).abs()
t_greedy = [None if k is None else level_of(trie_strings[k]) for k in t_picks]
vllm = [level_of(sampled[r[0]]) for r in rows]
report = dict(
    rows=len(rows), trie=trie_info, digit=digit_info,
    level_prob_abs_diff=dict(mean=float(diff.mean()), p99=float(diff.max(1).values.quantile(0.99)), max=float(diff.max())),
    mass=dict(trie=float(t_levels.sum(1).mean()), digit=float(d_levels.sum(1).mean()),
              digit_minus_trie_max=float((d_levels.sum(1) - t_levels.sum(1)).max())),
    greedy_agree=dict(trie_vs_digit=float(np.mean([x == y for x, y in zip(t_greedy, d_picks)])),
                      trie_vs_vllm=float(np.mean([x == y for x, y in zip(t_greedy, vllm)])),
                      digit_vs_vllm=float(np.mean([x == y for x, y in zip(d_picks, vllm)]))),
    disagreements=[dict(id=rows[i][0], vllm=sampled[rows[i][0]], trie=t_greedy[i], digit=d_picks[i],
                        max_level_diff=round(float(diff[i].max()), 4))
                   for i in range(len(rows)) if not (t_greedy[i] == d_picks[i] == vllm[i])][:20],
    seconds_per_row=dict(trie=round(t_sec / len(rows), 4), digit_batched=round(d_sec / len(rows), 4),
                         digit_batch1=round(d1_sec / len(rows), 4), digit_batch=a.digit_batch),
    peak_gb=dict(trie=round(t_mem, 2), digit_batched=round(d_mem, 2), digit_batch1=round(d1_mem, 2)),
    eval_trie=evaluate(model, rows, trie_values, trie_score, 1),
    eval_digit=evaluate(model, rows, digit_values, digit_score, a.digit_batch),
)
print(json.dumps(report, indent=1), flush=True)
with open(os.path.join(a.out, "compare_scorers.json"), "w") as f:
    json.dump(report, f, indent=1)
