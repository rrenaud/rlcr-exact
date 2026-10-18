"""Does a scorer reproduce the confidence vLLM generated? Greedy cached rollouts, untrained checkpoint.

For each well-formed row: the scorer's greedy candidate value vs the stated value in the cached text, the mass
on listed candidates, and train_frozen.evaluate's metrics; seconds per row.

usage: check_scorer.py ROLLOUTS_JSONL --scorer digit|trie|hotpot [--rows N]
Writes $GPUQ_RUN_DIR/check_scorer_<scorer>.json.
"""
import argparse, json, os, re, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_frozen import STOP_SCORERS, evaluate, load_rows, make_scorer, make_token_bytes

ap = argparse.ArgumentParser()
ap.add_argument("rollouts")
ap.add_argument("--scorer", required=True, choices=["digit", "trie", "hotpot", "hotpot-stop", "rlcr11"])
ap.add_argument("--model", default=None)
ap.add_argument("--rows", type=int, default=500)
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()
a.model = a.model or ("mehuldamani/hotpot-v2-brier-7b-no-split" if a.scorer.startswith("hotpot") or a.scorer == "rlcr11"
                      else "mehuldamani/big-math-digits-v2-brier-base-tabc")

tok = AutoTokenizer.from_pretrained(a.model)
stop = a.scorer in STOP_SCORERS  # rows ending in junk after </confidence> are kept; greedy must leave the list there
rows = load_rows(a.rollouts, make_token_bytes(tok), 4608, tail_ok=stop)[0][: a.rows]
stated, faulted = {}, set()
for q in map(json.loads, open(a.rollouts)):
    for j, text in enumerate(q["outputs"]):
        if not q["format_ok"][j]:
            faulted.add((q["id"], j))
        found = re.findall(r"<confidence>(.*?)</confidence>", text, re.S)
        try:
            stated[(q["id"], j)] = float(found[-1])
        except (IndexError, ValueError):
            pass
values, score, info = make_scorer(a.scorer, tok)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()

agree, mass, mass_fault, t = 0, [], [], time.time()
with torch.no_grad():
    for r in rows:
        logps, picks = score(model, [r.prefix], greedy=True)
        (mass_fault if (r.qid, r.j) in faulted else mass).append(logps.exp().sum().item())
        if (r.qid, r.j) in faulted:  # vLLM wrote junk instead of stopping: greedy along the trie should leave it
            agree += picks[0] is None
        else:
            agree += picks[0] is not None and abs(values[picks[0]].item() - stated.get((r.qid, r.j), -1)) < 1e-6
seconds = (time.time() - t) / len(rows)
report = dict(scorer=a.scorer, model=a.model, rows=len(rows), **info, greedy_matches_vllm=agree / len(rows),
              mass_mean=sum(mass) / len(mass), mass_min=min(mass), tail_fault_rows=len(mass_fault),
              mass_mean_tail_fault=sum(mass_fault) / len(mass_fault) if mass_fault else None, seconds_per_row=round(seconds, 4),
              eval=evaluate(model, rows, values, score, 1))
print(json.dumps(report, indent=1), flush=True)
with open(os.path.join(a.out, f"check_scorer_{a.scorer}.json"), "w") as f:
    json.dump(report, f, indent=1)
