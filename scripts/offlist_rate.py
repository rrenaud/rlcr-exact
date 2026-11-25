"""Share of greedy eval confidences written off the levels, and (--levels 21) the share on the 0.d5 half-steps:
offlist_rate.py [--levels 11|21] EVAL.jsonl.gz ...  (rlcr11: 0.0 .. 1.0; rlcr21: 0.0, 0.05, ..., 1.0 as RLCR spells them)"""
import argparse
import gzip
import json
import re

ap = argparse.ArgumentParser()
ap.add_argument("--levels", type=int, choices=[11, 21], default=11)
ap.add_argument("evals", nargs="+")
a = ap.parse_args()
LEVELS = {f"{i / 10:.1f}" for i in range(11)} | {"0", "1"}
HALF = {f"0.{d}5" for d in range(10)}
if a.levels == 21:
    LEVELS |= HALF

for path in a.evals:
    n = off = half = 0
    for line in gzip.open(path, "rt"):
        row = json.loads(line)
        out = row["outputs"] if isinstance(row["outputs"], str) else row["outputs"][0]
        found = re.findall(r"<confidence>\s*(.*?)\s*</confidence>", out)
        if found:
            n += 1
            off += found[-1] not in LEVELS
            half += found[-1] in HALF
    print(f"{off / max(n, 1):.3f}\thalf={half / max(n, 1):.3f}\tn={n}\t{path}")
