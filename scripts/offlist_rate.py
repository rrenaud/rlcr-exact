"""Share of greedy eval confidences written off the 11 rlcr11 levels: offlist_rate.py EVAL.jsonl.gz ..."""
import gzip
import json
import re
import sys

LEVELS = {f"{i / 10:.1f}" for i in range(11)} | {"0", "1"}

for path in sys.argv[1:]:
    n = off = 0
    for line in gzip.open(path, "rt"):
        row = json.loads(line)
        out = row["outputs"] if isinstance(row["outputs"], str) else row["outputs"][0]
        found = re.findall(r"<confidence>\s*(.*?)\s*</confidence>", out)
        if found:
            n += 1
            off += found[-1] not in LEVELS
    print(f"{off / max(n, 1):.3f}\tn={n}\t{path}")
