"""Murphy decomposition of greedy dev Brier for prereg_eval outputs: brier_decomp.py (no args: the rlcr21 comparison set).

Bins are the stated values (discrete, so Brier = REL - RES + UNC exactly). REL noise = sum_k v_k (1 - v_k) / N, the
REL a perfectly calibrated model still shows on N rows; it grows with the number of levels used. iso-CV = 5-fold
cross-validated isotonic recalibration of the stated values: about the best Brier a reshaping of the confidence
can reach without changing its ranking. Confidence parsing as in prereg_eval: last <confidence>, missing or
outside [0, 1] = 0. Run from /data/gpuq-runs/rrenaud.
"""
import gzip, json, re, sys, glob, ast, numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
def load(path):
    c, y = [], []
    for l in gzip.open(path, "rt"):
        d = json.loads(l)
        o = d["outputs"]; o = o if isinstance(o, str) else o[0]
        yy = d["correct"]; yy = yy if not isinstance(yy, list) else yy[0]
        f = re.findall(r"<confidence>(.*?)</confidence>", o, re.S)
        try: v = float(f[-1]); v = v if 0 <= v <= 1 else 0.0
        except (IndexError, ValueError): v = 0.0
        c.append(v); y.append(float(yy))
    return np.array(c), np.array(y)
def decomp(c, y):
    N = len(y); rel = res = floor = 0.0
    for v in np.unique(c):
        m = c == v; n = m.sum(); yk = y[m].mean()
        rel += n / N * (v - yk) ** 2; res += n / N * (yk - y.mean()) ** 2; floor += v * (1 - v) / N
    iso = np.zeros(N)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(c):
        iso[te] = IsotonicRegression(out_of_bounds="clip").fit(c[tr], y[tr]).predict(c[te])
    return dict(brier=np.mean((c - y) ** 2), REL=rel, REL_noise=floor, RES=res, UNC=y.mean() * (1 - y.mean()),
                iso_cv=np.mean((iso - y) ** 2), levels=len(np.unique(c)))
if __name__ == "__main__":
    runs = {
     "RLCR released r0": "rlcr-prereg-ref/*f48c2a7e5c/prereg_eval/dev_hotpot-v2-brier-7b-no-split_regen0.jsonl.gz",
     "RLCR released r1": "rlcr-prereg-ref/*f48c2a7e5c/prereg_eval/dev_hotpot-v2-brier-7b-no-split_regen1.jsonl.gz",
     "A2 rlcr11 s102 @1000": "rlcr-prereg-A/*9d2314faca/prereg_eval/dev_step1000.jsonl.gz",
     "A2 rlcr11 s1102 @1000": "rlcr-prereg-A/*3e9453e9de/prereg_eval/dev_step1000.jsonl.gz",
     "A2 rlcr11 s2102 @1000": "rlcr-prereg-A/*7656f17798/prereg_eval/dev_step1000.jsonl.gz",
     "A2 rlcr21 s102 @1000": "rlcr21-A/*b650f0b60e/prereg_eval/dev_step1000.jsonl.gz",
     "A2 rlcr21 s1102 @1000": "rlcr21-A/*80b1a0c088/prereg_eval/dev_step1000.jsonl.gz",
     "A2 rlcr21 s2102 @1000": "rlcr21-A/*3da62f73c4/prereg_eval/dev_step1000.jsonl.gz",
     "B2 rlcr11 s202 @600": "rlcr-prereg-B/*924698fd06/prereg_eval/dev_step600.jsonl.gz",
     "B2 rlcr11 s1202 @600": "rlcr-prereg-B/*905269e6e5/prereg_eval/dev_step600.jsonl.gz",
     "B2 rlcr11 s2202 @600": "rlcr-prereg-B/*6b0292e176/prereg_eval/dev_step600.jsonl.gz",
     "B2 rlcr21 s202 @600": "rlcr21-B/*416932e61c/prereg_eval/dev_step600.jsonl.gz",
    }
    print(f"{'run':24s} brier   REL  (noise)  RES    UNC   iso-CV  levels")
    for k, p in runs.items():
        d = decomp(*load(glob.glob(p)[0]))
        print(f"{k:24s} {d['brier']:.4f} {d['REL']:.4f} ({d['REL_noise']:.4f}) {d['RES']:.4f} {d['UNC']:.4f} {d['iso_cv']:.4f} {d['levels']}")
