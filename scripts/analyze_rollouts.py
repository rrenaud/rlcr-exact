"""Calibration of cached rollouts (from cache_rollouts.py): accuracy, Brier, ECE, AUROC, and confidence usage.

usage: analyze_rollouts.py ROLLOUTS_JSONL [...]
Reads the per-question schema of cache_rollouts.py (outputs, format_ok, correct, lengths, truncated lists).
ECE uses 10 equal-width bins, as in ~/rlcr_platt/recal_eval.py (which reproduces the paper's tables).
Rollouts whose confidence does not parse as a number in [0, 1] are counted and left out of the metrics.
"""
import collections, json, re, sys

import numpy as np
from sklearn.metrics import roc_auc_score


def ece(c, y, bins=10):
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in range(bins) if (b == k).any()))


def confidence(text):
    found = re.findall(r"<confidence>(.*?)</confidence>", text, re.S)
    try:
        c = float(found[-1])
    except (IndexError, ValueError):
        return None
    return c if 0 <= c <= 1 else None


for path in sys.argv[1:]:
    rows = [dict(id=q["id"], completion=text, correct=q["correct"][j], format_ok=q["format_ok"][j],
                 n_tokens=q["lengths"][j], finish_reason="length" if q["truncated"][j] else "stop")
            for q in map(json.loads, open(path)) for j, text in enumerate(q["outputs"])]
    conf = [confidence(r["completion"]) for r in rows]
    ok = np.array([c is not None for c in conf])
    c = np.array([x for x in conf if x is not None])
    y = np.array([r["correct"] for r, x in zip(rows, conf) if x is not None])
    print(f"== {path}")
    print(f"rollouts {len(rows)}  format_ok {np.mean([r['format_ok'] for r in rows]):.3f}  "
          f"confidence parsed {ok.mean():.3f}  truncated {np.mean([r['finish_reason'] == 'length' for r in rows]):.3f}  "
          f"tokens p50 {int(np.median([r['n_tokens'] for r in rows]))}")
    print(f"accuracy {y.mean():.3f}  mean confidence {c.mean():.3f}  Brier {np.mean((c - y) ** 2):.3f}  "
          f"ECE {ece(c, y):.3f}  AUROC {roc_auc_score(y, c):.3f}")
    strings = collections.Counter(re.findall(r"<confidence>(.*?)</confidence>", r["completion"], re.S)[-1]
                                  if "</confidence>" in r["completion"] else "<none>" for r in rows)
    print("confidence strings:", strings.most_common(12))
    print("accuracy by stated confidence:",
          {v: (round(float(y[c == v].mean()), 2), int((c == v).sum())) for v in sorted(set(c)) if (c == v).sum() >= 20})

    by_id = collections.defaultdict(list)
    for r, x in zip(rows, conf):
        by_id[r["id"]].append((r["correct"], x))
    if max(len(v) for v in by_id.values()) > 1:
        # Per problem: empirical accuracy over its samples vs its mean stated confidence.
        groups = [v for v in by_id.values() if sum(x is not None for _, x in v) > 1]
        acc = np.array([np.mean([a for a, _ in v]) for v in groups])
        mc = np.array([np.mean([x for _, x in v if x is not None]) for v in groups])
        within_sd = np.mean([np.std([x for _, x in v if x is not None]) for v in groups])
        print(f"problems {len(by_id)}  with mixed correctness {np.mean((acc > 0) & (acc < 1)):.3f}  "
              f"per-problem accuracy vs mean confidence: corr {np.corrcoef(acc, mc)[0, 1]:.3f}, "
              f"Brier against per-problem accuracy {np.mean((mc - acc) ** 2):.3f}  "
              f"within-problem confidence sd {within_sd:.3f}")
