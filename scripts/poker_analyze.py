"""Summarize poker_eval / tk_poker_eval outputs: stated equity (the model's P(win)) against the exact oracle.

Per tag, over completions with a parsed confidence:
  fmt_ok        RLCR's tabc format rate (all completions); parsed = share with a usable equity
  yes           share answering "yes" (the claim format asks for "yes")
  mae, rmse     |e - p_win|: equity error against the oracle
  spearman      rank correlation of e with p_win (discrimination against the truth)
  auroc         e against the sampled outcome (win = 1)
  brier_label   (e - win)^2 against the sampled label (what RLCR's reward sees)
  brier_oracle  E over the outcome with the oracle: p(1-e)^2 + (1-p)e^2 (no label noise)
  caller        expected caller loss (rlcr_exact.betting "caller": bets 1/4-2x pot), final pots per decision
  ..._exact     the same with E_q[c] (exact confidence over the rlcr11 grid) where available
Baselines on the same rows: const = the dev mean p_win for every spot; oracle = p_win itself (floor).
Unparsed completions are dropped (reported in parsed), so compare parsed rates before comparing scores.

usage: poker_analyze.py JSONL_GZ [...] [--by street,villain_kind]
"""
import argparse, collections, gzip, json, os, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rlcr_exact.betting import Score  # noqa: E402

CALLER = Score("caller")


def metrics(e, p, y):
    e, p, y = map(np.asarray, (e, p, y))
    out = dict(n=len(e), mae=float(np.abs(e - p).mean()), rmse=float(np.sqrt(((e - p) ** 2).mean())),
               brier_label=float(((e - y) ** 2).mean()), brier_oracle=float((p * (1 - e) ** 2 + (1 - p) * e ** 2).mean()),
               caller=float((CALLER.H(p) + CALLER.D(p, e)).mean()))
    out["spearman"] = float(spearmanr(e, p).correlation) if len(set(e)) > 1 else 0.0
    out["auroc"] = float(roc_auc_score(y, e)) if 0 < y.mean() < 1 and len(set(e)) > 1 else 0.5
    return out


def report(tag, R, by):
    ok = [r for r in R if r.get("equity_text") is not None]
    head = dict(tag=tag, completions=len(R), fmt_ok=np.mean([r["format_ok"] for r in R]), parsed=len(ok) / max(1, len(R)),
                yes=np.mean([r["answer"] == "yes" for r in R]), tokens=np.mean([r["tokens"] for r in R]),
                truncated=np.mean([bool(r["truncated"]) for r in R]))
    print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in head.items()))
    if not ok:
        return
    groups = [("all", ok)] + [(f"{k}={v}", [r for r in ok if r[k] == v]) for k in by for v in sorted({r[k] for r in ok})]
    cols = ["n", "mae", "rmse", "spearman", "auroc", "brier_label", "brier_oracle", "caller"]
    print(f"  {'group':22s} {'who':8s} " + " ".join(f"{c:>12s}" for c in cols))
    mean_p = np.mean([r["p_win"] for r in R])
    for g, rs in groups:
        if not rs:
            continue
        p = [r["p_win"] for r in rs]; y = [float(r["outcome"] > 0) for r in rs]
        rows = [("model", [r["equity_text"] for r in rs])]
        if all(r.get("equity_exact") is not None for r in rs):
            rows.append(("exactq", [r["equity_exact"] for r in rs]))
        rows += [("const", [mean_p] * len(rs)), ("oracle", p)]
        for who, e in rows:
            m = metrics(e, p, y)
            print(f"  {g:22s} {who:8s} " + " ".join(f"{m[c]:12.3f}" if c != "n" else f"{m[c]:12d}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--by", default="street,villain_kind")
    a = ap.parse_args()
    for f in a.files:
        R = [json.loads(l) for l in gzip.open(f, "rt")]
        report(os.path.basename(f).replace(".jsonl.gz", ""), R, [b for b in a.by.split(",") if b])
        print()


if __name__ == "__main__":
    main()
