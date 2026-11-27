"""Data for the Brier-decomposition page (docs/brier_site): brier_site_data.py > docs/brier_site/data.json
Run from /data/gpuq-runs/rrenaud (HotpotQA dev rows, as scripts/brier_decomp.py) in the rlcr21 checkout's python path;
the NBA rows come from docs/brier_site/nba_ft.csv (scripts/nba_ft_extract.py).

Every forecaster is reduced to (stated value v, rows n, successes k) cells, so the page gets reliability diagrams and
the decomposition is exact: Brier = REL - RES + UNC with bins = stated values. REL noise = sum_v v(1-v)/N, what a
calibrated forecaster still shows; iso-CV = 5-fold cross-validated isotonic recalibration of the stated values.
"""
import csv, glob, json, os, sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import betabinom
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from brier_decomp import load  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summarize(c, y, w=None, groups=None):
    """c, y, w: stated value, outcome, weight per row. groups: CV fold unit (default: the row)."""
    w = np.ones_like(c) if w is None else w
    N = w.sum(); ybar = (w * y).sum() / N
    cells, rel, res, noise = [], 0.0, 0.0, 0.0
    for v in np.unique(c):
        m = c == v; n = w[m].sum(); k = (w[m] * y[m]).sum(); yk = k / n
        rel += n / N * (v - yk) ** 2; res += n / N * (yk - ybar) ** 2; noise += v * (1 - v) / N
        cells.append([round(float(v), 4), round(float(n), 3), round(float(k), 3)])
    groups = np.arange(len(c)) if groups is None else groups
    ug = np.unique(groups); iso = np.zeros(len(c))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ug):
        a, b = np.isin(groups, ug[tr]), np.isin(groups, ug[te])
        iso[b] = IsotonicRegression(out_of_bounds="clip").fit(c[a], y[a], sample_weight=w[a]).predict(c[b])
    pos, neg = w * y, w * (1 - y)  # AUROC with ties counted half, weighted
    order = np.argsort(c); cs, ps, ns = c[order], pos[order], neg[order]
    auc, negs_below = 0.0, 0.0
    for v in np.unique(cs):
        m = cs == v; auc += ps[m].sum() * (negs_below + ns[m].sum() / 2); negs_below += ns[m].sum()
    auc /= pos.sum() * neg.sum()
    r = lambda x: round(float(x), 5)
    return dict(N=r(N), base=r(ybar), brier=r((w * (c - y) ** 2).sum() / N), REL=r(rel), RES=r(res),
                UNC=r(ybar * (1 - ybar)), REL_noise=r(noise), iso_cv=r((w * (iso - y) ** 2).sum() / N),
                auroc=r(auc), levels=len(cells), cells=cells)


HOTPOT = [  # (group, label, glob under /data/gpuq-runs/rrenaud)
    ("start", "Qwen2.5-7B base (prompted)", "rlcr-prereg-ref/*f48c2a7e5c/prereg_eval/dev_Qwen2.5-7B_regen0.jsonl.gz"),
    ("start", "RLVR (correctness-only RL)", "rlcr-prereg-ref/*f48c2a7e5c/prereg_eval/dev_hotpot-v2-correctness-7b_regen0.jsonl.gz"),
    ("start", "released RLCR (Brier RL)", "rlcr-prereg-ref/*f48c2a7e5c/prereg_eval/dev_hotpot-v2-brier-7b-no-split_regen0.jsonl.gz"),
    ("A", "A2 exact rlcr11 s102", "rlcr-prereg-A/*9d2314faca/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A2 exact rlcr11 s1102", "rlcr-prereg-A/*3e9453e9de/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A2 exact rlcr11 s2102", "rlcr-prereg-A/*7656f17798/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A3 sampled s103", "rlcr-prereg-A/*86cca6dbca/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A3 sampled s1103", "rlcr-prereg-A/*bdb33349cb/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A3 sampled s2103", "rlcr-prereg-A/*ef8e0a05f7/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A2 exact rlcr21 s102", "rlcr21-A/*b650f0b60e/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A2 exact rlcr21 s1102", "rlcr21-A/*80b1a0c088/prereg_eval/dev_step1000.jsonl.gz"),
    ("A", "A2 exact rlcr21 s2102", "rlcr21-A/*3da62f73c4/prereg_eval/dev_step1000.jsonl.gz"),
    ("B", "B2 exact rlcr11 s202", "rlcr-prereg-B/*924698fd06/prereg_eval/dev_step600.jsonl.gz"),
    ("B", "B2 exact rlcr11 s1202", "rlcr-prereg-B/*905269e6e5/prereg_eval/dev_step600.jsonl.gz"),
    ("B", "B2 exact rlcr11 s2202", "rlcr-prereg-B/*6b0292e176/prereg_eval/dev_step600.jsonl.gz"),
    ("B", "B0 sampled s200", "rlcr-prereg-B/*c337eaf8f4/prereg_eval/dev_step600.jsonl.gz"),
    ("B", "B0 sampled s1200", "rlcr-prereg-B/*0c3962510e/prereg_eval/dev_step600.jsonl.gz"),
    ("B", "B0 sampled s2200", "rlcr-prereg-B/*1f9422108a/prereg_eval/dev_step600.jsonl.gz"),
    ("B", "B2 exact rlcr21 s202", "rlcr21-B/*416932e61c/prereg_eval/dev_step600.jsonl.gz"),
    ("C", "C0 exact (from RLCR)", "rlcr-prereg-C/*390db7df09/prereg_eval/dev_step600.jsonl.gz"),
    ("C", "C0 sampled (from RLCR)", "rlcr-prereg-C/*00d95c574b/prereg_eval/dev_step600.jsonl.gz"),
]


def nba():
    rows = {}
    for r in csv.DictReader(open(os.path.join(REPO, "docs/brier_site/nba_ft.csv"))):
        rows.setdefault(r["season"], {})[r["player_id"]] = (int(r["ft"]), int(r["fta"]), r["player"])
    prev, cur = rows["2023"], rows["2024"]
    k0 = np.array([v[0] for v in prev.values()]); n0 = np.array([v[1] for v in prev.values()])
    keep = n0 > 0; k0, n0 = k0[keep], n0[keep]
    mu0 = k0.sum() / n0.sum()  # last season's league rate: the climatology forecast
    # beta-binomial prior fitted on last season only: the shrinkage a forecaster could have known in advance
    nll = lambda lk: -betabinom.logpmf(k0, n0, mu0 * np.exp(lk), (1 - mu0) * np.exp(lk)).sum()
    kappa = float(np.exp(minimize_scalar(nll, bounds=(0, 8), method="bounded").x))
    ids = [p for p in cur if cur[p][1] > 0]
    ft = np.array([cur[p][0] for p in ids], float); fta = np.array([cur[p][1] for p in ids], float)
    pk = np.array([prev.get(p, (0, 0))[0] for p in ids], float); pn = np.array([prev.get(p, (0, 0))[1] for p in ids], float)
    raw = np.where(pn > 0, pk / np.maximum(pn, 1), mu0)
    shrunk = (pk + kappa * mu0) / (pn + kappa)
    oracle = ft / fta  # in-sample: each shooter's own 2023-24 rate (not a forecast; the most player identity can say)
    # one weighted row per (player, outcome): makes and misses
    def rows_of(f, grid):
        f = np.clip(np.round(f / grid) * grid, 0, 1) if grid else f
        c = np.concatenate([f, f]); y = np.concatenate([np.ones_like(f), np.zeros_like(f)])
        w = np.concatenate([ft, fta - ft]); g = np.concatenate([np.arange(len(f))] * 2)
        m = w > 0
        return c[m], y[m], w[m], g[m]
    out = []
    for key, label, short, f, grid in [
            ("clim", "League rate for everyone", "League rate", np.full(len(ids), mu0), 0.01),
            ("raw", "Last season's own rate, raw", "Last season, raw", raw, 0.01),
            ("shrunk", "Last season's rate, shrunk to the league", "Last season, shrunk", shrunk, 0.01),
            ("shrunk10", "Shrunk, rounded to 0.1 (an 11-level grid)", "Shrunk, 0.1 grid", shrunk, 0.1),
            ("shrunk05", "Shrunk, rounded to 0.05 (a 21-level grid)", "Shrunk, 0.05 grid", shrunk, 0.05),
            ("oracle", "Oracle: this season's own rate (in-sample)", "Oracle (in-sample)", oracle, 0.01)]:
        out.append(dict(key=key, label=label, short=short, **summarize(*rows_of(f, grid)[:3], groups=rows_of(f, grid)[3])))
    meta = dict(players=len(ids), attempts=int(fta.sum()), makes=int(ft.sum()), kappa=round(kappa, 1),
                league_prev=round(float(mu0), 4), rookies=int((pn == 0).sum()))
    return out, meta


if __name__ == "__main__":
    hot = []
    for group, label, pat in HOTPOT:
        c, y = load(glob.glob(pat)[0])
        hot.append(dict(group=group, label=label, **summarize(c, y)))
        print(f"{label:34s} brier {hot[-1]['brier']:.4f} REL {hot[-1]['REL']:.4f} RES {hot[-1]['RES']:.4f} "
              f"AUROC {hot[-1]['auroc']:.3f}", file=sys.stderr)
    nb, meta = nba()
    for d in nb:
        print(f"{d['label']:44s} brier {d['brier']:.5f} REL {d['REL']:.5f} RES {d['RES']:.5f} UNC {d['UNC']:.5f} "
              f"noise {d['REL_noise']:.5f} iso {d['iso_cv']:.5f} AUROC {d['auroc']:.3f}", file=sys.stderr)
    json.dump(dict(hotpot=hot, nba=nb, nba_meta=meta), sys.stdout, separators=(",", ":"))
