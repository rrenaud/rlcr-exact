"""Data for the betting-scores page (docs/poker_site): python scripts/poker_site_data.py > docs/poker_site/data.json

Input: docs/poker_site/cells_rlcr21.json, a copy of docs/brier_site/data.json from branch rlcr21 @258efa4
(scripts/brier_site_data.py there). Every forecaster is (stated value v, rows n, hits k) cells, enough to re-score it.

The betting view of a proper score. A forecaster states c and is offered a ticket that costs t and pays 1 if y = 1.
It buys iff c > t. Against the ticket's fair value, its loss is t if it bought and y = 0, and 1 - t if it passed and
y = 1. Average that over a price t ~ w (w integrates to 1) and you get a proper scoring rule S_w(c, y):
  uniform w           -> Brier / 2
  w ∝ 1/(t(1-t))      -> log loss (truncated to t in [0.001, 0.999]: rewarding_doubt's clipped log score)
  poker caller        -> villain bets b x pot with b ~ U[1/4, 2]; calling costs t = b / (1 + 2b) of the final pot.
                         w(t) ∝ db/dt = 1 / (1 - 2t)^2 on [1/6, 0.4]. Units: final pots lost per decision.
Units throughout: expected loss per decision, as a fraction of the ticket payout (the final pot, for poker).

Decomposition (bins = stated values, exact): S = UNC_w - RES_w + REL_w,
  H_w(p) = ∫ w(t) min(p(1-t), (1-p)t) dt       (loss of someone who knows p: the luck left)
  D_w(p||c) = ∫_{t between c and p} w(t) |p - t| dt   (EV lost to wrong decisions at prices between belief and truth)
  UNC_w = H_w(ybar), REL_w = Σ n_v/N D_w(ybar_v || v), RES_w = UNC_w - Σ n_v/N H_w(ybar_v).
iso_cv = score after 5-fold cross-validated isotonic recalibration (HotpotQA only).
REL noise = E[REL_w] for a calibrated forecaster with the same cells (k_v ~ Bin(n_v, v)), computed exactly.
"""
import json, os, sys
import numpy as np
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from rlcr_exact.betting import T, Score  # noqa: E402
def decompose(cells, sc, iso=None):
    v = np.array([c[0] for c in cells]); n = np.array([c[1] for c in cells]); k = np.array([c[2] for c in cells])
    N = n.sum(); yb = k.sum() / N; yv = k / n
    unc = float(sc.H(yb)); within = float((n / N * sc.H(yv)).sum())
    rel = float((n / N * sc.D(yv, v)).sum())
    noise = 0.0
    for vi, ni in zip(v, n):
        ni = int(round(ni)); ks = np.arange(ni + 1)
        noise += ni / N * float((binom.pmf(ks, ni, vi) * sc.D(ks / ni, vi)).sum())
    r = lambda x: round(x, 6)
    out = dict(score=r(unc - (unc - within) + rel), UNC=r(unc), RES=r(unc - within), REL=r(rel), REL_noise=r(noise))
    if iso is not None:
        out["iso_cv"] = r(float(sc.S(iso[0], iso[1]).mean()))
    return out


def iso_cv(cells, seed=0):
    """5-fold CV isotonic recalibration on rows rebuilt from unweighted cells. PAV's fit is the same for every
    proper score (it minimises all of them at once), so one fit serves the three price distributions."""
    c = np.concatenate([[v] * int(n) for v, n, k in cells]).astype(float)
    y = np.concatenate([[1] * int(k) + [0] * int(n - k) for v, n, k in cells]).astype(float)
    pred = np.zeros_like(c)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(c):
        pred[te] = IsotonicRegression(out_of_bounds="clip").fit(c[tr], y[tr]).predict(c[te])
    return pred, y


def curves(sc):  # for the page's weight and regret figures
    xs = np.round(np.linspace(0, 1, 1001), 4)  # F, G: cumulative ∫w, ∫w t, so the page computes D_w exactly as here
    return dict(t=xs.tolist(), w=np.round(np.interp(xs, T, sc.w), 4).tolist(), F=np.round(sc.F_(xs), 6).tolist(),
                G=np.round(sc.G_(xs), 6).tolist())


if __name__ == "__main__":
    src = json.load(open(os.path.join(HERE, "docs/poker_site/cells_rlcr21.json")))
    scores = {k: Score(k) for k in ["uniform", "log", "caller"]}
    # sanity: uniform = Brier/2 on every forecaster
    rows = []
    for grp, items in [("nba", src["nba"]), ("hotpot", src["hotpot"])]:
        for d in items:
            out = dict(group=d.get("group", "nba"), label=d["label"], short=d.get("short"), brier=d["brier"],
                       auroc=d["auroc"], base=d["base"], cells=d["cells"])
            iso = iso_cv(d["cells"]) if grp == "hotpot" else None  # NBA cells are weighted; its folds need players
            for k, sc in scores.items():
                out[k] = decompose(d["cells"], sc, iso)
            assert abs(out["uniform"]["score"] * 2 - d["brier"]) < 2e-4, (d["label"], out["uniform"], d["brier"])
            rows.append(out)
            u, c = out["uniform"], out["caller"]
            print(f"{d['label'][:34]:34s} brier/2 {u['score']:.4f} REL {u['REL']:.4f}({u['REL_noise']:.4f}) | "
                  f"log {out['log']['score']:.4f} REL {out['log']['REL']:.4f} | caller {c['score']:.4f} UNC {c['UNC']:.4f} "
                  f"RES {c['RES']:.4f} REL {c['REL']:.4f}({c['REL_noise']:.4f}) iso {c.get('iso_cv', 0):.4f}", file=sys.stderr)
    json.dump(dict(rows=rows, curves={k: curves(s) for k, s in scores.items()}), sys.stdout, separators=(",", ":"))
