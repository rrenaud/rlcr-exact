"""Compare prompts from the math prompt probe (score_analysis.py outputs), per prompt and split.

Per row, from the trie log-probs (score_analysis.py): value mass (the 11 values through `</confidence`), q over the
11 values (renormalized), entropy, expected confidence, legal mass (value + closing + end token, i.e. well-formed
and stopped), p(stop) = legal / value mass, and the exact-objective gradient size sum_j |q_j (R_j - J)| with RLCR's reward (off-list mass earns
0). Per prompt: compliance (confidence-like numbers in the analysis), stated/expected Brier and AUROC, a linear
probe's AUROC on the tag hidden state (5-fold, grouped by problem, dev + train pooled), and for `range` the endpoint /
midpoint / within-range rates. Dev AUROCs get a problem-level paired bootstrap against control.

usage: analyze_prompts.py RUN_DIR [--prompts a,b]   (CPU)
"""
import argparse, collections, json, math, os, re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()

CONFNUM = re.compile(r"(confiden|probab|likel|certain|sure|chance)[^.\n]{0,40}?(\d{1,3}\s?%|\b0?\.\d+\b|\b1\.0\b)"
                     r"|\b\d{1,3}\s?%", re.I)
RANGE = re.compile(r"Plausible confidence range:\s*([01]\.\d)\s*to\s*([01]\.\d)")
STATED = re.compile(r"<confidence>\s*([0-9.]+)\s*</confidence>\s*$")


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 and len(set(np.round(s, 6))) > 1 else float("nan")


rows, H = [], []
for split in ("dev", "train"):
    p = os.path.join(a.run, f"scored_{split}")
    if not os.path.exists(p + ".jsonl"):
        continue
    cand = json.load(open(p + ".candidates.json"))
    vals, closes, ends = np.array(cand["values"]), cand["closes"], cand["ends"]
    stem = np.array([c is None for c in closes])  # each value through `confidence`
    with_end = np.array([e is not None for e in ends])  # value, closing, end token: well-formed and stopped
    grid = np.round(np.linspace(0, 1, 11), 1)
    hid = np.load(p + ".hidden.npy")
    for k, line in enumerate(open(p + ".jsonl")):
        r = json.loads(line)
        an = re.findall(r"<analysis>(.*?)</analysis>", r["gen"], re.S)
        an = an[-1] if an else ""
        m = STATED.search(r["gen"].strip())
        stated = float(m.group(1)) if m and re.fullmatch(r"[01]\.\d", m.group(1)) else None
        rg = RANGE.search(an)
        body = RANGE.sub("", an)
        r.update(an_len=len(an.split()), conf_numbers=len(CONFNUM.findall(body)), stated=stated,
                 range=(float(rg.group(1)), float(rg.group(2))) if rg else None, hidden_row=len(H))
        H.append(hid[k])
        if r.get("has_tag"):
            lp = np.array(r["logps"]); p_all = np.exp(lp)
            pv = np.array([p_all[stem & (vals == g)].sum() for g in grid])  # p(value), whatever follows
            pe = p_all[with_end]; ve = vals[with_end]
            legal = pe.sum()
            q = pv / max(pv.sum(), 1e-12)
            y = r["correct"]
            R = 0.5 * (1 + y + 1 - (y - ve) ** 2)
            J = (pe * R).sum()
            qq = q[q > 0]
            r.update(legal=legal, value_mass=float(pv.sum()), q=q, ent=float(-(qq * np.log(qq)).sum()),
                     exp_conf=float((q * grid).sum()), p_end=float(legal / max(pv.sum(), 1e-12)),
                     grad=float(np.abs(pe * (R - J)).sum() + (1 - legal) * J), exp_brier=float((q * (grid - y) ** 2).sum()))
        rows.append(r)
H = np.stack(H).astype(np.float32)

by = collections.defaultdict(list)
for r in rows:
    by[r["prompt"]].append(r)


def probe_scores(rs):
    """Out-of-fold P(correct) from the tag hidden state; 5 folds grouped by problem."""
    rs = [r for r in rs if r.get("has_tag")]
    X = H[[r["hidden_row"] for r in rs]]; y = np.array([r["correct"] for r in rs]); g = [r["id"] for r in rs]
    out = np.zeros(len(rs))
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return {(r["split"], r["id"], r["j"]): s for r, s in zip(rs, out)}


summary, dev_scores = {}, {}
for name, rs in by.items():
    probe = probe_scores(rs)
    for split in ("dev", "train"):
        s = [r for r in rs if r["split"] == split]
        if not s:
            continue
        t = [r for r in s if r.get("has_tag")]
        y = np.array([r["correct"] for r in t])
        st = [r for r in t if r["stated"] is not None]
        ys = np.array([r["correct"] for r in st]); cs = np.array([r["stated"] for r in st])
        e = np.array([r["exp_conf"] for r in t]); pr = np.array([probe[(r["split"], r["id"], r["j"])] for r in t])
        d = dict(rows=len(s), tag=len(t) / len(s), stated_ok=len(st) / len(s), acc=float(y.mean()),
                 value_mass=float(np.mean([r["value_mass"] for r in t])),
                 legal=float(np.mean([r["legal"] for r in t])), p_end=float(np.mean([r["p_end"] for r in t])),
                 ent=float(np.mean([r["ent"] for r in t])), ent_gt_0_1=float(np.mean([r["ent"] > 0.1 for r in t])),
                 grad=float(np.mean([r["grad"] for r in t])), mean_stated=float(cs.mean()) if len(cs) else float("nan"),
                 distinct_stated=len(set(cs.tolist())), brier_stated=float(((cs - ys) ** 2).mean()) if len(cs) else float("nan"),
                 brier_exp=float(((e - y) ** 2).mean()), exp_brier=float(np.mean([r["exp_brier"] for r in t])),
                 auroc_stated=auroc(ys, cs) if len(cs) else float("nan"), auroc_exp=auroc(y, e), auroc_probe=auroc(y, pr),
                 conf_numbers=float(np.mean([r["conf_numbers"] > 0 for r in s])), an_words=float(np.mean([r["an_len"] for r in s])),
                 gen_tokens=float(np.mean([r["gen_tokens"] for r in s])), hit_cap=float(np.mean([r["finish"] == "length" for r in s])))
        if name == "range":
            rr = [r for r in st if r["range"]]
            d.update(range_line=len([r for r in s if r["range"]]) / len(s),
                     range_width=float(np.mean([r["range"][1] - r["range"][0] for r in rr])) if rr else float("nan"),
                     within=float(np.mean([r["range"][0] - 1e-9 <= r["stated"] <= r["range"][1] + 1e-9 for r in rr])) if rr else float("nan"),
                     endpoint=float(np.mean([r["stated"] in r["range"] for r in rr])) if rr else float("nan"),
                     upper=float(np.mean([abs(r["stated"] - r["range"][1]) < 1e-9 for r in rr])) if rr else float("nan"),
                     midpoint=float(np.mean([abs(r["stated"] - round(sum(r["range"]) / 2, 1)) < 1e-9 for r in rr])) if rr else float("nan"))
        summary[(name, split)] = d
        if split == "dev":
            dev_scores[name] = {r["id"]: (r["correct"], r["exp_conf"], probe[(r["split"], r["id"], r["j"])]) for r in t}

rng = np.random.default_rng(0)
if "control" in dev_scores:
    for name in dev_scores:
        ids = sorted(set(dev_scores[name]) & set(dev_scores["control"]))
        diffs = {"exp": [], "probe": []}
        for _ in range(a.boot):
            b = rng.choice(ids, len(ids))
            for key, col in (("exp", 1), ("probe", 2)):
                yy = [dev_scores[name][i][0] for i in b]
                diffs[key].append(auroc(yy, [dev_scores[name][i][col] for i in b]) - auroc(yy, [dev_scores["control"][i][col] for i in b]))
        for key in diffs:
            v = np.array(diffs[key]); v = v[~np.isnan(v)]
            summary[(name, "dev")][f"d_auroc_{key}_vs_control"] = (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

cols = ["rows", "tag", "value_mass", "legal", "p_end", "ent", "ent_gt_0_1", "grad", "mean_stated", "distinct_stated", "brier_stated",
        "exp_brier", "auroc_stated", "auroc_exp", "auroc_probe", "conf_numbers", "an_words", "hit_cap"]
for split in ("dev", "train"):
    print(f"\n== {split}")
    print("prompt".ljust(12) + "".join(c[:11].rjust(12) for c in cols))
    for name in by:
        d = summary.get((name, split))
        if d:
            print(name.ljust(12) + "".join((f"{d[c]:.3f}" if isinstance(d[c], float) else str(d[c])).rjust(12) for c in cols))
print("\n== dev paired bootstrap vs control: mean [95% CI]")
for name in by:
    d = summary.get((name, "dev"), {})
    if "d_auroc_exp_vs_control" in d:
        print(name.ljust(12), " ".join(f"{k}: {d[f'd_auroc_{k}_vs_control'][0]:+.3f} [{d[f'd_auroc_{k}_vs_control'][1]:+.3f}, {d[f'd_auroc_{k}_vs_control'][2]:+.3f}]" for k in ("exp", "probe")))
if ("range", "dev") in summary:
    print("\n== range:", {k: round(v, 3) for k, v in summary[("range", "dev")].items() if k in ("range_line", "range_width", "within", "endpoint", "upper", "midpoint")})
json.dump({f"{n}|{s}": d for (n, s), d in summary.items()}, open(os.path.join(a.run, "prompt_summary.json"), "w"), indent=1)
