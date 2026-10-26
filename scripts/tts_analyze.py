"""How correlated are the confidence heads, and does a learned combination of them pick better answers from k samples?

Input: tts_score.py's scores.jsonl (k rollouts per question, each with every model's 11-level distribution).
1. Correlation of the models' expected confidences (renormalised over the listed mass) across rollouts, overall and
   within question (each question's mean removed), plus each one's AUROC for the rollout's correctness.
2. Answer selection. Rollouts are clustered by normalised answer; a question is solved if the picked cluster's answer
   is correct. Fixed rules: first sample, majority vote, and for each model the max-confidence rollout and the
   confidence-weighted vote (sum of confidence per cluster). Learned rules: a cluster-level scorer over features
   (cluster share of the k votes, mean and max of each model's confidence in the cluster, mean sampled log-prob)
   with label "cluster answer correct", picked by argmax. It is fit by progressive validation: questions in a fixed
   shuffled order, each block of --block questions scored by a model fit on all questions before it (the first block
   by majority vote), so every number is out-of-sample. Logistic regression and LightGBM.

usage: tts_analyze.py SCORES.jsonl [--block 100] [--seed 0] [--json OUT.json]
"""
import argparse, collections, json, random, warnings

import lightgbm as lgb
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("scores", nargs="+", help="tts_score.py outputs over the same rollouts; their models are merged")
ap.add_argument("--block", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--json", default=None)
a = ap.parse_args()

GRID = np.linspace(0, 1, 11)
expect = lambda p: float(np.dot(p, GRID) / sum(p)) if p and sum(p) > 1e-9 else np.nan
rows = [json.loads(l) for l in open(a.scores[0])]
for path in a.scores[1:]:  # same rollouts in the same order; later files add models (a repeated "base" is the same)
    for r, extra in zip(rows, map(json.loads, open(path))):
        assert (r["id"], r["j"]) == (extra["id"], extra["j"])
        r["conf"] |= extra["conf"]; r["prior"] = r.get("prior", {}) | extra.get("prior", {})
models = list(rows[0]["conf"])
priors = [f"prior:{m}" for m in rows[0].get("prior", {})]
for r in rows:
    for m in models:
        r[m] = expect(r["conf"][m])
    for m in priors:
        r[m] = expect(r["prior"][m.removeprefix("prior:")])
qs = collections.defaultdict(list)
for r in rows:
    qs[r["id"]].append(r)
qids = sorted(qs); random.Random(a.seed).shuffle(qids)
k = max(len(v) for v in qs.values())
out = {"questions": len(qids), "k": k, "models": models}
print(f"{len(qids)} questions x {k} rollouts; rollout accuracy {np.mean([r['correct'] for r in rows]):.3f}; "
      f"pass@{k} {np.mean([max(r['correct'] for r in qs[q]) for q in qids]):.3f}")

# 1. correlations and AUROC
X = np.array([[r[m] for m in models] for r in rows]); y = np.array([r["correct"] for r in rows])
ok = ~np.isnan(X).any(1)
Xw = X.copy()  # within-question: subtract each question's mean (over its scored rollouts)
pos = collections.defaultdict(list)
for i, r in enumerate(rows):
    if ok[i]: pos[r["id"]].append(i)
for ii in pos.values():
    Xw[ii] -= X[ii].mean(0)
agree = np.array([r["agree"][0] / max(r["agree"][1], 1) for r in rows])
feats = models + ["agree", "mean_logprob"]
F = np.column_stack([X, agree, [r["mean_logprob"] for r in rows]])
print(f"\nscored rollouts {ok.sum()} / {len(rows)}")
print("AUROC (rollout correct):", "  ".join(f"{f} {roc_auc_score(y[ok], F[ok, i]):.3f}" for i, f in enumerate(feats)))
out["auroc"] = {f: roc_auc_score(y[ok], F[ok, i]) for i, f in enumerate(feats)}
for name, M in (("spearman, overall", F[ok]), ("pearson, within question", np.column_stack([Xw, agree])[ok])):
    C = spearmanr(M).statistic if name.startswith("spearman") else np.corrcoef(M.T)
    names = feats if name.startswith("spearman") else models + ["agree"]
    print(f"\n{name}\n" + " " * 22 + "".join(f"{n[:10]:>11}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<22}" + "".join(f"{C[i, j]:11.2f}" for j in range(len(names))))
    out[name] = {"names": names, "matrix": np.round(C, 4).tolist()}

# question level: the pre-answer difficulty heads against what the k samples show (solve rate, majority share,
# mean stated confidence)
Q = []
for q in qids:
    rs = qs[q]; votes = collections.Counter(r["answer"] for r in rs if r["answer"])
    Q.append([np.mean([r["correct"] for r in rs]), max(votes.values()) / len(rs) if votes else 0.0]
             + [np.nanmean([r[m] for r in rs]) if not all(np.isnan(r[m]) for r in rs) else np.nan for m in models]
             + [rs[0][m] for m in priors])
Q = np.array(Q); qnames = ["solve_rate", "majority_share"] + [f"mean:{m}" for m in models] + priors
qok = ~np.isnan(Q).any(1)
C = spearmanr(Q[qok]).statistic
print(f"\nspearman, question level ({qok.sum()} questions)\n" + " " * 34 + "".join(f"{n[-10:]:>11}" for n in qnames))
for i, n in enumerate(qnames):
    print(f"{n:<34}" + "".join(f"{C[i, j]:11.2f}" for j in range(len(qnames))))
out["spearman, question level"] = {"names": qnames, "matrix": np.round(C, 4).tolist()}
solved_maj = Q[:, 0] >= 0.5
print("question AUROC for solve rate >= 0.5:", "  ".join(f"{n} {roc_auc_score(solved_maj[qok], Q[qok, i]):.3f}"
                                                      for i, n in enumerate(qnames) if i))


# 2. answer selection
def clusters(rs):
    c = collections.defaultdict(list)
    for r in rs:
        c[r["answer"]].append(r)
    return c


def cluster_feats(rs):
    """[(answer, correct, features)] per cluster of one question's rollouts (empty answers are never picked)."""
    out = []
    for ans, cr in clusters(rs).items():
        if not ans:
            continue
        f = [len(cr) / len(rs)]
        for m in models:
            v = [r[m] for r in cr if not np.isnan(r[m])]
            f += [np.mean(v) if v else 0.0, max(v) if v else 0.0, np.sum(v) / len(rs) if v else 0.0]
        f.append(np.mean([r["mean_logprob"] for r in cr]))
        out.append((ans, cr[0]["correct"], f))
    return out


FEAT_NAMES = ["share"] + [f"{m}_{s}" for m in models for s in ("mean", "max", "wsum")] + ["logprob"]
CF = {q: cluster_feats(qs[q]) for q in qids}


def solved(q, scores):
    c = CF[q]
    return float(c[int(np.argmax(scores))][1]) if c else 0.0


rules = {"first sample": lambda q: float(qs[q][0]["correct"]) if qs[q][0]["answer"] else 0.0,
         "majority vote": lambda q: solved(q, [f[0] + 1e-6 * f[-1] for _, _, f in CF[q]])}
for i, m in enumerate(models):
    rules[f"max conf: {m}"] = lambda q, i=i: solved(q, [f[2 + 3 * i] for _, _, f in CF[q]])
    rules[f"conf vote: {m}"] = lambda q, i=i: solved(q, [f[3 + 3 * i] for _, _, f in CF[q]])
res = {n: np.mean([fn(q) for q in qids]) for n, fn in rules.items()}


def progressive(make, cols=None):
    hits = []
    for b in range(0, len(qids), a.block):
        seen, block = qids[:b], qids[b: b + a.block]
        Xtr = [f for q in seen for _, _, f in CF[q]]; ytr = [c for q in seen for _, c, _ in CF[q]]
        fit = None
        if len(set(ytr)) > 1:
            Xtr = np.array(Xtr)[:, cols] if cols is not None else np.array(Xtr)
            fit = make().fit(Xtr, ytr)
        for q in block:
            if not CF[q]:
                hits.append(0.0); continue
            Xq = np.array([f for _, _, f in CF[q]])
            s = fit.predict_proba(Xq[:, cols] if cols is not None else Xq)[:, 1] if fit else Xq[:, 0]
            hits.append(solved(q, s))
    return np.mean(hits), hits[len(hits) // 2:]


logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
gbdt = lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20, verbose=-1)
share_lp = [0, len(FEAT_NAMES) - 1]
learned = {"logistic: share + logprob": (logit, share_lp), "logistic: all": (logit, None), "gbdt: all": (gbdt, None)}
for m in models:
    i = 1 + 3 * models.index(m)
    learned[f"logistic: share + {m}"] = (logit, [0, i, i + 1, i + 2])
second_half = {}
for n, (mk, cols) in learned.items():
    res[n], second_half[n] = progressive(mk, cols)
half = qids[len(qids) // 2:]
print(f"\nanswer accuracy over {len(qids)} questions (learned: progressive validation, blocks of {a.block}; "
      f"last column: second half only, where the learners have seen >= {len(qids) // 2} questions)")
mv_half = np.mean([rules["majority vote"](q) for q in half])
for n, v in sorted(res.items(), key=lambda kv: -kv[1]):
    h = np.mean(second_half[n]) if n in second_half else np.mean([rules[n](q) for q in half])
    print(f"  {n:<36}{v:.3f}   {h:.3f} ({h - mv_half:+.3f} vs vote)")
out["selection"] = res
out["selection_second_half"] = {n: float(np.mean(second_half[n]) if n in second_half else np.mean([rules[n](q) for q in half])) for n in res}
fit = gbdt().fit(np.array([f for q in qids for _, _, f in CF[q]]), [c for q in qids for _, c, _ in CF[q]])
imp = sorted(zip(FEAT_NAMES, fit.booster_.feature_importance("gain")), key=lambda t: -t[1])
print("\ngbdt gain (all questions):", ", ".join(f"{n} {g / max(sum(g for _, g in imp), 1e-9):.2f}" for n, g in imp[:8]))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1, default=float)
