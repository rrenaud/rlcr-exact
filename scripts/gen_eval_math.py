"""Math generation check, comparison side: does the confidence LoRA keep the answers and calibrate the generated
confidence? Compares gen_math.py rollouts with the adapter on against the base model under the same system prompt
(both greedy, same questions).

Per policy: format-free accuracy (`correct`, the training label), RLCR's strict accuracy, </answer> rate, stop rate
(nothing after </confidence>), truncation, confidence parse rate, mean confidence, Brier / ECE / AUROC of the
generated confidence against `correct` (parsed rows), and Brier with an unparsed confidence counted as 0. Paired: the
fraction of questions whose first <answer> changed, right->wrong / wrong->right flips, and a question-level bootstrap
95% CI of the accuracy change.

usage: gen_eval_math.py --base BASE_DIR --lora LORA_DIR [--splits dev test] [--out JSON]   (CPU)
reads DIR/gen_math_{split}.jsonl
"""
import argparse, json, os

import numpy as np
from sklearn.metrics import roc_auc_score


def ece(c, y, bins=10):
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in range(bins) if (b == k).any()))


def metrics(rows):
    y_all = np.array([r["correct"] for r in rows])
    p = [r for r in rows if r["confidence"] is not None]
    c = np.array([r["confidence"] for r in p]); y = np.array([r["correct"] for r in p])
    c0 = np.array([r["confidence"] if r["confidence"] is not None else 0.0 for r in rows])
    return dict(n=len(rows), accuracy=y_all.mean(), strict=np.mean([r["strict"] for r in rows]),
                has_answer=np.mean([r["has_answer"] for r in rows]), stop=np.mean([r["stop"] for r in rows]),
                truncated=np.mean([r["truncated"] for r in rows]), gen_tokens=np.mean([r["gen_tokens"] for r in rows]),
                conf_parsed=len(p) / len(rows), mean_conf=c.mean() if len(c) else None,
                mean_conf_correct=c[y == 1].mean() if (y == 1).any() else None,
                mean_conf_wrong=c[y == 0].mean() if (y == 0).any() else None,
                distinct_conf=len(set(np.round(c, 3).tolist())),
                brier=np.mean((c - y) ** 2) if len(c) else None, brier_missing_0=np.mean((c0 - y_all) ** 2),
                ece=ece(c, y) if len(c) else None,
                auroc=roc_auc_score(y, c) if len(c) and 0 < y.mean() < 1 and len(set(c.tolist())) > 1 else None)


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--splits", nargs="+", default=["dev", "test"])
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)
report = {}
for split in a.splits:
    load = lambda d: {r["id"]: r for r in map(json.loads, open(os.path.join(d, f"gen_math_{split}.jsonl")))}
    base, lora = load(a.base), load(a.lora)
    ids = sorted(set(base) & set(lora))
    b, l = metrics([base[i] for i in ids]), metrics([lora[i] for i in ids])
    d = np.array([lora[i]["correct"] - base[i]["correct"] for i in ids])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)]
    report[split] = dict(base=b, lora=l, questions=len(ids),
                         answer_changed=np.mean([base[i]["pred"] != lora[i]["pred"] for i in ids]),
                         right_to_wrong=np.mean(d < 0), wrong_to_right=np.mean(d > 0),
                         accuracy_delta=d.mean(), accuracy_delta_ci=[np.percentile(boot, 2.5), np.percentile(boot, 97.5)],
                         stop_delta=l["stop"] - b["stop"], strict_delta=l["strict"] - b["strict"])
report = json.loads(json.dumps(report, default=float))
f = lambda x: "nan" if x is None else f"{x:.3f}"
for split, r in report.items():
    lo, hi = r["accuracy_delta_ci"]
    print(f"== {split} ({r['questions']} questions): accuracy (format-free) {f(r['base']['accuracy'])} -> {f(r['lora']['accuracy'])} "
          f"(delta {r['accuracy_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}]); answer changed {f(r['answer_changed'])} "
          f"(right->wrong {f(r['right_to_wrong'])}, wrong->right {f(r['wrong_to_right'])})")
    print(f"   strict {f(r['base']['strict'])} -> {f(r['lora']['strict'])}, stop {f(r['base']['stop'])} -> {f(r['lora']['stop'])}, "
          f"truncated {f(r['base']['truncated'])} -> {f(r['lora']['truncated'])}")
    for side in ("base", "lora"):
        m = r[side]
        print(f"   {side:4s} conf {f(m['mean_conf'])} (correct {f(m['mean_conf_correct'])} / wrong {f(m['mean_conf_wrong'])}, "
              f"{m['distinct_conf']} values)  Brier {f(m['brier'])} (missing=0: {f(m['brier_missing_0'])})  ECE {f(m['ece'])}  "
              f"AUROC {f(m['auroc'])}  parsed {f(m['conf_parsed'])}")
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
