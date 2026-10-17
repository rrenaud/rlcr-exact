"""Does the confidence LoRA change the answers? Compare greedy generations with the adapter on against the base
model's cached greedy rollouts of the same questions (both from rollout_vllm.py, k=1, temperature 0).

The LoRA is trained only on confidence tokens, but at generation time it is active on every token, so it can move
the reasoning and the answer. Reports, per split and for both policies: format rate, accuracy (RLCR's correctness
label; zero when the format check fails), confidence parse rate, mean confidence, Brier/ECE/AUROC of the generated
confidence, plus the fraction of questions whose final <answer> string changed and the accuracy change.
Format failures here are almost all a junk token after </confidence> instead of end of text, so accuracy mixes
answer quality with stopping. The answer itself is em_any (exact match of the last <answer>, format ignored);
stop is the rate of completions that end right after </confidence>. Judge answer damage on em_any.

usage: gen_eval.py --base BASE_DIR --lora LORA_DIR [--splits dev test] [--out JSON]
"""
import argparse, json, os, re

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rlcr_exact.grading import em_correct
from sklearn.metrics import roc_auc_score


def last(tag, text):
    found = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return found[-1].strip() if found else None


def confidence(text):
    try:
        c = float(last("confidence", text))
    except (TypeError, ValueError):
        return None
    return c if 0 <= c <= 1 else None


def ece(c, y, bins=10):
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in range(bins) if (b == k).any()))


def load(path):
    out = {}
    for q in map(json.loads, open(path)):
        t = q["outputs"][0]
        out[q["id"]] = dict(text=t, correct=float(q["correct"][0]), format_ok=float(q["format_ok"][0]),
                            truncated=bool(q["truncated"][0]), em_any=em_correct(t, q["answer"]),
                            stop=float("</confidence>" in t and t.split("</confidence>")[-1].strip() == ""))
    return out


def metrics(rows):
    conf = [confidence(r["text"]) for r in rows]
    c = np.array([x for x in conf if x is not None])
    y = np.array([r["correct"] for r, x in zip(rows, conf) if x is not None])
    return dict(n=len(rows), em_any=np.mean([r["em_any"] for r in rows]), stop=np.mean([r["stop"] for r in rows]), format_ok=np.mean([r["format_ok"] for r in rows]), accuracy=np.mean([r["correct"] for r in rows]),
                truncated=np.mean([r["truncated"] for r in rows]), conf_parsed=len(c) / len(rows), mean_conf=c.mean(),
                mean_conf_correct=c[y == 1].mean() if (y == 1).any() else None, mean_conf_wrong=c[y == 0].mean() if (y == 0).any() else None,
                brier=np.mean((c - y) ** 2), ece=ece(c, y), auroc=roc_auc_score(y, c) if 0 < y.mean() < 1 else None)


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="dir with rollouts_hotpot_fresh_{split}.jsonl of the base policy")
ap.add_argument("--lora", required=True, help="same, generated with the adapter")
ap.add_argument("--splits", nargs="+", default=["dev", "test"])
ap.add_argument("--out", default=None)
a = ap.parse_args()
report = {}
for split in a.splits:
    lora = load(os.path.join(a.lora, f"rollouts_hotpot_fresh_{split}.jsonl"))
    base = load(os.path.join(a.base, f"rollouts_hotpot_fresh_{split}.jsonl"))
    ids = sorted(set(lora) & set(base))
    b, l = metrics([base[i] for i in ids]), metrics([lora[i] for i in ids])
    report[split] = dict(base=b, lora=l, questions=len(ids),
                         answer_changed=np.mean([last("answer", base[i]["text"]) != last("answer", lora[i]["text"]) for i in ids]),
                         correct_flipped_to_wrong=np.mean([base[i]["correct"] > lora[i]["correct"] for i in ids]),
                         wrong_flipped_to_correct=np.mean([base[i]["correct"] < lora[i]["correct"] for i in ids]),
                         accuracy_delta=l["accuracy"] - b["accuracy"], format_delta=l["format_ok"] - b["format_ok"],
                         em_any_delta=l["em_any"] - b["em_any"], stop_delta=l["stop"] - b["stop"])
report = json.loads(json.dumps(report, default=float))
for split, r in report.items():
    print(f"== {split} ({r['questions']} questions): answer EM (format ignored) {r['base']['em_any']:.3f} -> {r['lora']['em_any']:.3f} "
          f"(delta {r['em_any_delta']:+.3f}), stop after </confidence> {r['base']['stop']:.3f} -> {r['lora']['stop']:.3f}")
    print(f"   accuracy {r['base']['accuracy']:.3f} -> {r['lora']['accuracy']:.3f} "
          f"(delta {r['accuracy_delta']:+.3f}), format {r['base']['format_ok']:.3f} -> {r['lora']['format_ok']:.3f}, "
          f"answer changed {r['answer_changed']:.3f} (right->wrong {r['correct_flipped_to_wrong']:.3f}, wrong->right {r['wrong_flipped_to_correct']:.3f})")
    for side in ("base", "lora"):
        m = r[side]
        print(f"   {side:4s} conf {m['mean_conf']:.3f} (correct {m['mean_conf_correct'] or float('nan'):.3f} / wrong {m['mean_conf_wrong'] or float('nan'):.3f})  "
              f"Brier {m['brier']:.3f}  ECE {m['ece']:.3f}  AUROC {m['auroc'] or float('nan'):.3f}  parsed {m['conf_parsed']:.3f}  truncated {m['truncated']:.3f}")
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
