"""Is a model plastic at its confidence tag? For cached rollouts (cache_rollouts.py schema):
format rate, accuracy, Brier / AUROC of the stated confidence, the probability of the sampled confidence string
(product of its token probs between <confidence> and </confidence>; for greedy rows, the top string's
probability, so 1 - p bounds the unselected mass from below), and how often <analysis> already states the
tag's value (as 0.d, .d or d0%), the copying that made RLCR-math's tag a deterministic readout. If the top-64
sidecar (ROLLOUTS.topk.jsonl.gz) exists, also the entropy at the confidence span's most uncertain token (top-64
plus a rest bucket, nats) and that token's unselected mass 1 - sum p^2.

usage: tag_spread.py MODEL ROLLOUTS_JSONL [...]   (CPU; MODEL gives the tokenizer)
"""
import gzip, json, math, os, re, sys

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])


def spans(ids):
    """Char offset of each token start in the decoded text (byte-level pieces, concatenated)."""
    starts, text = [], ""
    for t in ids:
        starts.append(len(text))
        text += tok.decode([t])
    return starts, text


def stated(analysis, c):
    forms = {f"{c:.1f}", f"{c:.2f}".rstrip("0")} | ({f"{round(100 * c)}%", f"{round(100 * c)} %"} if c > 0 else set())
    forms |= {f.lstrip("0") for f in forms if f.startswith("0.")}
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d])", analysis) for f in forms)


def span_spread(path):
    """(id, j) -> (max entropy, max unselected mass) over the confidence span's tokens."""
    out = {}
    if not os.path.exists(path + ".topk.jsonl.gz"):
        return out
    for line in gzip.open(path + ".topk.jsonl.gz", "rt"):
        d = json.loads(line)
        ent = uns = 0.0
        for lps in (d["confidence"] or {}).get("top_logprobs", []):
            p = np.exp(np.array(lps, float))
            rest = max(0.0, 1 - p.sum())
            ps = np.append(p, rest)
            ps = ps[ps > 0]
            ent = max(ent, float(-(ps * np.log(ps)).sum()))
            uns = max(uns, float(1 - (p ** 2).sum()))
        out[(d["id"], d["j"])] = (ent, uns)
    return out


for path in sys.argv[2:]:
    spread = span_spread(path)
    ents, unss = [], []
    n = fmt = 0
    conf, y, ptag, copied = [], [], [], []
    for line in open(path):
        q = json.loads(line)
        for j, ids in enumerate(q["completion_ids"]):
            n += 1
            starts, text = spans(ids)
            m = list(re.finditer(r"<confidence>\s*([0-9.]+)\s*</confidence>", text))
            if not m:
                continue
            try:
                c = float(m[-1].group(1))
            except ValueError:
                continue
            if not 0 <= c <= 1:
                continue
            fmt += 1
            a, b = m[-1].start(1), m[-1].end(1)
            lp = [q["sampled_logprobs"][j][i] for i, s in enumerate(starts)
                  if s < b and s + len(tok.decode([ids[i]])) > a]
            conf.append(c); y.append(q["correct"][j]); ptag.append(math.exp(sum(lp)))
            an = re.findall(r"<analysis>(.*?)</analysis>", text, re.S)
            copied.append(bool(an) and stated(an[-1], c))
            if (q["id"], j) in spread:
                ents.append(spread[(q["id"], j)][0]); unss.append(spread[(q["id"], j)][1])
    c, y, p = np.array(conf), np.array(y, float), np.array(ptag)
    print(json.dumps(dict(
        file=path.rsplit("/", 1)[-1], rollouts=n, format_ok=round(fmt / max(n, 1), 3), accuracy=round(y.mean(), 3),
        mean_conf=round(c.mean(), 3), conf_sd=round(c.std(), 3), distinct_conf=len(set(c.tolist())),
        brier=round(((c - y) ** 2).mean(), 3), auroc=round(roc_auc_score(y, c), 3) if 0 < y.mean() < 1 else None,
        p_tag_mean=round(p.mean(), 3), p_tag_median=round(float(np.median(p)), 3),
        frac_p_tag_below_0_9=round((p < 0.9).mean(), 3), frac_p_tag_above_0_999=round((p > 0.999).mean(), 3),
        analysis_states_value=round(np.mean(copied), 3),
        **(dict(tag_entropy_mean=round(np.mean(ents), 3), tag_entropy_median=round(float(np.median(ents)), 3),
                tag_unselected_mean=round(np.mean(unss), 3), frac_entropy_above_0_1=round(np.mean(np.array(ents) > 0.1), 3))
           if ents else {}))))
