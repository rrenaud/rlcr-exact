"""Does the stop after the number depend on the number? p(suffix | v) per confidence value, within rows.

For each cached row (well-formed and tail-fault) and each canonical spaced value v (0.05 grid plus 1.0), score
`Ġ v Ġ</ confidence > <|endoftext|>` through one trie and split log p into the number, the close `Ġ</`, and
`confidence > eos`. Reports the per-value stop probability s(v), its spread across v within a row, and the
error in J = sum_v q(v) s(v) R(v) from replacing s(v) by one shared s (fixed v* = 0.5, or the q-weighted mean).

usage: stop_by_value.py ROLLOUTS_JSONL [--adapter DIR ...] [--rows N]
Writes $GPUQ_RUN_DIR/stop_by_value.json.
"""
import argparse, json, os, sys

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_frozen import load_rows, make_token_bytes
from rlcr_exact.trie_score import build_trie, candidate_logps

ap = argparse.ArgumentParser()
ap.add_argument("rollouts")
ap.add_argument("--model", default="mehuldamani/hotpot-v2-correctness-7b")
ap.add_argument("--adapter", action="append", default=[])
ap.add_argument("--rows", type=int, default=500)
ap.add_argument("--out", default=os.environ.get("GPUQ_RUN_DIR", "."))
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tid = tok.convert_tokens_to_ids
texts = [f"0.{d}" for d in range(10)] + [f"0.{d}5" for d in range(10)] + ["1.0"]
values = np.array([float(t) for t in texts])
close, rest = [tid("Ġ</")], [tid("confidence"), tid(">"), tid("<|endoftext|>")]
seqs = []
for t in texts:  # per value: number, + close, + confidence, + >, + eos (prefix paths share the trie's nodes)
    num = [tid("Ġ")] + [tid(c) for c in t]
    full = num + close + rest
    seqs += [full[: len(num) + k] for k in range(5)]
trie = build_trie(seqs)
V = len(texts)

rows = load_rows(a.rollouts, make_token_bytes(tok), 4608, tail_ok=True)[0][: a.rows]
faulted = set()
for q in map(json.loads, open(a.rollouts)):
    faulted |= {(q["id"], j) for j, ok in enumerate(q["format_ok"]) if not ok}

base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()


def analyse(model):
    lp = []  # [rows, V, 5] cumulative log p: number, +close, +confidence, +>, +eos
    with torch.no_grad():
        for r in rows:
            lp.append(candidate_logps(model, r.prefix, trie).float().cpu().numpy().reshape(V, 5))
    lp = np.stack(lp)
    y = np.array([r.correct for r in rows], dtype=float)
    fault = np.array([(r.qid, r.j) in faulted for r in rows])
    step = np.diff(lp, axis=2)  # [rows, V, 4] log p of close, confidence, >, eos given what precedes
    s = np.exp(lp[:, :, 4] - lp[:, :, 0])  # p(Ġ</ confidence > eos | v)
    s_after_close = np.exp(lp[:, :, 4] - lp[:, :, 1])  # p(confidence > eos | v, Ġ</)
    qn = np.exp(lp[:, :, 0]); qn = qn / qn.sum(1, keepdims=True)  # number distribution over the grid
    R = 0.5 * (1 + y[:, None] + 1 - (y[:, None] - values[None]) ** 2)
    J = (qn * s * R).sum(1)
    s_bar = (qn * s).sum(1, keepdims=True)
    s_fix = s[:, [texts.index("0.5")]]
    s_fix_close = s_after_close[:, [texts.index("0.5")]] * np.exp(step[:, :, 0])  # close exact, rest from 0.5
    errs = {name: np.abs((qn * sh * R).sum(1) - J) / J for name, sh in
            (("qmean", s_bar), ("fixed_0.5", s_fix), ("close_exact_rest_0.5", s_fix_close))}
    # Gradient direction wrt the number logits: d J / d logit_v = q_v (g_v - J), g_v = s_v R_v. Cosine exact vs shared s.
    def grad(sh):
        g = sh * R; Jh = (qn * g).sum(1, keepdims=True); return qn * (g - Jh)
    ge = grad(s)
    cos = {name: float(np.median((ge * grad(sh)).sum(1) / (np.linalg.norm(ge, axis=1) * np.linalg.norm(grad(sh), axis=1) + 1e-12)))
           for name, sh in (("qmean", s_bar), ("fixed_0.5", s_fix), ("close_exact_rest_0.5", s_fix_close))}

    def summ(m):
        sw = s[m]; qw = qn[m]
        within_sd = np.sqrt((qw * (sw - (qw * sw).sum(1, keepdims=True)) ** 2).sum(1))  # q-weighted sd across v
        return dict(rows=int(m.sum()), stop_mean=float((qw * sw).sum(1).mean()),
                    within_row_sd_qweighted_median=float(np.median(within_sd)),
                    within_row_range_median=float(np.median(sw.max(1) - sw.min(1))),
                    between_row_sd=float((qw * sw).sum(1).std()),
                    factor_means={k: float(np.exp((qw[..., None] * step[m])[:, :, i].sum(1)).mean())
                                  for i, k in enumerate(["close", "confidence", ">", "eos"])},
                    rel_J_error_median={k: float(np.median(e[m])) for k, e in errs.items()},
                    rel_J_error_p90={k: float(np.quantile(e[m], 0.9)) for k, e in errs.items()})
    return dict(all=summ(np.ones(len(rows), bool)), well_formed=summ(~fault), tail_fault=summ(fault),
                per_value_stop_mean={t: float(s[:, i].mean()) for i, t in enumerate(texts)},
                per_value_close_mean={t: float(np.exp(step[:, i, 0]).mean()) for i, t in enumerate(texts)},
                grad_cosine_median=cos)


report = dict(model=a.model, rows=len(rows), values=texts, trie_nodes=len(trie.tokens), results={})
report["results"]["base"] = analyse(base)
print(json.dumps(report["results"]["base"], indent=1), flush=True)
for ad in a.adapter:
    m = PeftModel.from_pretrained(base, ad).eval()
    report["results"][ad] = analyse(m)
    print(ad, json.dumps(report["results"][ad], indent=1), flush=True)
    base = m.unload()
with open(os.path.join(a.out, "stop_by_value.json"), "w") as f:
    json.dump(report, f, indent=1)
