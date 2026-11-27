"""Zero-shot (or checkpoint) eval on the poker showdown-equity task (scripts/poker_data.py dev/test jsonl).

Per spot: --k completions at T 0.7 (RLCR's sampling temperature) from in-process vLLM, then the exact confidence
distribution q over the rlcr11 grid at each completion's <confidence> tag (trie through the stop, same weights,
as synth_eval.py on branch synth-data-expt). Graded with RLCR's reward_fns (format_reward "tabc", accuracy_reward
exact match against the sampled-showdown gold "yes"/"no").

Per completion (OUT/poker_eval/<tag>.jsonl.gz): answer, format, text confidence, E_q[c], q, and the stated EQUITY
e = the model's P(win): c in the claim format (defined so whatever the answer), else c if "yes", 1 - c if "no"; oracle p_win; sampled outcome. Errors are against
the oracle as well as the label. Analysis: scripts/poker_analyze.py.

usage (vLLM env):
  poker_eval.py --eval data/poker_v1_claim/dev.jsonl --model M --tag T --out DIR [--k 8] [--limit N]
"""
import argparse, collections, gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from rlcr_exact import shared_vllm  # noqa: F401  (in-process vLLM; import before vllm)
from rlcr_exact.shared_vllm import bind_hf, start_vllm
import numpy as np
import torch

from rlcr_exact.confidence import rlcr11_candidates
from rlcr_exact.grading import last_answer, normalize_answer
from rlcr_exact.online import Roller, _confidence, load_questions
from rlcr_exact.trie_score import packed_candidate_logps
from train_frozen import question_rows, token_parts
from logprob_spans import make_token_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
    ap.add_argument("--kv-gb", type=float, default=4.0)
    ap.add_argument("--pack-rows", type=int, default=8)
    ap.add_argument("--pack-tokens", type=int, default=3000)
    a = ap.parse_args()
    out = os.path.join(a.out, "poker_eval")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, f"{a.tag}.json")):
        print(f"{a.tag} done already", flush=True); sys.stdout.flush(); os._exit(0)

    meta = [json.loads(l) for l in open(a.eval)][: a.limit]
    qs = load_questions(a.eval)[: a.limit]
    llm = start_vllm(a.model, kv_gb=a.kv_gb, max_model_len=4096 + a.max_new)
    hf = bind_hf(llm, a.model); hf.eval()
    tok = llm.get_tokenizer()
    token_bytes = make_token_bytes(tok)
    cands = rlcr11_candidates(tok); values = cands.values
    roller = Roller(llm, a.rlcr, k=a.k, temperature=a.temperature, max_new=a.max_new, seed=0)
    res = roller(qs, None)
    print(json.dumps(dict(event="generated", spots=len(res))), flush=True)
    recs = []
    for q, m in zip(res, meta):
        rows = question_rows(q, token_bytes, 4096 + a.max_new, True, collections.Counter())
        qd = {}
        for s, e in token_parts([len(r.prefix) - r.start for r in rows], a.pack_rows, a.pack_tokens):
            part = rows[s:e]
            with torch.no_grad():
                lp, _ = packed_candidate_logps(hf, [r.prefix for r in part], cands.trie, temperature=0.7)
            for r, p1 in zip(part, lp.exp().float().cpu()):
                qd[r.j] = p1
            del lp
        torch.cuda.empty_cache()
        for j, text in enumerate(q["outputs"]):
            ans = normalize_answer(last_answer(text))
            ct = _confidence(text)
            p1 = qd.get(j)
            ce = float((p1 * values).sum() / max(p1.sum(), 1e-9)) if p1 is not None else None
            # claim: confidence is defined as P(win) whatever the answer; rlcr: P(answer right), so P(win) = c or 1 - c
            eq = lambda c: None if c is None else c if m.get("fmt") == "claim" else None if ans not in ("yes", "no") else (c if ans == "yes" else 1 - c)
            recs.append(dict(id=q["id"], j=j, street=m["street"], villain_kind=m["villain_kind"], p_win=m["p_win"],
                             p_tie=m["p_tie"], outcome=m["outcome"], gold=m["answer"], answer=ans[:40],
                             format_ok=float(q["format_ok"][j]), correct=float(q["correct"][j]), truncated=q["truncated"][j],
                             conf_text=ct, conf_exact=ce, equity_text=eq(ct), equity_exact=eq(ce),
                             mass=float(p1.sum()) if p1 is not None else None,
                             q=[round(float(x), 5) for x in p1] if p1 is not None else None,
                             tokens=len(q["completion_ids"][j]), text=text if j == 0 else None))
    with gzip.open(os.path.join(out, f"{a.tag}.jsonl.gz"), "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    f_ = lambda k: float(np.mean([r[k] for r in recs if r[k] is not None])) if any(r[k] is not None for r in recs) else None
    err = lambda k: float(np.mean([(r[k] - r["p_win"]) ** 2 for r in recs if r[k] is not None])) if any(r[k] is not None for r in recs) else None
    summ = dict(tag=a.tag, model=a.model, k=a.k, eval=os.path.abspath(a.eval), completions=len(recs),
                format_ok=f_("format_ok"), correct=f_("correct"), yes_rate=float(np.mean([r["answer"] == "yes" for r in recs])),
                conf_text=f_("conf_text"), equity_text=f_("equity_text"), p_win=f_("p_win"),
                mse_equity_text_vs_oracle=err("equity_text"), mse_equity_exact_vs_oracle=err("equity_exact"),
                mass=f_("mass"), tokens=f_("tokens"), truncated=f_("truncated"))
    tmp = os.path.join(out, f".{a.tag}.json")
    json.dump(summ, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(out, f"{a.tag}.json"))
    print(json.dumps(dict(event="done", **summ)), flush=True)
    sys.stdout.flush()
    os._exit(0)  # in-process vLLM's shutdown can hang holding the GPU


if __name__ == "__main__":
    main()
