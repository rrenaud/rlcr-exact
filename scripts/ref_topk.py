"""Cache the untrained model's top-k log-probs at every cached completion position, for train_frozen.py --ref-topk.

The answer-KL penalty compares the current model with the untrained one along each row's cached completion.
The rows are fixed, so the untrained side can be computed once and reused by every sweep run instead of a
reference forward pass per row per step.

usage: ref_topk.py ROLLOUTS OUT_DIR [--k 64] [--max-prefix 4608]
Writes OUT_DIR/ids.int32 and OUT_DIR/logp.float16 ([positions, k] memmaps, rows back to back) and
OUT_DIR/meta.json (index of (qid, j, offset, length), and what the cache was built from).
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_frozen import load_rows, make_token_bytes

LM_HEAD_CHUNK = 1024  # completion positions per lm_head call, to bound the fp32 [positions, vocab] buffer


@torch.no_grad()
def write_cache(model, rows, out_dir, k, meta):
    """Top-k log-probs of `model` at each row's completion positions (start - 1 .. len - 2, which predict the
    completion tokens) into out_dir; returns the meta dict written to meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    lengths = [len(r.prefix) - r.start for r in rows]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    n = int(offsets[-1])
    ids = np.memmap(os.path.join(out_dir, "ids.int32"), dtype=np.int32, mode="w+", shape=(n, k))
    logp = np.memmap(os.path.join(out_dir, "logp.float16"), dtype=np.float16, mode="w+", shape=(n, k))
    device = next(model.parameters()).device
    t = time.time()
    for i, row in enumerate(rows):
        hidden = model.model(input_ids=torch.tensor([row.prefix], device=device), use_cache=False).last_hidden_state[0]
        hidden = hidden[row.start - 1: len(row.prefix) - 1]
        for c in range(0, len(hidden), LM_HEAD_CHUNK):
            top = model.lm_head(hidden[c: c + LM_HEAD_CHUNK]).float().log_softmax(-1).topk(k, dim=-1)
            lo = offsets[i] + c
            ids[lo: lo + len(top.indices)] = top.indices.cpu().numpy().astype(np.int32)
            logp[lo: lo + len(top.values)] = top.values.cpu().numpy().astype(np.float16)
        if (i + 1) % 1000 == 0:
            print(json.dumps(dict(event="progress", rows=i + 1, of=len(rows), seconds=round(time.time() - t, 1))), flush=True)
    ids.flush(); logp.flush()
    meta = dict(meta, k=k, positions=n, rows=len(rows),
                index=[[r.qid, r.j, int(offsets[i]), lengths[i]] for i, r in enumerate(rows)])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts"); ap.add_argument("out_dir")
    ap.add_argument("--model", default="mehuldamani/big-math-digits-v2-brier-base-tabc")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tail-ok", action="store_true", help="also rows whose only fault is after </confidence> (scorer hotpot-stop)")
    ap.add_argument("--max-prefix", type=int, default=4608)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    rows, dropped = load_rows(a.rollouts, make_token_bytes(tok), a.max_prefix, a.tail_ok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0}).eval()
    meta = write_cache(model, rows, a.out_dir, a.k, dict(rollouts=os.path.abspath(a.rollouts), max_prefix=a.max_prefix, tail_ok=a.tail_ok,
                                                          model=a.model, dropped=dropped))
    print(json.dumps(dict(event="done", rows=meta["rows"], positions=meta["positions"], out=a.out_dir)), flush=True)


if __name__ == "__main__":
    main()
