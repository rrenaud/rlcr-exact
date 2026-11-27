"""Poker showdown-equity dataset (src/rlcr_exact/poker_env.py): spots, exact oracle equity, one sampled showdown.

Writes OUT/{train,dev,test}.jsonl. Each row: id, split, street, villain_kind, hero, board, villain (or null), p_win,
p_tie, n_cases, dealt_villain, runout, outcome (+1 win / 0 split / -1 loss), answer (gold "yes"/"no"), problem
(text for --fmt), source = "hotpot" (so RLCR's accuracy_reward grades by exact match), and prompt_ids when --model
is given (RLCR's tabc_long chat prompt, tokenized as scripts/synth_data.py on branch synth-data-expt does).

Spots are uniform random deals: street uniform over flop/turn/river, villain known/random 50/50. Splits use
disjoint seeds; duplicate (hero, board, villain) spots across splits are dropped from dev/test.

usage: poker_data.py OUT [--train 8000 --dev 500 --test 1000] [--fmt claim|rlcr] [--model M] [--workers 24]
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rlcr_exact import poker_env  # noqa: E402


def make(args):
    split, i, seed = args
    rng = np.random.default_rng([seed, i])
    spot = poker_env.label(poker_env.solve(poker_env.sample_spot(rng)), rng)
    return dict(id=f"{split}_{i}", split=split, **spot)


def key(r):
    return (tuple(r["hero"]), tuple(sorted(r["board"][:3])) + tuple(r["board"][3:]), tuple(r["villain"] or ()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--dev", type=int, default=500)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--fmt", default="claim", choices=["claim", "rlcr"])
    ap.add_argument("--guide", default=None, choices=["estimate", "estimate2"], help="length/estimation guidance (knob; off = RLCR prompt)")
    ap.add_argument("--sym", action="store_true", help="suit symbols (A♠) instead of letters (As)")
    ap.add_argument("--model", default=None, help="tokenizer for prompt_ids, e.g. mehuldamani/hotpot-v2-correctness-7b")
    ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    tok = None
    if a.model:
        sys.path.insert(0, a.rlcr)
        from system_prompts import get_sys_prompt
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model); sys_prompt = get_sys_prompt("tabc_long")
    os.makedirs(a.out, exist_ok=True)
    seen = set()
    with Pool(a.workers) as pool:
        for split, n, seed in (("train", a.train, 1), ("dev", a.dev, 2), ("test", a.test, 3)):
            rows = pool.map(make, [(split, i, seed) for i in range(n)], chunksize=8)
            kept = 0
            with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
                for r in rows:
                    k = key(r)
                    if split != "train" and k in seen:
                        continue
                    seen.add(k)
                    r["problem"] = poker_env.problem_text(r, a.fmt, a.sym, a.guide); r["fmt"] = a.fmt; r["guide"] = a.guide; r["source"] = "hotpot"
                    if tok is not None:
                        text = tok.apply_chat_template([{"role": "system", "content": sys_prompt},
                                                        {"role": "user", "content": f"\n\nPROBLEM: {r['problem']}\n\n"}],
                                                       tokenize=False, add_generation_prompt=True)
                        r["prompt_ids"] = tok(text, add_special_tokens=False)["input_ids"]
                    f.write(json.dumps(r) + "\n"); kept += 1
            if not rows:
                continue
            p = np.array([r["p_win"] for r in rows]); o = np.array([r["outcome"] for r in rows])
            print(f"{split}: {kept} rows; mean p_win {p.mean():.3f}, win rate {(o > 0).mean():.3f}; "
                  f"p_win deciles {np.round(np.quantile(p, np.linspace(0, 1, 11)), 2).tolist()}", file=sys.stderr)


if __name__ == "__main__":
    main()
