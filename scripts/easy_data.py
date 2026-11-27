"""Write the easy-game ladder (src/rlcr_exact/easy_games.py) as jsonl: OUT/<game>/{train,dev,test}.jsonl.

Each row: game, hero, board, villain, p_win, p_tie (exact), outcome, answer (gold "yes"/"no" from the sampled
outcome), problem (claim-format text), source "hotpot" (RLCR exact-match grading), id. Add --model to include
prompt_ids (RLCR tabc_long chat prompt) for LLM evals.

usage: easy_data.py OUT [--games g1_high,...] [--train 20000 --dev 1000 --test 2000] [--model M]
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rlcr_exact import easy_games as G  # noqa: E402


def make(args):
    game, split, i, seed = args
    r = G.sample(game, np.random.default_rng([seed, i]))
    r.update(id=f"{game}_{split}_{i}", split=split, source="hotpot", problem=G.problem_text(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--games", default=",".join(G.GAMES))
    ap.add_argument("--train", type=int, default=20000); ap.add_argument("--dev", type=int, default=1000)
    ap.add_argument("--test", type=int, default=2000); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default=None); ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
    a = ap.parse_args()
    tok = None
    if a.model:
        sys.path.insert(0, a.rlcr)
        from system_prompts import get_sys_prompt
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model); sysp = get_sys_prompt("tabc_long")
    G._three()  # build the three-card table once before forking
    from rlcr_exact import poker, poker_toy
    poker_toy.enumerate_flops(poker.cards("As Kd"), poker.cards("8h 8c"))  # warm the AK vs 88 cache
    with Pool(a.workers) as pool:
        for game in a.games.split(","):
            os.makedirs(os.path.join(a.out, game), exist_ok=True)
            for split, n, seed in (("train", a.train, 11), ("dev", a.dev, 12), ("test", a.test, 13)):
                rows = pool.map(make, [(game, split, i, seed) for i in range(n)], chunksize=64)
                with open(os.path.join(a.out, game, f"{split}.jsonl"), "w") as f:
                    for r in rows:
                        if tok is not None:
                            text = tok.apply_chat_template([{"role": "system", "content": sysp},
                                                            {"role": "user", "content": f"\n\nPROBLEM: {r['problem']}\n\n"}],
                                                           tokenize=False, add_generation_prompt=True)
                            r["prompt_ids"] = tok(text, add_special_tokens=False)["input_ids"]
                        f.write(json.dumps(r) + "\n")
                p = np.array([r["p_win"] for r in rows])
                print(f"{game:10s} {split:5s} {n:6d} mean p {p.mean():.3f} distinct {len(np.unique(np.round(p, 6)))}",
                      file=sys.stderr)


if __name__ == "__main__":
    main()
