"""How well the 11-level one-decimal scorer (digit_score.py) covers what the model actually writes. CPU only.

usage: format_coverage.py ROLLOUTS_JSONL [...]   (reads ROLLOUTS_JSONL.topk.jsonl.gz next to each file if present)

For each cached rollout (cut through the final <confidence>, as train_frozen.py does) it reports:
  1. the sampled confidence string: one of 0.0 .. 1.0, or something else ("0", "1", "0.85", ...);
  2. the sampled token tail: whether it is exactly `0|1` `.` `d` `</` `confidence` `>` <end>, i.e. whether
     ignoring everything after the digit (as digit_score does) loses nothing on the sampled path;
  3. from the raw top-64 log-probs over the confidence span (and the sampled log-probs after it), the
     probability at each position of leaving the one-decimal format along the sampled path:
       first token not `0`/`1`; after `0`, not `.`; after `0.`, not a digit; after `0.d`, not `</`;
       after the digit, any tail other than the sampled `</confidence>` + end.
Only well-formed rollouts with a parseable final <confidence> are rows the trainer uses; malformed ones
are counted separately.
"""
import collections, gzip, json, math, os, re, sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rlcr-reward-conf-weighted-tt-scaling"))
sys.path.insert(0, "/home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling")  # from a worktree, which lacks it
from logprob_spans import cut_point, make_token_bytes

LEVEL = re.compile(r"(0\.\d|1\.0)")


def read_jsonl(path):
    """Records of a JSONL file, skipping a partial last line (a cache still being written)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def describe(name, xs):
    xs = np.array(xs)
    if not len(xs):
        return f"{name}: n=0"
    xs = np.maximum(xs, 0.0)
    return (f"{name}: n={len(xs)} mean={xs.mean():.2e} p50={np.median(xs):.1e} p99={np.quantile(xs, 0.99):.1e} "
            f"max={xs.max():.2e} rows>1e-3={np.mean(xs > 1e-3):.4f} rows>1e-2={np.mean(xs > 1e-2):.4f}")


def main():
    tok = AutoTokenizer.from_pretrained("mehuldamani/big-math-digits-v2-brier-base-tabc")
    token_bytes = make_token_bytes(tok)
    zero, one, dot = (tok.convert_tokens_to_ids(c) for c in "01.")
    digits = {tok.convert_tokens_to_ids(c) for c in "0123456789"}
    tail = tok("</confidence>", add_special_tokens=False).input_ids
    ends = {tok.convert_tokens_to_ids(e) for e in ("<|endoftext|>", "<|im_end|>")}

    for path in sys.argv[1:]:
        topk = {}
        if os.path.exists(path + ".topk.jsonl.gz"):
            topk = {(r["id"], r["j"]): r["confidence"] for r in read_jsonl(path + ".topk.jsonl.gz") if r.get("confidence")}
        n_q, total, strings, tails, kinds = 0, 0, collections.Counter(), collections.Counter(), collections.Counter()
        off = collections.defaultdict(list)
        for q in read_jsonl(path):
            n_q += 1
            for j, (text, ids, lps, ok) in enumerate(zip(q["outputs"], q["completion_ids"], q["sampled_logprobs"], q["format_ok"])):
                total += 1
                cut = cut_point(text, ids, "confidence", token_bytes) if ok else None
                if cut is None:
                    kinds["malformed (not trained on)"] += 1
                    continue
                s = re.findall(r"<confidence>(.*?)</confidence>", text, re.S)[-1]
                strings[s] += 1
                kinds["string is a level 0.0..1.0" if LEVEL.fullmatch(s) else "string is not a level"] += 1
                t = ids[cut:]
                head_ok = len(t) >= 3 and t[0] in (zero, one) and t[1] == dot and t[2] in digits
                exact = head_ok and t[3:3 + len(tail)] == tail and len(t) == 3 + len(tail) + 1 and t[-1] in ends
                tails["exact `0|1 . d </confidence> <end>`" if exact else
                      "level head, other tail" if head_ok else "head not `0|1 . d`"] += 1
                if not exact:
                    tails["  e.g. " + repr(tok.convert_ids_to_tokens(t))[:120]] += 1
                    # How sure the model was of the step that left the format (sampled raw logprob).
                    k = 1 if t[:1] == [zero] and t[1:2] != [dot] else 3 if head_ok else 0
                    tails[f"  sampled p of {tok.convert_ids_to_tokens(t[k])!r} at tail position {k}: "
                          f"{math.exp(lps[cut + k]):.3f}"] += 1
                # Probability of leaving the format after the digit, along the sampled path (raw logprobs).
                if head_ok and t[3:4] == tail[:1]:
                    off["after `0.d`: not `</`"].append(-math.expm1(lps[cut + 3]))
                    n_tail = len(tail) + 1  # `</` `confidence` `>` <end>
                    off["after `0.d`: not the sampled tail `</confidence><end>`"].append(-math.expm1(sum(lps[cut + 3: cut + 3 + n_tail])))
                span = topk.get((q["id"], j))
                if not span:
                    continue
                pos = {p: k for k, p in enumerate(span["positions"])}
                dist = lambda p: dict(zip(span["top_ids"][pos[p]], span["top_logprobs"][pos[p]])) if p in pos else None
                mass = lambda d, ts: sum(math.exp(d[x]) for x in ts if x in d)
                first, second, third = (dist(cut + i) for i in range(3))  # positions index completion_ids
                if first is not None:
                    off["first token: not `0` or `1`"].append(max(0.0, 1 - mass(first, (zero, one))))
                    off["first token: `1` (all of it counts as 1.0)"].append(mass(first, (one,)))
                if t[:1] == [zero] and second is not None:
                    off["after `0`: not `.`"].append(max(0.0, 1 - mass(second, (dot,))))
                if t[:2] == [zero, dot] and third is not None:
                    off["after `0.`: not a digit"].append(max(0.0, 1 - mass(third, digits)))
        print(f"== {path}\nproblems {n_q}  rollouts {total}  topk spans {len(topk)}")
        for k, v in kinds.most_common():
            print(f"  {k}: {v} ({v / total:.4f})")
        print("  confidence strings:", strings.most_common(16))
        print("  non-level strings:", [(s, c) for s, c in strings.most_common() if not LEVEL.fullmatch(s)][:20])
        rows = sum(strings.values())
        for k, v in sorted(tails.items(), key=lambda kv: (kv[0].startswith("  "), -kv[1])):
            print(f"  tail {k}: {v} ({v / rows:.4f})" if not k.startswith("  e.g.") else f"  tail {k}: {v}")
        print("  probability of leaving the format along the sampled path (per row):")
        for k, v in off.items():
            print("   ", describe(k, v))


if __name__ == "__main__":
    main()
