"""Tables of the J-space readouts written by jspace.py readout: per model, lens, site and layer, the tokens whose mean
lens log-prob is highest in a Brier group relative to the low-Brier group (a difference of group means, so each token's
frequency offset cancels), and the low group's raw top tokens. Groups: rest (Brier > 0.04), over, under. Tokens without a letter or digit are left out (the J++
lens's readout filter).

usage: python scripts/jspace_report.py DIR [--layers 10 14 18 22 26] [--k 12] > report.md
"""
import argparse, json, os, re

import torch
from transformers import AutoTokenizer

MODEL = "mehuldamani/hotpot-v2-correctness-7b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 14, 18, 22, 26])
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--min-group", type=int, default=15, help="skip groups with fewer rows")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    words = [tok.decode([i]) for i in range(len(tok))]
    keep = torch.tensor([bool(re.search(r"[A-Za-z0-9]", w)) for w in words])
    keep = torch.cat([keep, torch.zeros(max(0, 152064 - len(keep)), dtype=torch.bool)])

    def top(v, k):
        v = v.float().clone()
        v[~keep[: len(v)]] = -float("inf")
        idx = v.topk(k).indices.tolist()
        return ", ".join(f"`{words[i].strip() or repr(words[i])}`" for i in idx)

    for tag in ("base", "exact"):
        path = os.path.join(args.dir, f"means_{tag}.pt")
        if not os.path.exists(path):
            continue
        d = torch.load(path)
        means, counts = d["means"], d["counts"]
        rows = [json.loads(l) for l in open(os.path.join(args.dir, f"rows_{tag}.jsonl"))]
        n = {g: counts.get(("logit", d["layers"][0], "conf", g), 0) for g in ("low", "over", "under", "all")}
        brier = sum((r["c"] - r["y"]) ** 2 for r in rows) / len(rows)
        print(f"\n## {tag}: {len(rows)} rows, Brier {brier:.3f}, groups {n}\n")
        for site in ("conf", "analysis"):
            for lens in d["lenses"]:
                print(f"\n### {tag} / {site} / {lens}-lens\n")
                print("| layer | low: raw top | rest minus low | over minus low | under minus low |")
                print("|---|---|---|---|---|")
                for l in args.layers:
                    low = means.get((lens, l, site, "low"))
                    if low is None:
                        continue
                    cells = [top(low, args.k)]
                    # rest = every row not in low (Brier > 0.04), from the all and low means and their counts
                    n_all, n_low = counts[(lens, l, site, "all")], counts[(lens, l, site, "low")]
                    rest = (means[(lens, l, site, "all")].float() * n_all - low.float() * n_low) / max(n_all - n_low, 1)
                    cells.append(top(rest - low.float(), args.k))
                    for g in ("over", "under"):
                        m = means.get((lens, l, site, g))
                        ok = m is not None and counts[(lens, l, site, g)] >= args.min_group
                        cells.append(top(m.float() - low.float(), args.k) if ok else f"(n = {counts.get((lens, l, site, g), 0)})")
                    print(f"| {l} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
