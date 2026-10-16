"""Stage A sweep summary under the protocol in NOTES.md (fixed before the runs).

For every train_*.jsonl under the given run directories: the final dev metrics, eligibility (mass >= 0.99,
greedy off-list <= 1%, not collapsed: greedy confidence sd >= 0.05 and greedy AUROC >= untrained - 0.02), answer
drift as the change from step 0 (the metric has a bf16 floor), F1-label metrics alongside, and per objective
the selected lr: lowest final dev greedy Brier among eligible runs, preferring the lower lr within 0.005.

usage: sweep_summary.py RUN_DIR [...]
"""
import glob, json, os, sys


def load(path):
    rows = [json.loads(l) for l in open(path)]
    dev = [r for r in rows if r["event"] == "eval" and r["split"] == "dev"]
    test = [r for r in rows if r["event"] == "eval" and r["split"] == "test"]
    args = next(r["args"] for r in rows if r["event"] == "data")
    return args, dev, test[-1] if test else None


def main():
    runs = []
    for d in sys.argv[1:]:
        for path in glob.glob(os.path.join(d, "train_*.jsonl")):
            args, dev, test = load(path)
            if len(dev) < 2:
                continue
            start, end = dev[0], dev[-1]
            eligible = (end["mass"] >= 0.99 and end["greedy_off_list"] <= 0.01 and end["greedy_sd"] >= 0.05
                        and end["greedy_auroc"] >= start["greedy_auroc"] - 0.02)
            runs.append(dict(objective=args["objective"], lr=args["lr"], seed=args["seed"], steps=end["step"],
                             finished=end["step"] == args["steps"], eligible=eligible, start=start, end=end, test=test,
                             drift=end.get("answer_kl", float("nan")) - start.get("answer_kl", float("nan")),
                             run=os.path.basename(os.path.dirname(path))))
    runs.sort(key=lambda r: (r["objective"], r["lr"], r["seed"]))
    cols = ("greedy_brier", "greedy_auroc", "greedy_ece", "expected_brier", "greedy_brier_f1", "greedy_auroc_f1")
    print(f"{'objective':8s} {'lr':>7s} {'seed':>4s} {'step':>5s} {'elig':>4s} " + " ".join(f"{c:>15s}" for c in cols) + f" {'drift':>7s}")
    for r in runs:
        cells = " ".join(f"{r['start'][c]:.4f}->{r['end'][c]:.4f}" for c in cols)
        print(f"{r['objective']:8s} {r['lr']:7.0e} {r['seed']:4d} {r['steps']:5d} {'yes' if r['eligible'] else 'NO':>4s} {cells} {r['drift']:7.3f}")
    for objective in sorted({r["objective"] for r in runs}):
        cands = [r for r in runs if r["objective"] == objective and r["eligible"] and r["finished"] and r["seed"] == 0]
        if not cands:
            print(f"{objective}: no eligible finished seed-0 run yet"); continue
        best = min(r["end"]["greedy_brier"] for r in cands)
        pick = min((r for r in cands if r["end"]["greedy_brier"] <= best + 0.005), key=lambda r: r["lr"])
        t = pick["test"] or {}
        print(f"{objective}: selected lr {pick['lr']:g} (dev greedy Brier {pick['end']['greedy_brier']:.4f}; best {best:.4f}); "
              f"test greedy Brier {t.get('greedy_brier', float('nan')):.4f} AUROC {t.get('greedy_auroc', float('nan')):.4f} "
              f"ECE {t.get('greedy_ece', float('nan')):.4f}")


if __name__ == "__main__":
    main()
