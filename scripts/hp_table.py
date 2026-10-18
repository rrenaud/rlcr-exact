"""One line per finished hotpot-rlvr run: config, dev greedy Brier / AUROC averaged over the last 3 evals (single
evals bounce 0.02-0.05), final dev conditional means and answer KL, test, and the generation check
(gen_eval/gen_eval.json): answer EM ignoring format (em) and stop-after-</confidence> rate (stop), as changes
from the base policy. Sorted by the dev average.
usage: hp_table.py [EXPERIMENT_DIR]"""
import glob, json, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen"
out = []
for path in glob.glob(f"{root}/run-*/train_*.jsonl"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    a = next((r["args"] for r in rows if r["event"] == "data"), {})
    if a.get("steps", 0) < 100 or a.get("train_problems"):
        continue
    dev = [e for e in rows if e["event"] == "eval" and e["split"] == "dev"]
    test = [e for e in rows if e["event"] == "eval" and e["split"] == "test"]
    if not dev or not test:
        continue
    d, t = dev[-1], test[-1]
    last = dev[-3:]
    brier3 = sum(e["greedy_brier"] for e in last) / len(last)
    auroc3 = sum(e["greedy_auroc"] for e in last) / len(last)
    run = path.split("/")[-2]
    g = os.path.join(root, run, "gen_eval", "gen_eval.json")
    gen = json.load(open(g)) if os.path.exists(g) else None
    cfg = (f"{a['objective']:7s} lr={a['lr']:<7g} W={a.get('answer_kl', 0):<6g} F={a.get('support_floor', 0):<6g}"
           + (f" {a['scorer']}" if a.get("scorer", "hotpot") != "hotpot" else "")  # other row sets: not comparable
           + (f" k={a['samples_per_row']}" if a.get("samples_per_row", 1) > 1 else "")
           + (f" s{a['seed']}" if a.get("seed") else "")
           + (f" wsd={a.get('warmup_steps', 0)}/{a.get('decay_frac', 0):g}" if a.get("warmup_steps") or a.get("decay_frac") else ""))
    line = (f"{cfg:52s} dev3 {brier3:.3f}/{auroc3:.3f} last {d['greedy_brier']:.3f} c|y {d.get('greedy_mean_correct', float('nan')):.2f}/"
            f"{d.get('greedy_mean_wrong', float('nan')):.2f} kl {d.get('answer_kl', float('nan')):.2f}  test {t['greedy_brier']:.3f}/{t['greedy_auroc']:.3f}")
    if gen:
        line += "  gen " + " ".join(f"{s}:em{r['em_any_delta']:+.3f} stop{r['stop_delta']:+.3f} B{r['lora']['brier']:.3f} A{r['lora']['auroc']:.3f}"
                                    if "em_any_delta" in r else f"{s}:(old json)" for s, r in gen.items())
    out.append((brier3, line + f"  [{run[4:19]}]"))
for _, line in sorted(out):
    print(line)
