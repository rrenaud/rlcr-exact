"""Pre-registered generation evals (docs/prereg_fair_sweep.md): greedy vLLM completions of one split at each of a run's
registered checkpoints, or of a released model with no adapter (the reference panel, --regens times).

One vLLM load serves every checkpoint (each a LoRARequest). Each (split, checkpoint, regeneration) writes
OUT/prereg_eval/<split>_<tag>.json (metrics) and .jsonl.gz (completions) and is skipped if its json exists, so a
preempted job ($GPUQ_PREEMPT_FILE, checked between items; exit 75) resumes where it stopped.

Metrics (gen_metrics, plus the pre-registered conventions): a completion's confidence is its last <confidence> parsed
into [0, 1]; a missing or unparseable one scores as 0 (brier, auroc, ece over all questions). The label is RLCR's
correctness reward (format-gated). brier_parsed / auroc_parsed are over parsed completions only (secondary).

usage (vLLM env):
  prereg_eval.py --split dev --run-dir RUN_DIR [--steps 50,100,...]      # every checkpoints/*/stepN, or the listed ones
  prereg_eval.py --split test --run-dir RUN_DIR --steps 300,1000          # only where the selection rule points
  prereg_eval.py --split dev --model M --regens 3 --out DIR               # reference panel: no adapter
"""
import argparse, glob, gzip, json, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rlcr_exact import shared_vllm  # noqa: F401  (sets vLLM's in-process mode before vllm is imported)
import numpy as np
from sklearn.metrics import roc_auc_score
from vllm import LLM
from vllm.lora.request import LoRARequest

from rlcr_exact.online import Roller, _confidence, gen_metrics, load_questions

CACHE = "/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137"  # prompts (identical in every cache)
SPLIT_N = {"dev": 500, "test": 500}  # dev: the first 500 fresh-dev questions (the trainer's --dev-limit 500)
RLVR = "mehuldamani/hotpot-v2-correctness-7b"


def ece(c, y, bins=10):
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - c[b == k].mean()) for k in range(bins) if (b == k).any()))


def score(qs, base):
    m = gen_metrics(qs, base)
    conf = [_confidence(q["outputs"][0]) for q in qs]
    c = np.array([0.0 if x is None else x for x in conf])
    y = np.array([float(q["correct"][0]) for q in qs])
    m["brier_parsed"], m["auroc_parsed"] = m.pop("brier"), m.pop("auroc")
    m |= dict(brier=float(np.mean((c - y) ** 2)), auroc=float(roc_auc_score(y, c)) if 0 < y.mean() < 1 else None,
              ece=ece(c, y), mean_conf_all=float(c.mean()))
    return m


def preempted():
    f = os.environ.get("GPUQ_PREEMPT_FILE")
    return bool(f and os.path.exists(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLIT_N, required=True)
    ap.add_argument("--run-dir", help="a training run: evaluate its checkpoints/*/stepN adapters")
    ap.add_argument("--steps", default="", help="comma-separated checkpoint steps (default: all found)")
    ap.add_argument("--model", default=RLVR)
    ap.add_argument("--regens", type=int, default=1, help="no-adapter mode: independent greedy regenerations")
    ap.add_argument("--out", help="output dir (default: --run-dir)")
    ap.add_argument("--rlcr", default="/home/rrenaud/rlcr_platt/RLCR")
    ap.add_argument("--max-new", type=int, default=1536)
    args = ap.parse_args()
    out = os.path.join(args.out or args.run_dir, "prereg_eval")
    os.makedirs(out, exist_ok=True)

    items = []  # (tag, LoRARequest or None)
    if args.run_dir:
        want = {int(s) for s in args.steps.split(",") if s}
        dirs = {int(re.search(r"step(\d+)$", d).group(1)): d for d in glob.glob(os.path.join(args.run_dir, "checkpoints", "*", "step*"))}
        missing = want - set(dirs)
        assert not missing, f"no checkpoint for steps {sorted(missing)}"
        for i, step in enumerate(sorted(want or dirs)):
            items.append((f"step{step}", LoRARequest(f"step{step}", i + 1, dirs[step])))
    else:
        items = [(f"{args.model.split('/')[-1]}_regen{r}", None) for r in range(args.regens)]
    todo = [(t, l) for t, l in items if not os.path.exists(os.path.join(out, f"{args.split}_{t}.json"))]
    print(f"{len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        return

    qs = load_questions(os.path.join(CACHE, f"rollouts_hotpot_fresh_{args.split}.jsonl"))[: SPLIT_N[args.split]]
    ids = [q["id"] for q in qs]
    base = {q["id"]: q for q in map(json.loads, open(os.path.join(CACHE, f"rollouts_hotpot_fresh_{args.split}.jsonl")))}
    llm = LLM(model=args.model, enable_lora=True, max_lora_rank=8, max_loras=1, max_model_len=4096 + args.max_new,
              gpu_memory_utilization=0.85, enable_prefix_caching=True, seed=0)
    roller = Roller(llm, args.rlcr, k=1, max_new=args.max_new)
    for tag, lora in todo:
        if preempted():
            print("preempted between items", flush=True)
            sys.stdout.flush()
            os._exit(75)
        order = qs
        if lora is None and "_regen" in tag and not tag.endswith("_regen0"):
            # Greedy decoding is deterministic for a fixed batch; a regeneration in a fresh job differs through vLLM's
            # batch composition. A reshuffled question order reproduces that noise inside one process.
            order = random.Random(int(tag.rsplit("regen", 1)[1])).sample(qs, len(qs))
        res = sorted(roller(order, lora, greedy=True), key=lambda q: ids.index(q["id"]))
        m = score(res, base) | dict(split=args.split, tag=tag, model=args.model,
                                     adapter=lora.lora_path if lora else None)
        with gzip.open(os.path.join(out, f"{args.split}_{tag}.jsonl.gz"), "wt") as f:
            for q in res:
                f.write(json.dumps({k: q[k] for k in ("id", "outputs", "correct", "format_ok", "truncated")}) + "\n")
        tmp = os.path.join(out, f".{args.split}_{tag}.json")
        json.dump(m, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(out, f"{args.split}_{tag}.json"))
        print(json.dumps(m), flush=True)
    sys.stdout.flush()
    os._exit(0)  # in-process vLLM's shutdown can hang holding the GPU


if __name__ == "__main__":
    main()
