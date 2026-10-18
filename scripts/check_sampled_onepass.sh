#!/bin/bash
# gpuq payload: the sampled objective drawing from the training forward's own q (one pass) must reproduce the old
# two-pass path (--sampled-prepass) exactly: same seed, 20 steps, k=4; compares every logged train/eval number and
# the step time. Writes $GPUQ_RUN_DIR/sampled_onepass.json.
REF=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen/run-20261018T050522Z-eeac77918e/ref_topk_train
A="--steps 20 --eval-every 20 --dev-limit 64 --drift-rows 16 --answer-kl 0.02 --support-floor 0.053 --samples-per-row 4 --ref-topk $REF"
mkdir -p $GPUQ_RUN_DIR/onepass $GPUQ_RUN_DIR/prepass
GEN_EVAL=0 TASK=hotpot-rlvr-11 GPUQ_RUN_DIR=$GPUQ_RUN_DIR/onepass bash scripts/run_frozen.sh sampled 0.00037 0 $A --out $GPUQ_RUN_DIR/onepass || echo "FAILED onepass"
GEN_EVAL=0 TASK=hotpot-rlvr-11 GPUQ_RUN_DIR=$GPUQ_RUN_DIR/prepass bash scripts/run_frozen.sh sampled 0.00037 0 $A --out $GPUQ_RUN_DIR/prepass --sampled-prepass || echo "FAILED prepass"
/home/rrenaud/envs/pytorch/bin/python - "$GPUQ_RUN_DIR" <<'PY'
import glob, json, sys
root = sys.argv[1]
def load(d):
    rows = [json.loads(l) for l in open(glob.glob(f"{root}/{d}/train_*.jsonl")[0])]
    return [r for r in rows if r["event"] in ("train", "eval")]
a, b = load("onepass"), load("prepass")
diffs = {}
for x, y in zip(a, b):
    for k, v in x.items():
        if isinstance(v, float) and k not in ("seconds",):
            diffs[f"{x['event']}{x['step']}{x.get('split', '')}:{k}"] = abs(v - y[k])
worst = max(diffs.values())
t = lambda rs: [r["seconds"] for r in rs if r["event"] == "train"][-1]
out = dict(events=len(a), events_prepass=len(b), max_abs_diff=worst, worst=max(diffs, key=diffs.get),
           seconds_onepass=t(a), seconds_prepass=t(b), identical=worst == 0)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{root}/sampled_onepass.json", "w"), indent=1)
PY
