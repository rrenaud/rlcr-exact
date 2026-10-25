#!/bin/bash
# gpuq payload: --answer-pg with the group split into parts (--micro-batch, rewards and baseline from a no-grad
# prepass) must match the one-forward group: same seed and rollouts, step-1 per-term gradient norms and train stats.
# Writes $GPUQ_RUN_DIR/split_pg.json.
A="--steps 1 --eval-every 100 --skip-initial-eval --backend vllm --refresh-every 1 --answer-pg 1 --pg-balance rlcr --pg-reward exact
   --grad-decomp-every 1 --answer-kl 0.02 --problems-per-step 4 --rollouts-per-question 8 --policy-temperature 0.7 --dev-limit 16 --drift-rows 4 --no-log-q"
for m in 8 4; do
  mkdir -p $GPUQ_RUN_DIR/m$m
  GEN_EVAL=0 TASK=hotpot-rlvr-11 PY=/home/rrenaud/envs/vllm/bin/python GPUQ_RUN_DIR=$GPUQ_RUN_DIR/m$m bash scripts/run_frozen.sh exact 0.0001 0 $A --micro-batch $m --out $GPUQ_RUN_DIR/m$m || echo "FAILED m$m"
done
/home/rrenaud/envs/pytorch/bin/python - "$GPUQ_RUN_DIR" <<'PY'
import glob, json, sys
root = sys.argv[1]
def load(d):
    rows = [json.loads(l) for l in open(glob.glob(f"{root}/{d}/train_*.jsonl")[0])]
    return {r["event"]: r for r in rows if r["event"] in ("grad", "train", "refresh")}
a, b = load("m8"), load("m4")
out = {"same_rollouts": a["refresh"]["per_q"] == b["refresh"]["per_q"]}
for ev in ("grad",):  # train records come every 10 steps
    for k, v in a[ev].items():
        if isinstance(v, float) and k not in ("seconds",) and isinstance(b[ev].get(k), float):
            out[f"{ev}:{k}"] = [v, b[ev][k]]
print(json.dumps(out, indent=1))
json.dump(out, open(f"{root}/split_pg.json", "w"), indent=1)
PY
