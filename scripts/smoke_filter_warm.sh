#!/bin/bash
# gpuq payload: plumbing smoke for the pitfalls-sustech arms, Stage B recipe (online from RLVR-hotpot, B0 lr/W) at 2 steps.
#   a: exact,   --filter-uniform-groups refill            (+ dev eval of both checkpoints)
#   b: sampled, --filter-uniform-groups refill, --warm-ce 1
#   c: sampled, --warm-ce 1 (no filter)
D=${GPUQ_RUN_DIR:?}
cd "$(dirname "$0")/.."
trap '' USR1
C4=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261019T233718Z-4511bf5189
VLLM=/home/rrenaud/envs/vllm/bin/python
# this clone has no sibling rlcr-reward-conf-weighted-tt-scaling/ (train_frozen.py imports logprob_spans from it)
export PYTHONPATH=/home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling TASK=hotpot-rlvr-11 NO_TEST=1 GEN_EVAL=0 HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
run() {  # name arm extra...
  local n=$1 arm=$2; shift 2; mkdir -p $D/$n
  env GPUQ_RUN_DIR=$D/$n GPUQ_PREEMPT_FILE=$D/$n/preempt.json PY=$VLLM timeout 2400 bash scripts/run_frozen.sh $arm 1e-4 200 \
    --answer-kl 0.02 --steps 2 --skip-initial-eval --eval-every 100000 --save-steps 1,2 --backend vllm --refresh-every 1 \
    --answer-pg 1 --pg-balance rlcr --pg-reward $arm --problems-per-step 16 --train-questions $C4/rollouts_hotpot_fresh_train4k.jsonl \
    "$@" > $D/$n/out.log 2>&1
  echo "$n $arm $*: exit $? checkpoints: $(ls -d $D/$n/checkpoints/*/step* 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
  grep -h '"refresh"\|filter_' $D/$n/train_*.jsonl | cut -c1-400
  grep -h '"train"' $D/$n/train_*.jsonl | tail -1 | cut -c1-400
}
run a exact --filter-uniform-groups refill
run b sampled --filter-uniform-groups refill --warm-ce 1 --warm-steps 2
run c sampled --warm-ce 1 --warm-steps 2
timeout 1800 $VLLM scripts/prereg_eval.py --split dev --run-dir $D/a --steps 1,2 --model mehuldamani/hotpot-v2-correctness-7b > $D/a/eval.log 2>&1
echo "eval a: exit $? $(ls $D/a/prereg_eval/*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
echo SMOKE_DONE
