#!/bin/bash
# gpuq payload: one pitfalls-sustech run = the pre-registered Stage B recipe (scripts/prereg_run.sh, config B0: online from
# RLVR-hotpot, lr 1e-4, answer-KL 0.02, 600 steps of 16 questions x 8 fresh rollouts) plus extra train_frozen.py flags,
# then the same dev eval at every saved step. Paired with prereg-B B0 / B0rK (same seed = 200 + 1000 K).
# usage: pitfalls_run.sh ARM SEED [extra train_frozen.py args]
#   e.g. pitfalls_run.sh sampled 200 --warm-ce 1 --warm-steps 50
#        pitfalls_run.sh exact 200 --filter-uniform-groups refill
# Resumable like prereg_run.sh (preempted: exit 75, requeued, resumes). PITFALLS_SMOKE=1: 2 steps.
arm=${1:?arm} seed=${2:?seed}; shift 2
trap '' USR1
cd "$(dirname "$0")/.."
C4=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261019T233718Z-4511bf5189
VLLM=/home/rrenaud/envs/vllm/bin/python
# this clone has no sibling rlcr-reward-conf-weighted-tt-scaling/ (train_frozen.py imports logprob_spans from it)
export PYTHONPATH=/home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling TASK=hotpot-rlvr-11 NO_TEST=1 GEN_EVAL=0 \
  HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
steps=$(seq -s, 50 50 600) n=600
[ "${PITFALLS_SMOKE:-0}" = 1 ] && steps=1,2 n=2
PY=$VLLM bash scripts/run_frozen.sh $arm 1e-4 $seed --answer-kl 0.02 --steps $n --skip-initial-eval --eval-every 100000 \
  --save-steps $steps --backend vllm --refresh-every 1 --answer-pg 1 --pg-balance rlcr --pg-reward $arm \
  --problems-per-step 16 --train-questions $C4/rollouts_hotpot_fresh_train4k.jsonl "$@" || exit $?
$VLLM scripts/prereg_eval.py --split dev --run-dir $GPUQ_RUN_DIR --steps $steps --model mehuldamani/hotpot-v2-correctness-7b || exit $?
