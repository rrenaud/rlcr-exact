#!/bin/bash
# Pilot for docs/prereg_rlcr_init.md (dev only, not a registered run), a gpuq payload: Stage C's online recipe from
# released RLCR-hotpot, exact arm, 150 steps (a quarter of a registered run), adapters at 50/100/150 scored on dev.
# It screens the lr x W box for harm before the pre-registration is frozen; test is never touched.
# usage: pilot_rlcr_init.sh LR W SEED
lr=$1 W=$2 seed=$3
trap '' USR1  # the trainer and evaluator poll $GPUQ_PREEMPT_FILE; untrapped, SIGUSR1 would end this shell
cd /home/rrenaud/rlcr-exact
C4=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261019T233718Z-4511bf5189
VLLM=/home/rrenaud/envs/vllm/bin/python
export TASK=hotpot-rlcr-11 NO_TEST=1 GEN_EVAL=0 HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
PY=$VLLM bash scripts/run_frozen.sh exact $lr $seed --answer-kl $W --steps 150 --skip-initial-eval --eval-every 100000 \
  --save-steps 50,100,150 --backend vllm --refresh-every 1 --answer-pg 1 --pg-balance rlcr --pg-reward exact \
  --problems-per-step 16 --train-questions $C4/rollouts_hotpot_fresh_train4k.jsonl || exit $?
$VLLM scripts/prereg_eval.py --split dev --run-dir $GPUQ_RUN_DIR --steps 50,100,150 --model mehuldamani/hotpot-v2-brier-7b-no-split
