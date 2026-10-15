#!/bin/bash
# One Stage A run (gpuq payload): run_frozen.sh OBJECTIVE LR SEED [extra train_frozen.py args]
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-math-cache/run-20261015T212410Z-d16f220bfe
obj=$1 lr=$2 seed=$3; shift 3
exec /home/rrenaud/envs/pytorch/bin/python scripts/train_frozen.py --objective $obj --lr $lr --seed $seed \
  --train $C/rollouts_math_fresh_train.jsonl --dev $C/rollouts_math_fresh_dev.jsonl --dev-limit 500 \
  --test $C/rollouts_math_fresh_test.jsonl "$@"
