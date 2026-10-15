#!/bin/bash
# gpuq payload: the untrained model's top-64 reference for the Stage A training rows (train_frozen.py --ref-topk).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-math-cache/run-20261015T212410Z-d16f220bfe
exec timeout 7200 /home/rrenaud/envs/pytorch/bin/python scripts/ref_topk.py $C/rollouts_math_fresh_train.jsonl "$GPUQ_RUN_DIR/ref_topk_train" --k 64
