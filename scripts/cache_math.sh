#!/bin/bash
# Rollouts of the released RLCR-math checkpoint: official test and fresh dev greedily (released eval
# settings), fresh train sampled at the training temperature, 8 per problem. Logging as in conf-weighted-vote.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
PY="timeout 7200 /home/rrenaud/envs/vllm/bin/python scripts/cache_rollouts.py --rlcr /home/rrenaud/rlcr_platt/RLCR"
$PY --data data/math_fresh_test.jsonl --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_math_fresh_test.jsonl" || echo "FAILED test"
$PY --data data/math_fresh_dev.jsonl --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_math_fresh_dev.jsonl" || echo "FAILED dev"
$PY --data data/math_fresh_train.jsonl --k 8 --temperature 0.7 --limit 2000 --out "$GPUQ_RUN_DIR/rollouts_math_fresh_train.jsonl" || echo "FAILED train"
