#!/bin/bash
# Think+answer prefixes of RLVR-math (big-math-digits-v2-correctness) under tabc, stopped at the first </answer>, for
# analysis-only regeneration under the chosen prompt: fresh test and dev greedily, fresh train 2,000 x 8 at T=0.7.
# Labels: math-verify on the prefix plus a canonical analysis/confidence (prompt-independent).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
PY="timeout 14400 /home/rrenaud/envs/vllm/bin/python scripts/cache_rollouts.py --rlcr /home/rrenaud/rlcr_platt/RLCR --model mehuldamani/big-math-digits-v2-correctness --stop-at-answer"
$PY --data data/math_fresh_test.jsonl --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_math_fresh_test.jsonl" || echo "FAILED test"
$PY --data data/math_fresh_dev.jsonl --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_math_fresh_dev.jsonl" || echo "FAILED dev"
$PY --data data/math_fresh_train.jsonl --k 8 --temperature 0.7 --limit 2000 --out "$GPUQ_RUN_DIR/rollouts_math_fresh_train.jsonl" || echo "FAILED train"
