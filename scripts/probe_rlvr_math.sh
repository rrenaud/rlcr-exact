#!/bin/bash
# Is an uncalibrated math start plastic at the confidence tag? RLVR-math (big-math-digits-v2-correctness: same data
# as RLCR-math, correctness reward) and Qwen2.5-7B base (RLCR-math's start) under tabc: 300 fresh dev problems
# greedily and 100 fresh train problems x 8 at T=0.7, top-64 logprobs logged. Mirrors probe_uncalibrated.sh (hotpot).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
PY="timeout 3600 /home/rrenaud/envs/vllm/bin/python scripts/cache_rollouts.py --rlcr /home/rrenaud/rlcr_platt/RLCR"
for m in mehuldamani/big-math-digits-v2-correctness Qwen/Qwen2.5-7B; do
  t=${m##*/}
  $PY --model $m --data data/math_fresh_dev.jsonl --limit 300 --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_${t}_dev.jsonl" || echo "FAILED $t dev"
  $PY --model $m --data data/math_fresh_train.jsonl --limit 100 --k 8 --temperature 0.7 --out "$GPUQ_RUN_DIR/rollouts_${t}_train.jsonl" || echo "FAILED $t train"
done
