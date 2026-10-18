#!/bin/bash
# gpuq payload: per-value stop probability on RLVR-hotpot dev rows, base and two trained adapters.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTHONPATH=src
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
R=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen
timeout 3000 /home/rrenaud/envs/pytorch/bin/python scripts/stop_by_value.py $C/rollouts_hotpot_fresh_dev.jsonl --rows 500 \
  --adapter $R/run-20261017T135711Z-92097925a7/checkpoints/hotpot_exact_lr0.00013_W0.031_s0_F0.024 \
  --adapter $R/run-20261017T205754Z-5cced962dd/checkpoints/hotpot_exact_lr0.00037_W0.02_s2_F0.053
