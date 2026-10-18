#!/bin/bash
# gpuq payload: check_sampled_q.py at the exact rlcr11 adapter (run 2f428b8df7).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 1800 /home/rrenaud/envs/pytorch/bin/python scripts/check_sampled_q.py \
  /data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen/run-20261018T054143Z-2f428b8df7/checkpoints/rlcr11_exact_lr0.00037_W0.02_s0_F0.053
