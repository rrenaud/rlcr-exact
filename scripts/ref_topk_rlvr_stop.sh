#!/bin/bash
# gpuq payload: untrained RLVR-hotpot top-64 reference for the hotpot-stop training rows (incl. tail-fault rows).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
exec timeout 10800 /home/rrenaud/envs/pytorch/bin/python scripts/ref_topk.py $C/rollouts_hotpot_fresh_train.jsonl "$GPUQ_RUN_DIR/ref_topk_train" \
  --k 64 --tail-ok --model mehuldamani/hotpot-v2-correctness-7b
