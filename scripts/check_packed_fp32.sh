#!/bin/bash
# gpuq payload: packed vs per-row on Qwen2.5-0.5B in float32 and bfloat16 over real HotpotQA groups. fp32 should
# agree to ~1e-4 if the packing is exact; bf16 shows the rounding noise alone.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-cache/run-20261016T003439Z-bc13159a85
for dt in float32 bfloat16; do
  timeout 1200 /home/rrenaud/envs/pytorch/bin/python scripts/check_packed.py $C/rollouts_hotpot_fresh_train.jsonl --groups 4 \
    --model Qwen/Qwen2.5-0.5B --dtype $dt || echo "FAILED $dt"
done
