#!/bin/bash
# gpuq payload for grad_check.py (RLVR-hotpot start, full training data).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec timeout 2400 /home/rrenaud/envs/pytorch/bin/python scripts/grad_check.py "$@"
