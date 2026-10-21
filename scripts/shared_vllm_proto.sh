#!/bin/bash
# gpuq payload: shared_vllm_proto.py (one copy of the base weights for vLLM generation and LoRA training).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
timeout 2400 /home/rrenaud/envs/vllm/bin/python /home/rrenaud/rlcr-exact/scripts/shared_vllm_proto.py "$@"
