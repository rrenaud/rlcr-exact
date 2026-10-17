#!/bin/bash
# gpuq payload: check_trie_layout.py (does packed bf16 numerics depend on the trie size / placement?)
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec timeout 1800 /home/rrenaud/envs/pytorch/bin/python scripts/check_trie_layout.py --groups 12
