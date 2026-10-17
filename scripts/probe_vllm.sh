#!/bin/bash
# Runs probe_vllm.py once per model in separate processes so each frees its GPU memory.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
for m in Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  timeout 900 /home/rrenaud/envs/vllm/bin/python scripts/probe_vllm.py /home/rrenaud/rlcr_platt/RLCR "$m" 0.85 || echo "FAILED $m"
done
