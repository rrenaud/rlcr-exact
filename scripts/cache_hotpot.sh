#!/bin/bash
# Rollouts of the released RLCR-hotpot checkpoint with the shared logging (rollout_vllm.py): the paper's in-domain
# HotpotQA-Modified test and fresh dev greedily (released eval settings), fresh train at temperature 0.7, 8 each.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
PY="timeout 7200 /home/rrenaud/envs/vllm/bin/python /home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling/rollout_vllm.py --rlcr /home/rrenaud/rlcr_platt/RLCR --task hotpot --util 0.9"
$PY --data data/hotpot_fresh_test.jsonl --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_hotpot_fresh_test.jsonl" || echo "FAILED test"
$PY --data data/hotpot_fresh_dev.jsonl --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_hotpot_fresh_dev.jsonl" || echo "FAILED dev"
$PY --data data/hotpot_fresh_train.jsonl --k 8 --temperature 0.7 --n 2000 --out "$GPUQ_RUN_DIR/rollouts_hotpot_fresh_train.jsonl" || echo "FAILED train"
