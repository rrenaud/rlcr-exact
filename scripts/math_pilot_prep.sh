#!/bin/bash
# Math pilot data for one or more prompts (math_pilot_prep.sh PROMPT [PROMPT ...]): regenerate the analysis of the
# RLVR-math think+answer cache under each prompt (dev/test greedy, train 2,000 x 8 at T=0.7), convert to the
# rollout schema, and build the top-64 answer-KL reference for the train rows. Writes $GPUQ_RUN_DIR/PROMPT/.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=${MATH_CACHE:-/data/gpuq-runs/rrenaud/rlcr-math-rlvr-cache/run-20261019T072732Z-a698e7495d}
PY=/home/rrenaud/envs/pytorch/bin/python
for P in "$@"; do
  O=$GPUQ_RUN_DIR/$P; mkdir -p $O
  for s in test dev train; do
    T=0; K=1; [ $s = train ] && { T=0.7; K=8; }
    timeout 10800 /home/rrenaud/envs/vllm/bin/python scripts/regen_analysis.py --rollouts $C/rollouts_math_fresh_$s.jsonl \
      --split $s --temperature $T --per-question $K --prompts $P --out $O/regen_$s.jsonl || { echo "FAILED regen $P $s"; continue; }
    $PY scripts/regen_to_rollouts.py $O/regen_$s.jsonl $P $O/rollouts_math_fresh_$s.jsonl || echo "FAILED convert $P $s"
  done
  timeout 10800 $PY scripts/ref_topk.py $O/rollouts_math_fresh_train.jsonl $O/ref_topk_train --k 64 \
    --model mehuldamani/big-math-digits-v2-correctness || echo "FAILED ref $P"
done
