#!/bin/bash
# Analysis-only regeneration of RLVR-math under the 9 prompts of math_prompts.py, then trie scoring at the tag.
# Prefixes: the uncal probe's RLVR-math rollouts (run af3dc5c20a), cut at the first </answer>. Dev: 300 questions,
# greedy. Train: 100 questions x 4 cached rollouts, T=0.7.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
P=/data/gpuq-runs/rrenaud/rlcr-math-uncal/run-20261019T050640Z-af3dc5c20a
O=${GPUQ_RUN_DIR:-.}
V="timeout 3600 /home/rrenaud/envs/vllm/bin/python scripts/regen_analysis.py"
$V --rollouts $P/rollouts_big-math-digits-v2-correctness_dev.jsonl --split dev --temperature 0 --out $O/regen_dev.jsonl || echo "FAILED regen dev"
$V --rollouts $P/rollouts_big-math-digits-v2-correctness_train.jsonl --split train --temperature 0.7 --per-question 4 --out $O/regen_train.jsonl || echo "FAILED regen train"
for s in dev train; do
  [ -s $O/regen_$s.jsonl ] && { timeout 3600 /home/rrenaud/envs/pytorch/bin/python scripts/score_analysis.py --regen $O/regen_$s.jsonl --out $O/scored_$s || echo "FAILED score $s"; }
done
