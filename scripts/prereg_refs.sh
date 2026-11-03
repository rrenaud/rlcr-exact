#!/bin/bash
# Reference panel (docs/prereg_fair_sweep.md), a gpuq payload: RLVR-hotpot (the start), released RLCR-hotpot and
# Qwen2.5-7B base, 3 greedy regenerations each of one split, through the same evaluator as the sweep's checkpoints.
# usage: prereg_refs.sh dev|test   (test only in the test phase, with the selected checkpoints)
trap '' USR1
cd /home/rrenaud/rlcr-exact
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
for m in mehuldamani/hotpot-v2-correctness-7b mehuldamani/hotpot-v2-brier-7b-no-split Qwen/Qwen2.5-7B; do
  /home/rrenaud/envs/vllm/bin/python scripts/prereg_eval.py --split $1 --model $m --regens 3 --out $GPUQ_RUN_DIR || exit $?
done
