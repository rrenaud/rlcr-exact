#!/bin/bash
# Re-score the math prompt probe (run 636fd3e715) with the candidates that cover both closings (`>` and `>\n`).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
D=/data/gpuq-runs/rrenaud/rlcr-math-uncal/run-20261019T072732Z-636fd3e715
for s in dev train; do
  timeout 3600 /home/rrenaud/envs/pytorch/bin/python scripts/score_analysis.py --regen $D/regen_$s.jsonl --out $GPUQ_RUN_DIR/scored_$s || echo "FAILED score $s"
done
