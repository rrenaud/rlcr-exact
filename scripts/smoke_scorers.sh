#!/bin/bash
# GPU smoke for the confidence scorers on the untrained RLCR-math checkpoint: trie vs digit agreement on 300
# greedy test prefixes, then 3 training steps (1 problem x 8 rollouts each) per scorer x objective.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="timeout 1800 /home/rrenaud/envs/pytorch/bin/python"
C=/data/gpuq-runs/rrenaud/rlcr-math-cache/run-20261015T212410Z-d16f220bfe
$PY scripts/compare_scorers.py $C/rollouts_math_fresh_test.jsonl --rows 300 --digit-batch 8 || echo "FAILED compare"
for scorer in trie digit; do
  [ $scorer = digit ] && B="--micro-batch 4 --eval-batch 8" || B=""
  for obj in exact sampled; do
    $PY scripts/train_frozen.py --train $C/rollouts_math_fresh_train.jsonl --dev $C/rollouts_math_fresh_dev.jsonl \
      --dev-limit 16 --scorer $scorer --objective $obj --steps 3 --problems-per-step 1 --eval-every 3 $B \
      || echo "FAILED train $scorer $obj"
  done
done
