#!/bin/bash
# gpuq payload for the RLVR-hotpot start: its top-64 reference cache on the training rows, the hotpot scorer's
# agreement with vLLM's greedy confidences and its mass on the candidate list (fresh dev), then 3 training
# steps per objective with the cache.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="timeout 3600 /home/rrenaud/envs/pytorch/bin/python"
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
M=mehuldamani/hotpot-v2-correctness-7b
$PY scripts/ref_topk.py $C/rollouts_hotpot_fresh_train.jsonl "$GPUQ_RUN_DIR/ref_topk_train" --model $M --k 64 || echo "FAILED ref_topk"
$PY scripts/check_scorer.py $C/rollouts_hotpot_fresh_dev.jsonl --scorer hotpot --model $M --rows 500 || echo "FAILED check"
for obj in exact sampled; do
  TASK=hotpot-rlvr timeout 1800 /bin/bash scripts/run_frozen.sh $obj 3e-4 0 --steps 3 --eval-every 3 --dev-limit 32 \
    --drift-rows 16 --ref-topk "$GPUQ_RUN_DIR/ref_topk_train" || echo "FAILED train $obj"
done
