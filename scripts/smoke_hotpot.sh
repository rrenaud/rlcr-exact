#!/bin/bash
# gpuq payload: the HotpotQA top-64 reference cache for the training rows, a check that the hotpot scorer
# reproduces vLLM's greedy confidences on the paper's test, then 3 training steps per objective with the cache.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="timeout 3600 /home/rrenaud/envs/pytorch/bin/python"
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-cache/run-20261016T003439Z-bc13159a85
M=mehuldamani/hotpot-v2-brier-7b-no-split
$PY scripts/ref_topk.py $C/rollouts_hotpot_fresh_train.jsonl "$GPUQ_RUN_DIR/ref_topk_train" --model $M --k 64 || echo "FAILED ref_topk"
$PY scripts/check_scorer.py $C/rollouts_hotpot_fresh_test.jsonl --scorer hotpot --rows 500 || echo "FAILED check"
for obj in exact sampled; do
  TASK=hotpot timeout 1800 /bin/bash scripts/run_frozen.sh $obj 3e-4 0 --steps 3 --eval-every 3 --dev-limit 32 \
    --drift-rows 16 --ref-topk "$GPUQ_RUN_DIR/ref_topk_train" || echo "FAILED train $obj"
done
