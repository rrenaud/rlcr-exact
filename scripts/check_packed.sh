#!/bin/bash
# gpuq payload: packed shared-prefix scoring vs per-row on RLCR-hotpot, then 20 training steps for the step time.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-cache/run-20261016T003439Z-bc13159a85
timeout 1800 /home/rrenaud/envs/pytorch/bin/python scripts/check_packed.py $C/rollouts_hotpot_fresh_train.jsonl --groups 4 || echo "FAILED check"
TASK=hotpot timeout 1800 /bin/bash scripts/run_frozen.sh exact 3e-4 0 --steps 20 --eval-every 1000 --dev-limit 32 --drift-rows 16 \
  --ref-topk /data/gpuq-runs/rrenaud/rlcr-hotpot-frozen/run-20261016T011428Z-e9054eafab/ref_topk_train || echo "FAILED train"
