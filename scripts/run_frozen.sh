#!/bin/bash
# One Stage A run (gpuq payload): [TASK=math|hotpot|hotpot-rlvr] run_frozen.sh OBJECTIVE LR SEED [extra train_frozen.py args]
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "${TASK:-math}" = hotpot-rlvr ]; then  # uncalibrated start: RLVR-hotpot's rollouts, scored by RLVR-hotpot
  C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137; T=hotpot
  S="--scorer hotpot --model mehuldamani/hotpot-v2-correctness-7b"
elif [ "${TASK:-math}" = hotpot ]; then
  C=/data/gpuq-runs/rrenaud/rlcr-hotpot-cache/run-20261016T003439Z-bc13159a85; T=hotpot; S="--scorer hotpot"
else
  C=/data/gpuq-runs/rrenaud/rlcr-math-cache/run-20261015T212410Z-d16f220bfe; T=math; S=""
fi
obj=$1 lr=$2 seed=$3; shift 3
exec /home/rrenaud/envs/pytorch/bin/python scripts/train_frozen.py --objective $obj --lr $lr --seed $seed $S \
  --train $C/rollouts_${T}_fresh_train.jsonl --dev $C/rollouts_${T}_fresh_dev.jsonl --dev-limit 500 \
  --test $C/rollouts_${T}_fresh_test.jsonl "$@"
