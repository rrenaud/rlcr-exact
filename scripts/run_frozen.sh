#!/bin/bash
# One Stage A run (gpuq payload): [TASK=math|hotpot|hotpot-rlvr|hotpot-rlvr-stop|hotpot-rlvr-11] run_frozen.sh OBJECTIVE LR SEED [extra train_frozen.py args]
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case "${TASK:-math}" in hotpot-rlvr|hotpot-rlvr-stop|hotpot-rlvr-11) RLVR=1;; *) RLVR=0;; esac
if [ $RLVR = 1 ]; then  # uncalibrated start: RLVR-hotpot
  C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137; T=hotpot
  # -stop and -11 score the end token after </confidence> and keep tail-fault rows (pass the --tail-ok --ref-topk)
  SC=$(case "$TASK" in hotpot-rlvr-stop) echo hotpot-stop;; hotpot-rlvr-11) echo rlcr11;; *) echo hotpot;; esac)
  S="--scorer $SC --model mehuldamani/hotpot-v2-correctness-7b"; G="--model mehuldamani/hotpot-v2-correctness-7b"
elif [ "${TASK:-math}" = hotpot ]; then
  C=/data/gpuq-runs/rrenaud/rlcr-hotpot-cache/run-20261016T003439Z-bc13159a85; T=hotpot; S="--scorer hotpot"
else
  C=/data/gpuq-runs/rrenaud/rlcr-math-cache/run-20261015T212410Z-d16f220bfe; T=math; S=""
fi
obj=$1 lr=$2 seed=$3; shift 3
/home/rrenaud/envs/pytorch/bin/python scripts/train_frozen.py --objective $obj --lr $lr --seed $seed $S \
  --train $C/rollouts_${T}_fresh_train.jsonl --dev $C/rollouts_${T}_fresh_dev.jsonl --dev-limit 500 \
  --test $C/rollouts_${T}_fresh_test.jsonl "$@" || exit $?
# Hotpot runs end with a generation check: the adapter on during greedy generation must not hurt accuracy.
# GEN_EVAL=0 skips it. The adapter is the one checkpoint this run saved.
if [ "$T" = hotpot ] && [ "${GEN_EVAL:-1}" != 0 ]; then
  bash scripts/gen_eval.sh "$(ls -d ${GPUQ_RUN_DIR:-.}/checkpoints/*/ | head -1)" $C ${GPUQ_RUN_DIR:-.} $G
fi
