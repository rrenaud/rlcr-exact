#!/bin/bash
# Plumbing smoke for the pre-registered pipeline (gpuq payload): Stage A and B payloads at a few steps (PREREG_SMOKE=1),
# their dev evals, an evaluator rerun that must find nothing to do, and one reference regeneration.
D=${GPUQ_RUN_DIR:?}
cd /home/rrenaud/rlcr-exact
for x in "A0 exact a" "B0 sampled b"; do
  set -- $x; mkdir -p $D/$3
  env GPUQ_RUN_DIR=$D/$3 GPUQ_PREEMPT_FILE=$D/$3/preempt.json PREREG_SMOKE=1 bash scripts/prereg_run.sh $1 $2 > $D/$3/out.log 2>&1
  echo "$1 $2: exit $? checkpoints: $(ls -d $D/$3/checkpoints/*/step* | xargs -n1 basename | tr '\n' ' ') evals: $(ls $D/$3/prereg_eval/*.json | xargs -n1 basename | tr '\n' ' ')"
  grep '"checkpoint"' $D/$3/train_*.jsonl
done
env HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 \
  /home/rrenaud/envs/vllm/bin/python scripts/prereg_eval.py --split dev --run-dir $D/a --steps 2,4 2>&1 | grep "to do"
env HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 \
  /home/rrenaud/envs/vllm/bin/python scripts/prereg_eval.py --split dev --model mehuldamani/hotpot-v2-brier-7b-no-split --regens 2 --out $D/ref > $D/ref.log 2>&1
echo "ref: exit $?"; for f in $D/ref/prereg_eval/*.json $D/a/prereg_eval/*.json; do python3 -c "
import json,sys; m=json.load(open('$f')); print('$(basename $f)', {k: m[k] for k in ('accuracy','em','format_ok','brier','auroc','ece','conf_parsed','answer_changed')})"; done
