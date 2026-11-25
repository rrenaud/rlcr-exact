#!/bin/bash
# One pre-registered run (docs/prereg_fair_sweep.md), a gpuq payload: train config CID under ARM (exact | sampled),
# saving the adapter at the registered steps, then score every saved checkpoint on dev by greedy generation. Test is
# not touched here. Both parts are resumable: a preempted run (exit 75) is requeued by gpuq and picks up where it was.
# usage: prereg_run.sh CID ARM    (CID = a config of docs/prereg/configs.json, or CIDrK: its K-th replicate, seed + 1000 K)
# Stage C configs (docs/prereg_rlcr_init/configs.json) run Stage B's recipe from released RLCR-hotpot (docs/prereg_rlcr_init.md).
# LEVELS=21: the rlcr21 scorer (0.0, 0.05, ..., 1.0) instead of rlcr11, same configs and seeds (docs/rlcr21.md; not registered runs).
# Runs the code of the checkout this script lives in (main, or a worktree such as rlcr21).
cid=$1 arm=$2 base=${1%%r*} rep=0; [ "$base" != "$1" ] && rep=${1#*r}
trap '' USR1  # the trainer and evaluator poll $GPUQ_PREEMPT_FILE; untrapped, SIGUSR1 would end this shell
cd "$(dirname "$0")/.."
cfg=docs/prereg/configs.json; [ "${base:0:1}" = C ] && cfg=docs/prereg_rlcr_init/configs.json
read stage seed lr W F < <(python3 -c "
import json; c = {c['cid']: c for c in json.load(open('$cfg'))['configs']}['$base']
print(c['stage'], c['seed'] + 1000 * $rep, c['lr'], c['W'], c['F'])")
C4=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261019T233718Z-4511bf5189
VLLM=/home/rrenaud/envs/vllm/bin/python
L=${LEVELS:-11}; case $L in 11|21) ;; *) echo "LEVELS must be 11 or 21"; exit 2;; esac
model=mehuldamani/hotpot-v2-correctness-7b task=hotpot-rlvr-$L
[ "$stage" = C ] && model=mehuldamani/hotpot-v2-brier-7b-no-split task=hotpot-rlcr-$L
export TASK=$task NO_TEST=1 GEN_EVAL=0 HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
# PREREG_SMOKE=1: the same commands at a few steps (plumbing check; not a registered run)
if [ "$stage" = A ]; then  # offline: one pass over the 4k cached questions (4 per step x 1000 steps)
  steps=$(seq -s, 50 50 100),$(seq -s, 200 100 1000) n=1000
  [ "${PREREG_SMOKE:-0}" = 1 ] && steps=2,4 n=4
  # MAX_STEPS=200: the first 200 steps only (constant lr, no schedule: an exact prefix of the full run), saved at the
  # registered steps up to it (rlcr21 transfer probes; not registered runs)
  [ -n "$MAX_STEPS" ] && n=$MAX_STEPS steps=$(tr , '\n' <<< $steps | awk -v n=$MAX_STEPS '$1 <= n' | paste -sd,)
  bash scripts/run_frozen.sh $arm $lr $seed --answer-kl $W --support-floor $F --steps $n --skip-initial-eval \
    --eval-every 100000 --save-steps $steps --train $C4/rollouts_hotpot_fresh_train4k.jsonl --ref-topk $C4/ref_topk_train4k \
    || exit $?
else  # online (B, C): 16 questions x 8 fresh rollouts per step from the same 4k questions
  steps=$(seq -s, 50 50 600) n=600
  [ "${PREREG_SMOKE:-0}" = 1 ] && steps=1,2 n=2
  PY=$VLLM bash scripts/run_frozen.sh $arm $lr $seed --answer-kl $W --steps $n --skip-initial-eval --eval-every 100000 \
    --save-steps $steps --backend vllm --refresh-every 1 --answer-pg 1 --pg-balance rlcr --pg-reward $arm \
    --problems-per-step 16 --train-questions $C4/rollouts_hotpot_fresh_train4k.jsonl || exit $?
fi
$VLLM scripts/prereg_eval.py --split dev --run-dir $GPUQ_RUN_DIR --steps $steps --model $model || exit $?
