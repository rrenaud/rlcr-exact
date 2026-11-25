#!/bin/bash
# gpuq payload: rlcr21 plumbing smoke. (1) Scorer check: greedy rlcr21 vs the text vLLM wrote, on released RLCR (writes
# 0.25 / 0.75 itself, so agreement should beat rlcr11's) and RLVR-hotpot, rlcr11 alongside for comparison. (2) Stage A
# (A2) and Stage C (C0) exact payloads at a few steps with LEVELS=21, each with its dev eval. Every stage prints its exit code.
D=${GPUQ_RUN_DIR:?}
cd "$(dirname "$0")/.."
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/rrenaud/envs/pytorch/bin/python
RLCR_C=/data/gpuq-runs/rrenaud/rlcr-hotpot-cache/run-20261016T003439Z-bc13159a85
RLVR_C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
for m in "rlcr $RLCR_C mehuldamani/hotpot-v2-brier-7b-no-split" "rlvr $RLVR_C mehuldamani/hotpot-v2-correctness-7b"; do
  set -- $m
  for sc in rlcr21 rlcr11; do
    mkdir -p $D/check_$1
    timeout 1800 $PY scripts/check_scorer.py $2/rollouts_hotpot_fresh_dev.jsonl --scorer $sc --model $3 --rows 300 \
      --out $D/check_$1 > $D/check_$1/$sc.log 2>&1
    echo "check $1 $sc: exit $?"
    python3 -c "import json; r = json.load(open('$D/check_$1/check_scorer_$sc.json')); e = r['eval']
print({k: r[k] for k in ('trie_nodes', 'greedy_matches_vllm', 'mass_mean', 'seconds_per_row')}, {k: e[k] for k in ('greedy_brier', 'greedy_auroc', 'expected_brier', 'level_entropy')})" 2>&1
  done
done
for x in "A2 a" "C0 c"; do
  set -- $x; mkdir -p $D/$2
  env LEVELS=21 GPUQ_RUN_DIR=$D/$2 GPUQ_PREEMPT_FILE=$D/$2/preempt.json PREREG_SMOKE=1 timeout 3600 bash scripts/prereg_run.sh $1 exact > $D/$2/out.log 2>&1
  echo "$1 exact rlcr21: exit $? checkpoints: $(ls -d $D/$2/checkpoints/*/step* 2>/dev/null | xargs -n1 basename | tr '\n' ' ') evals: $(ls $D/$2/prereg_eval/*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
  ls $D/$2/train_*.jsonl
  grep -h '"train"' $D/$2/train_*.jsonl | tail -1 | cut -c1-400
  python3 scripts/offlist_rate.py --levels 21 $D/$2/prereg_eval/dev_step*.jsonl.gz
done
echo SMOKE_DONE
