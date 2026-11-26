#!/bin/bash
# gpuq payload: rlcr21 Stage B plumbing smoke (online from RLVR-hotpot): B2 exact at 2 steps with LEVELS=21, its dev evals.
D=${GPUQ_RUN_DIR:?}
cd "$(dirname "$0")/.."
mkdir -p $D/b
env LEVELS=21 GPUQ_RUN_DIR=$D/b GPUQ_PREEMPT_FILE=$D/b/preempt.json PREREG_SMOKE=1 timeout 3600 bash scripts/prereg_run.sh B2 exact > $D/b/out.log 2>&1
echo "B2 exact rlcr21: exit $? checkpoints: $(ls -d $D/b/checkpoints/*/step* 2>/dev/null | xargs -n1 basename | tr '\n' ' ') evals: $(ls $D/b/prereg_eval/*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
grep -h '"train"' $D/b/train_*.jsonl | tail -1 | cut -c1-300
python3 scripts/offlist_rate.py --levels 21 $D/b/prereg_eval/dev_step*.jsonl.gz
echo SMOKE_DONE
