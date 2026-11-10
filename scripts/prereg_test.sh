#!/bin/bash
# Test phase (docs/prereg_fair_sweep.md), a gpuq payload: score on test exactly the checkpoints that
# docs/prereg/selections.json (written by prereg_analysis.py select, dev only) names. Resumable: prereg_eval.py skips
# items already scored and exits 75 when preempted.
trap '' USR1
cd /home/rrenaud/rlcr-exact
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
python3 -c "
import json
for t in json.load(open('docs/prereg/selections.json'))['test']: print(t['run_dir'], ','.join(map(str, t['steps'])))" |
while read dir steps; do
  /home/rrenaud/envs/vllm/bin/python scripts/prereg_eval.py --split test --run-dir $dir --steps $steps || exit $?
done
