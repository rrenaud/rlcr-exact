#!/bin/bash
# Submit Stage A runs one at a time: each waits until the previous run leaves the gpuq queue, so other
# work streams' jobs can run in between. usage: [SWEEP_ARGS=...] sweep_chain.sh AFTER_RUN_ID "OBJ LR SEED [ARGS]" ...
after=$1; shift
wait_for() { while gpuq status | grep -q "$1"; do sleep 60; done; }
wait_for "$after"
# Each config is "OBJ LR SEED [extra train_frozen.py args]"; $SWEEP_ARGS is appended to every run.
for cfg in "$@"; do
  set -- $cfg
  obj=$1 lr=$2 seed=$3; shift 3
  id=$(gpuq submit --experiment rlcr-math-frozen --description "Stage A sweep: digit, $obj, lr $lr, seed $seed, 500 steps $* $SWEEP_ARGS" \
        --expected-minutes 120 --cwd /home/rrenaud/rlcr-exact -- /bin/bash scripts/run_frozen.sh $obj $lr $seed "$@" $SWEEP_ARGS 2>&1 | tail -1)
  echo "$(date -u +%H:%M) submitted $cfg as $id"
  wait_for "$id"
  echo "$(date -u +%H:%M) $id ended: $(gpuq show $id | grep -m1 '"state"')"
done
