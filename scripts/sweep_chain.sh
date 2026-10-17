#!/bin/bash
# Submit Stage A runs one at a time: each waits until the previous run leaves the gpuq queue, so other
# work streams' jobs can run in between. TASK (math|hotpot) is passed into the payload, which gpuq starts in
# a systemd service that does not inherit this shell's environment. usage: [TASK=...] [SWEEP_ARGS=...] sweep_chain.sh AFTER_RUN_ID "OBJ LR SEED [ARGS]" ...
after=$1; shift
wait_for() { while gpuq status | grep -q "$1"; do sleep 60; done; }
wait_for "$after"
# Each config is "OBJ LR SEED [extra train_frozen.py args]"; $SWEEP_ARGS is appended to every run.
# JITTER=S (e.g. 0.35) multiplies lr and the loss/schedule knobs by exp(N(0, S)) at submit time (jitter.py), so a
# config is a center to explore around; the submitted values are in the echo line and the run's own args.
for cfg in "$@"; do
  [ -n "$JITTER" ] && cfg=$(python3 /home/rrenaud/rlcr-exact/scripts/jitter.py $JITTER "$cfg")
  set -- $cfg
  obj=$1 lr=$2 seed=$3; shift 3
  id=$(gpuq submit --experiment rlcr-${TASK:-math}-frozen --description "Stage A sweep: ${TASK:-math}, $obj, lr $lr, seed $seed, 500 steps $* $SWEEP_ARGS" \
        --expected-minutes 120 --cwd /home/rrenaud/rlcr-exact -- /usr/bin/env TASK=${TASK:-math} /bin/bash scripts/run_frozen.sh $obj $lr $seed "$@" $SWEEP_ARGS 2>&1 | tail -1)
  echo "$(date -u +%H:%M) submitted $cfg as $id"
  wait_for "$id"
  echo "$(date -u +%H:%M) $id ended: $(gpuq show $id | grep -m1 '"state"')"
done
