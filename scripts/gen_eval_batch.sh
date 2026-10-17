#!/bin/bash
# gpuq payload: gen_eval.sh for already-finished hotpot-rlvr runs. usage: gen_eval_batch.sh RUN_DIR ...
# The word control (or a RUN_DIR with no checkpoints/) regenerates the base policy instead, as a noise-floor control.
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
for run in "$@"; do
  [ "$run" = control ] && run=$GPUQ_RUN_DIR/base_regen  # base policy, no adapter
  echo "== $run"; mkdir -p $run
  bash /home/rrenaud/rlcr-exact/scripts/gen_eval.sh "$(ls -d $run/checkpoints/*/ 2>/dev/null | head -1 | grep . || echo none)" $C $run --model mehuldamani/hotpot-v2-correctness-7b
done
