#!/bin/bash
# Screening tier (gpuq payload, ~25 min on hotpot): run_frozen.sh with 200 steps, one dev eval at the end, no step-0 eval
# (identical across runs from one base), no gradient decomposition, no generation check. It can reject a config (dev
# Brier at step 200 >= ~0.02 worse than the leader's band); it cannot rank good ones (top-20 runs: step-200 vs final
# rank rho ~ -0.1). Later flags override these.
# usage: [TASK=...] screen_frozen.sh OBJECTIVE LR SEED [extra train_frozen.py args]
obj=$1 lr=$2 seed=$3; shift 3
GEN_EVAL=0 exec bash /home/rrenaud/rlcr-exact/scripts/run_frozen.sh $obj $lr $seed --steps 200 --eval-every 200 \
  --skip-initial-eval --grad-decomp-every 0 "$@"
