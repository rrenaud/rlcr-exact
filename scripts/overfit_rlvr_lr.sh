#!/bin/bash
# gpuq payload: the overfit check at lower learning rates. At 3e-4 every 4-problem run collapsed within 25 steps
# to one confidence for every row (AUROC 0.5); rewarding_doubt's frozen-answer sweep found the same above 1.6e-4.
R=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen/run-20261016T144115Z-73d8ea5b8b/ref_topk_train
run() { TASK=hotpot-rlvr timeout 2400 /bin/bash scripts/run_frozen.sh "$@" --steps 300 --eval-every 25 --dev-limit 64 \
          --drift-rows 16 --ref-topk $R || echo "FAILED $*"; }
for lr in 1e-5 3e-5 1e-4; do run exact $lr 0 --train-problems 4; done
run sampled 3e-5 0 --train-problems 4
