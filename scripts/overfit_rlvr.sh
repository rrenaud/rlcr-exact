#!/bin/bash
# gpuq payload: can the frozen pipeline move RLVR-hotpot's confidence at all? Train on 1 or 4 problems (mixed
# correctness, 8 rollouts each) for 200 steps and evaluate on those same rows; with and without the answer-KL
# penalty; exact, and sampled for comparison.
R=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen/run-20261016T144115Z-73d8ea5b8b/ref_topk_train
run() { TASK=hotpot-rlvr timeout 1800 /bin/bash scripts/run_frozen.sh "$@" --steps 200 --eval-every 25 --dev-limit 64 \
          --drift-rows 16 --ref-topk $R || echo "FAILED $*"; }
run exact 3e-4 0 --train-problems 1 --answer-kl 0
run exact 3e-4 0 --train-problems 1 --answer-kl 1
run exact 3e-4 0 --train-problems 4 --answer-kl 0
run exact 3e-4 0 --train-problems 4 --answer-kl 1
run sampled 3e-4 0 --train-problems 4 --answer-kl 1
