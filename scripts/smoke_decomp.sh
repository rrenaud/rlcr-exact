#!/bin/bash
# gpuq payload: 6-step smoke of the per-term gradient decomposition (every step vs every 3) for both objectives.
R=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen/run-20261016T144115Z-73d8ea5b8b/ref_topk_train
for obj in exact sampled; do
  TASK=hotpot-rlvr timeout 1200 /bin/bash scripts/run_frozen.sh $obj 3e-4 0 --steps 6 --eval-every 6 --dev-limit 16 \
    --drift-rows 4 --support-floor 0.03 --answer-kl 0.1 --grad-decomp-every 3 --ref-topk $R || echo "FAILED $obj"
done
