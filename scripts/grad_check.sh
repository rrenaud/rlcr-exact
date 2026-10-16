#!/bin/bash
# gpuq payload for grad_check.py (RLVR-hotpot, full training data): the untrained start, then the step-500 adapters
# of the exact 3e-4 runs without and with the support floor.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="timeout 2400 /home/rrenaud/envs/pytorch/bin/python scripts/grad_check.py"
F=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-frozen
$PY --tag start || echo "FAILED start"
$PY --tag exact3e-4_step500 --adapter $F/run-20261016T152545Z-a68cc2e613/checkpoints/hotpot_exact_lr0.0003_W1_s0 || echo "FAILED W1"
$PY --tag exact3e-4_F0.03_step500 --adapter $F/run-20261016T190650Z-de699b78e8/checkpoints/hotpot_exact_lr0.0003_W1_s0_F0.03 || echo "FAILED F"
