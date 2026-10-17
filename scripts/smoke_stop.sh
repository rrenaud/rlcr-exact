#!/bin/bash
# gpuq payload: hotpot-stop scorer on RLVR-hotpot. Greedy agreement with vLLM (including stopping), list mass on
# well-formed vs tail-fault rows, then 3 training steps per objective with the live (adapter-off) KL reference,
# to measure step time and memory with the larger trie.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
timeout 3600 /home/rrenaud/envs/pytorch/bin/python scripts/check_scorer.py $C/rollouts_hotpot_fresh_dev.jsonl --scorer hotpot-stop \
  --model mehuldamani/hotpot-v2-correctness-7b --rows 500 || echo "FAILED check"
for obj in exact sampled; do
  GEN_EVAL=0 TASK=hotpot-rlvr-stop timeout 1800 /bin/bash scripts/run_frozen.sh $obj 3e-4 0 --steps 10 --eval-every 10 --dev-limit 32 \
    --drift-rows 16 --answer-kl 0.02 --support-floor 0.05 || echo "FAILED train $obj"
done
