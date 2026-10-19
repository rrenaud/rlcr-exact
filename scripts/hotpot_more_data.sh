#!/bin/bash
# More HotpotQA training data for RLVR-hotpot: rollouts (8 per question, T=0.7, as cache_rlvr_hotpot.sh) for the
# 2,000 fresh-train questions the first cache left unused (data/hotpot_fresh_train2.jsonl = lines 2001-4000), then
# the combined 4,000-question train file and its top-64 answer-KL reference (tail-ok rows, as ref_topk_rlvr_stop.sh).
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=/data/gpuq-runs/rrenaud/rlcr-hotpot-rlvr-cache/run-20261016T073943Z-04bf759137
O=$GPUQ_RUN_DIR
timeout 7200 /home/rrenaud/envs/vllm/bin/python /home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling/rollout_vllm.py --rlcr /home/rrenaud/rlcr_platt/RLCR --task hotpot --util 0.9 --model mehuldamani/hotpot-v2-correctness-7b \
  --data data/hotpot_fresh_train2.jsonl --k 8 --temperature 0.7 --n 2000 --out $O/rollouts_hotpot_fresh_train2.jsonl || { echo "FAILED rollouts"; exit 1; }
cat $C/rollouts_hotpot_fresh_train.jsonl $O/rollouts_hotpot_fresh_train2.jsonl > $O/rollouts_hotpot_fresh_train4k.jsonl
timeout 14400 /home/rrenaud/envs/pytorch/bin/python scripts/ref_topk.py $O/rollouts_hotpot_fresh_train4k.jsonl $O/ref_topk_train4k \
  --k 64 --tail-ok --model mehuldamani/hotpot-v2-correctness-7b || { echo "FAILED ref"; exit 1; }
