#!/bin/bash
# Does an uncalibrated start follow RLCR-hotpot's format under tabc_long? Qwen2.5-7B base (what RLCR-hotpot was
# trained from) and RLVR-hotpot (same data, correctness reward, GEN prompt): 300 fresh dev questions greedily
# and 100 fresh train questions x 8 at T=0.7, with the shared logging.
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
PY="timeout 3600 /home/rrenaud/envs/vllm/bin/python /home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling/rollout_vllm.py --rlcr /home/rrenaud/rlcr_platt/RLCR --task hotpot --util 0.9"
for m in Qwen/Qwen2.5-7B mehuldamani/hotpot-v2-correctness-7b; do
  t=${m##*/}
  $PY --model $m --data data/hotpot_fresh_dev.jsonl --n 300 --k 1 --temperature 0 --out "$GPUQ_RUN_DIR/rollouts_${t}_dev.jsonl" || echo "FAILED $t dev"
  $PY --model $m --data data/hotpot_fresh_train.jsonl --n 100 --k 8 --temperature 0.7 --out "$GPUQ_RUN_DIR/rollouts_${t}_train.jsonl" || echo "FAILED $t train"
done
