#!/bin/bash
# Generation check for a trained confidence LoRA (gpuq payload, or the tail of run_frozen.sh): greedy vLLM
# rollouts of the fresh HotpotQA dev and test splits with the adapter on, compared by gen_eval.py with the base
# policy's cached greedy rollouts. Writes OUT_DIR/gen_eval/{rollouts_*,gen_eval.json}.
# usage: gen_eval.sh ADAPTER_DIR|none BASE_CACHE_DIR OUT_DIR [--model M]   (--model omitted = released RLCR-hotpot)
# ADAPTER none regenerates the base policy: the run-to-run noise floor of answer_changed (vLLM batching).
adapter=$1 base=$2 out=$3/gen_eval; shift 3
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
mkdir -p "$out"
PY="timeout 3600 /home/rrenaud/envs/vllm/bin/python /home/rrenaud/rlcr-exact/rlcr-reward-conf-weighted-tt-scaling/rollout_vllm.py --rlcr /home/rrenaud/rlcr_platt/RLCR --task hotpot --util 0.85 $([ "$adapter" != none ] && echo --adapter $adapter) $*"
for split in dev test; do
  $PY --data /home/rrenaud/rlcr-exact/data/hotpot_fresh_$split.jsonl --k 1 --temperature 0 --out "$out/rollouts_hotpot_fresh_$split.jsonl" || echo "FAILED gen $split"
done
/home/rrenaud/envs/pytorch/bin/python /home/rrenaud/rlcr-exact/scripts/gen_eval.py --base "$base" --lora "$out" --out "$out/gen_eval.json"
