#!/bin/bash
# Math generation check for a confidence LoRA (gpuq payload, or the tail of run_frozen.sh TASK=math-rlvr): greedy
# full rollouts of math_fresh dev and test under PROMPT with the adapter on (gen_math.py), compared by
# gen_eval_math.py with the base model under the same prompt. The base rollouts are generated into BASE_DIR the first
# time (shared by every run with that prompt). Writes OUT_DIR/gen_eval_math/{gen_math_*.jsonl,gen_eval_math.json}.
# usage: gen_eval_math.sh PROMPT ADAPTER_DIR OUT_DIR BASE_DIR
prompt=$1 adapter=$2 out=$3/gen_eval_math base=$4
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_DATASETS_CACHE=/home/rrenaud/.cache/huggingface/datasets HF_HUB_OFFLINE=1
GEN="timeout 5400 /home/rrenaud/envs/vllm/bin/python /home/rrenaud/rlcr-exact/scripts/gen_math.py --prompt $prompt"
mkdir -p "$out" "$base"
for split in dev test; do
  D=/home/rrenaud/rlcr-exact/data/math_fresh_$split.jsonl
  if [ ! -s "$base/gen_math_$split.jsonl" ]; then  # write elsewhere, then move: a failed run leaves no partial file
    $GEN --data $D --out "$base/gen_math_$split.jsonl.part" && mv "$base/gen_math_$split.jsonl.part" "$base/gen_math_$split.jsonl" || echo "FAILED base gen $split"
  fi
  $GEN --data $D --adapter "$adapter" --out "$out/gen_math_$split.jsonl" || echo "FAILED gen $split"
done
/home/rrenaud/envs/pytorch/bin/python /home/rrenaud/rlcr-exact/scripts/gen_eval_math.py --base "$base" --lora "$out" --out "$out/gen_eval_math.json"
