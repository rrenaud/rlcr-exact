#!/bin/bash
# gpuq payload: zero-shot of the three RLCR starting points on the easy-game ladder (data/easy_v1/dev_mix.jsonl,
# 300 dev spots per game x 7 games), k 4, claim format. [EASY_SMOKE=1]: 14 spots, k 2, RLCR only.
cd /home/rrenaud/rlcr-exact/explain-to-expert-poker-player
VLLM=/home/rrenaud/envs/vllm/bin/python
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
models="base:Qwen/Qwen2.5-7B rlvr:mehuldamani/hotpot-v2-correctness-7b rlcr:mehuldamani/hotpot-v2-brier-7b-no-split"
extra="--k 4"
[ "${EASY_SMOKE:-0}" = 1 ] && extra="--limit 14 --k 2" models="rlcr:mehuldamani/hotpot-v2-brier-7b-no-split"
for mt in $models; do
  tag=${mt%%:*}_easy model=${mt#*:}
  echo "=== $tag $model $(date -Is)"
  timeout 5400 $VLLM scripts/poker_eval.py --eval data/easy_v1/dev_mix.jsonl --model $model --tag $tag --out $GPUQ_RUN_DIR $extra || exit $?
done
echo "=== all done $(date -Is)"
