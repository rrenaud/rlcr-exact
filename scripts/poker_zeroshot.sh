#!/bin/bash
# gpuq payload: zero-shot poker equity eval (scripts/poker_eval.py) of the three RLCR starting points, one vLLM
# process each, sequentially: Qwen2.5-7B base, RLVR-hotpot, released RLCR-hotpot. Same dev prompt_ids for all three
# (RLCR's tabc_long chat prompt with the shared Qwen2.5 tokenizer, as prereg_refs.sh does).
# usage: [POKER_SMOKE=1] poker_zeroshot.sh [EVAL_JSONL]     smoke: 4 spots, k 2, base + RLCR only
cd /home/rrenaud/rlcr-exact/explain-to-expert-poker-player
VLLM=/home/rrenaud/envs/vllm/bin/python
export HF_HOME=/home/rrenaud/.cache/huggingface HF_HUB_CACHE=/scratch/hf/hub HF_HUB_OFFLINE=1
ev=${1:-data/poker_v1_claim/dev.jsonl}
fmt=$(basename $(dirname $ev))
models="base:Qwen/Qwen2.5-7B rlvr:mehuldamani/hotpot-v2-correctness-7b rlcr:mehuldamani/hotpot-v2-brier-7b-no-split"
extra=""
[ "${POKER_SMOKE:-0}" = 1 ] && extra="--limit 4 --k 2" models="base:Qwen/Qwen2.5-7B rlcr:mehuldamani/hotpot-v2-brier-7b-no-split"
for mt in $models; do
  tag=${mt%%:*}_$fmt model=${mt#*:}
  echo "=== $tag $model $(date -Is)"
  timeout 5400 $VLLM scripts/poker_eval.py --eval $ev --model $model --tag $tag --out $GPUQ_RUN_DIR $extra || exit $?
done
echo "=== all done $(date -Is)"
