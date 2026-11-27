#!/bin/bash
# Exact vs sampled rlcr11 confidence training on Tinker (Inkling-2-Mini free) on the easy-game mix
# (data/tinker_easy: g1b, g2, g3, g5; 2000 train rows each; dev 50 each), using rlcr11-tinker's tk/train.py with
# its chosen hparams (lr 2.4e-4, W 0.0054, 16 x 8, max 1536 tokens). 200 steps per arm.
# usage: tk_easy_pair.sh exact|sampled [SEED]   (run from anywhere; trainer runs in the rlcr11-tinker worktree)
arm=$1 seed=${2:-400}
D=/home/rrenaud/rlcr-exact/explain-to-expert-poker-player/data/tinker_easy
cd /home/rrenaud/rlcr-exact/rlcr11-tinker
source /home/rrenaud/envs/tinker/bin/activate
export TINKER_API_KEY=$(cat ~/.tinker_api_key)
mkdir -p $D/$arm\_s$seed
exec timeout 20h python tk/train.py --arm $arm --seed $seed --out $D/$arm\_s$seed --train $D/train.jsonl --dev $D/dev.jsonl \
  --dev-n 200 --steps ${STEPS:-200} --eval-every 50 --save-every 50 > $D/$arm\_s$seed/stdout.log 2>&1
