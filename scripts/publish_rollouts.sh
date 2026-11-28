#!/bin/bash
# Export every rlcr11 prereg run's per-rollout dev/test answers (scripts/export_rollouts.py) into results/rollouts/
# and push them. Run on the GPU box, from the repo: bash scripts/publish_rollouts.sh [--with-text]
# CPU only, no gpuq needed. --with-text adds the full completions (hundreds of MB; check sizes before pushing).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/rrenaud/envs/pytorch/bin/python}
$PY scripts/export_rollouts.py "$@"
du -ah results/rollouts | sort -h
big=$(find results/rollouts -type f -size +95M)
[ -z "$big" ] || { echo "over GitHub's 100 MB file limit, not pushing: $big"; exit 1; }
git add results/rollouts
git diff --cached --quiet && { echo "nothing new to publish"; exit 0; }
git commit -m "results/rollouts: per-rollout dev/test answers of the rlcr11 prereg runs (A, B, C, ref panel)"
git push origin HEAD
