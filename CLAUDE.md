# rlcr-exact: working rules

## Publishing experiment data

- When an experiment finishes, export its data for analysis into `results/` and push it to GitHub. This covers
  per-rollout answers, confidences, labels and per-checkpoint metrics.
- Budget: at most **1 MB of committed data per GPU-hour the experiment used**. Measure GPU-hours from gpuq
  run time or the trainer's `compute_s`. For example, a 10 GPU-hour sweep may add up to 10 MB.
- Over budget: slim the export first, for example by dropping full completion text or keeping only registered
  checkpoints. If it still doesn't fit, put the bulk elsewhere (a Hugging Face dataset) and commit only a pointer
  plus the slim tables.
- Template: `scripts/export_rollouts.py` + `scripts/publish_rollouts.sh` (rlcr11 prereg runs →
  `results/rollouts/`). Write gzip with `mtime=0` so re-exports of the same data are byte-identical.
