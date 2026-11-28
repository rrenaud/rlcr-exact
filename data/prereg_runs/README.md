# Prereg per-question eval dumps

Copied from `/data/gpuq-runs/rrenaud/rlcr-prereg-{A,B,C,ref}/run-*/` on Prune (2026-11-27), same layout, so scripts
that take `--root` can point here, e.g. `python scripts/isotonic_analysis.py --root data/prereg_runs`.

- `manifest.json`: trimmed gpuq manifest (run id, description `prereg@REV CID ARM`, command, times, start revision).
- `prereg_eval/{dev,dev2,test}_stepN.json[l.gz]`: greedy-completion eval summary and per-question completions at each
  saved checkpoint. `rlcr-prereg-ref` holds the released RLCR / RLVR models (`*_regen0`).
- Runs that are still queued or produced no evals are omitted.
