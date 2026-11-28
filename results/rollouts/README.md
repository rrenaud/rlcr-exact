# Per-rollout eval answers, rlcr11 pre-registered runs

Written by `scripts/export_rollouts.py` (pushed with `scripts/publish_rollouts.sh`) from every
`prereg_eval/{dev,test}_<tag>.jsonl.gz` under `/data/gpuq-runs/rrenaud/rlcr-prereg-{A,B,C,ref}/run-*`.
Generation is greedy vLLM, one completion per question, 500 dev / 500 test HotpotQA questions
(`docs/prereg_fair_sweep.md`, `docs/prereg_rlcr_init.md`).

## Files

| file | one row per | columns |
|---|---|---|
| `runs.csv` | gpuq run | `run`, `experiment`, `state`, `description`, `parent_run`, `rev` (prereg commit), `stage` (A/B/C/ref), `cid`, `base_cid`, `arm` (exact/sampled), `replicate` (0, 1, 2), `seed`, `lr`, `W`, `F`, `n_eval_files` |
| `rollouts_{dev,test}.csv.gz` | run × checkpoint × question | `run`, `stage`, `cid`, `arm`, `step` (training step; empty for ref), `regen` (ref panel only), `qid`, `answer` (last `<answer>`), `confidence` (last `<confidence>` in [0, 1], else empty), `confidence_text` (raw), `correct` (RLCR's format-gated label), `format_ok`, `truncated`, `n_chars` |
| `questions_{dev,test}.jsonl.gz` | question | `id`, `answer` (gold), `source` |
| `completions_{dev,test}.jsonl.gz` | as rollouts | full completion text (only if exported with `--with-text`) |

## Conventions

- The prereg metrics score an empty `confidence` as 0: `brier = mean((confidence.fillna(0) - correct)**2)`.
- Test was scored only at the checkpoints the dev-only selection rule picked (`docs/prereg*/selections.json`).
- A crashed run and its rerun (`parent_run`) can both appear; filter on `state == "completed"`.
- Greedy completions are not bitwise reproducible across vLLM batch shapes (NOTES.md), so compare aggregates.

```python
import pandas as pd
runs = pd.read_csv("results/rollouts/runs.csv")
dev = pd.read_csv("results/rollouts/rollouts_dev.csv.gz").merge(runs[["run", "seed", "lr", "W", "F", "state"]], on="run")
dev["brier"] = (dev.confidence.fillna(0) - dev.correct) ** 2
dev.groupby(["stage", "cid", "arm", "step"]).brier.mean()
```
