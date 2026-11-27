# Tinker: exact vs sampled confidence RL on the easy-game mix (2026-11-27)

**Setup.** Inkling-2-Mini (free tier) trained with rlcr11-tinker's `tk/train.py`, unchanged: lr 2.4e-4, W 0.0054,
16 questions x 8 rollouts, 1,536 max tokens, rlcr11 grid, RLCR reward (format + correct + Brier). 200 steps per arm,
seed 400. Data: `data/tinker_easy`, 8,000 train rows (g1b two-card high, g2 Leduc, g3 three-card, g5 face-up turn,
2,000 each) and 200 dev. Claim format: the answer is always "yes" and the confidence is P(win). The gold label is one
sampled outcome per row. Launcher `scripts/tk_easy_pair.sh`. Sampled ran 5:23-9:10 AM PT, exact 6:18-10:57.

**Inkling before training.** It solves these games when it finishes thinking (4k cap: Spearman 0.94-0.99). At the
1,536-token training budget only 5% of completions finish, but those are near perfect (Spearman 0.89).

## Trainer's dev eval (greedy, 200 dev, against the sampled label)

| step | sampled: format / mean conf / Brier | exact: format / mean conf / Brier |
|---|---|---|
| 0 | 0.04 / 1.00 / – | 0.04 / 1.00 / – |
| 50 | 0.97 / 0.97 / 0.458 | 0.83 / 0.82 / 0.383 |
| 100 | 0.74 / 0.59 / 0.236 | 0.97 / **0.99** / 0.470 |
| 150 | 0.93 / 0.65 / **0.220** (AUROC 0.645) | 0.89 / 0.85 / 0.342 |
| 200 | 0.73 / 0.52 / 0.237 | 0.91 / 0.59 / 0.253 |

- **Exact collapsed to confidence 1.0 for about 100 steps,** then escaped. Early on, the only completions that
  finished in budget were confident; once q sits on 1.0, the exact gradient for the other levels vanishes. There
  is no support floor in tk/train.py, and Stage B runs elsewhere show the same failure.
- **Per-row KL to the base model was 130-200 nats,** a large rewrite of its reasoning to fit the budget.

## Against the exact oracle (scripts/tk_poker_eval.py --state, same 200 dev spots)

| checkpoint | decoding | parsed | Spearman vs p_win | Brier vs oracle | constant |
|---|---|---|---|---|---|
| base | T 0.7, k 4 | 0.05 | 0.89 | 0.019 | 0.20 |
| sampled step 150 | T 0.7, k 4 | 0.99 | -0.03 | 0.304 | 0.246 |
| sampled step 150 | greedy | 0.93 | -0.13 | 0.309 | |
| sampled step 200 | T 0.7, k 4 | 0.82 | 0.16 | 0.248 | 0.246 |
| exact step 150 | T 0.7, k 4 | 0.98 | -0.02 | 0.414 | 0.246 |
| exact step 200 | T 0.7, k 4 | 0.98 | -0.05 | 0.286 | 0.246 |
| exact step 200 | greedy | 0.91 | 0.11 | 0.270 | |

- **RL destroyed the equity skill in both arms.** They learned to finish inside the budget by dropping the
  computation, and their stated equity is uncorrelated with the truth (|Spearman| <= 0.16) and no better than a
  constant. Confidence never goes below 0.4, even on spots that are certain losses.
- **The trainer's dev metric misled.** Sampled step 150 looked best against the noisy labels (Brier 0.220,
  AUROC 0.645). The greedy oracle eval of the same checkpoint does not reproduce that (Brier vs label 0.295, AUROC
  0.435). With 200 spots and one coin-flip label each, label-based dev metrics cannot rank these runs.
- **No exact-vs-sampled conclusion is possible here.** Both arms end at chance; exact also suffered the collapse.

## Why, and what would fix it

The reward pays for format and for Brier against one coin flip per question. Computing equity is expensive at
1,536 tokens, and its payoff through one noisy label is tiny, so the cheapest policy is to stop computing and hedge.
This is the toy's lesson (notebooks/poker_akv88.py) at LLM scale: sampled labels cannot teach a function when every
state gets about one label.

Next:
- **Oracle reward:** Brier against p_win instead of the coin flip, i.e. exact over the outcome. This removes the
  label noise.
- **A larger budget:** 4,096 tokens, where base Inkling already finishes 38-98% of spots, so RL refines rather
  than rewrites.
- **A support floor for the exact arm,** so a confident start does not absorb it.
