# Poker prior work (survey, 2026-11-27)

Done by a research subagent; [V] = page read, [S] = search snippet only. Not independently re-checked.

**Bottom line.** No prior work trains or scores an LLM's stated confidence that it wins at showdown against exact
enumerated equity with a proper score. The closest is Poker Arena (arXiv 2606.13815, [V]): evaluation only, MAE of
equity stated in free-form reasoning against engine Monte Carlo. Released poker checkpoints learned action labels,
not equity, and none is Qwen2.5-7B. So the contamination risk is low, apart from memorized preflop matchup tables
(AA vs KK). v1 is therefore postflop only.

## Trained poker LLMs (all action/decision tasks)
- PokerBench (arXiv 2501.08328, AAAI-25) [V]: 11k solver-labelled decisions; SFT on Llama-3-8B, Llama-2-7B and
  Gemma-2B; no equity labels. Data: github.com/pokerllm/pokerbench, HF RZ412/PokerBench.
- YiPz/qwen3-4b-pokerbench-grpo [V]: Qwen3-4B, SFT then GRPO with an LLM-judge action reward (Apache-2.0).
- ToolPoker (arXiv 2602.00528, ICLR-26) [V]: behavior cloning plus RL fell short; solver tools added.
- SPIRAL (arXiv 2506.24119) [V/S]: self-play RL on Kuhn poker transfers to math (MIT code).
- Also [S]: PokerGPT (2401.06781), MemoPilot (2606.08656), ECHO-2 (2602.02192).
- Emergent World Beliefs (2512.23722) [V]: a model pretrained on hand histories encodes equity in probes.

## Benchmarks
- GTO Wizard Benchmark (2603.23660) [V]: full HUNL against GTO Wizard AI, scored by win rate.
- PokerSkill (2605.30094) [V]: prompt and skill library, no training.
- chrystians/poker-holdembench [V]: has a self-stated `equity_estimate` field, null in the rows checked.

## Calibration against exact chance
- Dice, coin and sampling-fidelity papers (2511.00620, 2601.05414, 2506.09998) [S]: about sampling, not stated
  confidence.
- Probabilistic Calibration Is a Trainable Capability (2605.11845) [V]: trie soft targets for sampling
  distributions; method is close to our trie work.
- None found on stated confidence against exact card probabilities.

## Gyms and libraries
- Multi-step poker environments: TextArena (MIT; wrapped by verifiers), RLCard, PettingZoo, OpenSpiel, PokerKit
  (MIT, best game engine).
- None is single-step equity. reasoning-gym and prime-environments have no poker task.
- Evaluators: eval7 (MIT, Cython, exact hand-vs-range), phevaluator (Apache-2.0, fast C), treys (pure Python).
- We use our own numpy evaluator (src/rlcr_exact/poker.py, tested against a reference). eval7 or phevaluator is
  the upgrade path if speed matters.
