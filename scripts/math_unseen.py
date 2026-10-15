"""Big-Math-Digits train rows the released RLCR-math run never reached (it stopped at epoch 0.4992).

Replays RLCR's RepeatSampler order: torch.randperm(len(train), generator seeded with the config seed 43),
in chunks of prompts per step. 208 steps x 72 prompts (6 processes x 6 per device x 64 accumulation / 32
generations) = 14,976 = 0.4992 x 30,000, matching trainer_state.json.
"""
import json
import torch

N, SEED, STEPS, PROMPTS_PER_STEP = 30000, 43, 208, 72
g = torch.Generator(); g.manual_seed(SEED)
order = torch.randperm(N, generator=g).tolist()
seen = order[: STEPS * PROMPTS_PER_STEP]
unseen = sorted(set(range(N)) - set(seen))
json.dump({"seen": sorted(seen), "unseen": unseen, "torch": torch.__version__}, open("data/big_math_digits_seen_split.json", "w"))
print(f"seen {len(seen)}, unseen {len(unseen)}, first steps' indices {order[:5]}")
