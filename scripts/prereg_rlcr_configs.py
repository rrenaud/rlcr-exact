"""Draw the pre-registered configs of the RLCR-start sweep (docs/prereg_rlcr_init.md) and write its configs.json.

Stage C = Stage B of docs/prereg_fair_sweep.md (online GRPO, 16 x 8 fresh rollouts, 600 steps) started from released
RLCR-hotpot instead of RLVR-hotpot. Each config runs under both arms with the same seed. C0 is the RLVR-start winner
(B2: the transfer point, no tuning for this start); C1-C4 are a Latin hypercube over log-uniform boxes, as in
prereg_configs.py. Fixed RNG seed, 2 significant figures.
usage: prereg_rlcr_configs.py > docs/prereg_rlcr_init/configs.json
"""
import json, math, random

rng = random.Random(20261119)


def lhs(n, boxes):
    """n points; boxes: knob -> (lo, hi) on a log scale."""
    cols = {}
    for knob, (lo, hi) in boxes.items():
        strata = rng.sample(range(n), n)
        cols[knob] = [float(f"{math.exp(math.log(lo) + (k + rng.random()) / n * (math.log(hi) - math.log(lo))):.2g}")
                      for k in strata]
    return [{knob: cols[knob][i] for knob in boxes} for i in range(n)]


configs = [dict(stage="C", cid="C0", default=True, seed=300, lr=2.4e-4, W=0.0054, F=0.0)]
for i, c in enumerate(lhs(4, dict(lr=(2e-5, 4e-4), W=(0.005, 0.1))), 1):
    configs.append(dict(stage="C", cid=f"C{i}", default=False, seed=300 + i, F=0.0, **c))
print(json.dumps(dict(rng_seed=20261119, start="mehuldamani/hotpot-v2-brier-7b-no-split", arms=["exact", "sampled"],
                      configs=configs), indent=1))
