"""Draw the pre-registered paired configs (docs/prereg_fair_sweep.md) and write docs/prereg/configs.json.

Each config is run under both arms (exact, and RLCR's sampled k = 1) with the same seed. Values are log-uniform in
the registered boxes as a Latin hypercube (each knob's log range cut into n equal strata, one config per stratum,
strata paired at random), so few configs still cover every range; fixed RNG seed, 2 significant figures. Config 0 of each stage is the "default"
(the hyperparameters earlier work used, tuned on exact): the no-tuning point.
usage: prereg_configs.py > docs/prereg/configs.json
"""
import json, math, random

rng = random.Random(20261103)


def lhs(n, boxes):
    """n points; boxes: knob -> (lo, hi) on a log scale."""
    cols = {}
    for knob, (lo, hi) in boxes.items():
        strata = rng.sample(range(n), n)
        cols[knob] = [float(f"{math.exp(math.log(lo) + (k + rng.random()) / n * (math.log(hi) - math.log(lo))):.2g}")
                      for k in strata]
    return [{knob: cols[knob][i] for knob in boxes} for i in range(n)]


configs = [dict(stage="A", cid="A0", default=True, seed=100, lr=3.7e-4, W=0.02, F=0.053)]
for i, c in enumerate(lhs(6, dict(lr=(1e-4, 1e-3), W=(0.005, 0.1), F=(0.01, 0.15))), 1):
    configs.append(dict(stage="A", cid=f"A{i}", default=False, seed=100 + i, **c))
configs.append(dict(stage="B", cid="B0", default=True, seed=200, lr=1e-4, W=0.02, F=0.0))
for i, c in enumerate(lhs(4, dict(lr=(5e-5, 5e-4), W=(0.005, 0.1))), 1):
    configs.append(dict(stage="B", cid=f"B{i}", default=False, seed=200 + i, F=0.0, **c))
print(json.dumps(dict(rng_seed=20261103, arms=["exact", "sampled"], configs=configs), indent=1))
