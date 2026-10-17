"""Perturb a sweep config for exploration: each continuous knob is multiplied by exp(N(0, s)), so a grid point
becomes a random point near it and repeated centers do not repeat. Zero stays zero (a switched-off term stays
off); decay fractions are clipped to [0.05, 0.9]; values keep 2 significant figures.

usage: jitter.py S "OBJ LR SEED [train_frozen.py args]"   -> prints the jittered config
Jittered: LR and the values of --answer-kl, --support-floor, --decay-frac, --warmup-steps (rounded to an int).
"""
import math, random, sys

KNOBS = {"--answer-kl", "--support-floor", "--decay-frac", "--warmup-steps"}


def perturb(x, s, key):
    v = float(x)
    if v == 0:
        return x
    v *= math.exp(random.gauss(0, s))
    if key == "--warmup-steps":
        return str(max(1, round(v)))
    if key == "--decay-frac":
        v = min(max(v, 0.05), 0.9)
    return f"{float(f'{v:.2g}'):g}"


s, cfg = float(sys.argv[1]), sys.argv[2].split()
cfg[1] = perturb(cfg[1], s, "lr")
for i in range(3, len(cfg) - 1):
    if cfg[i] in KNOBS:
        cfg[i + 1] = perturb(cfg[i + 1], s, cfg[i])
print(" ".join(cfg))
