import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rlcr_exact import poker, poker_toy as T


def test_flop_equity_matches_equity(tmp_path):
    H, V = poker.cards("As Kd"), poker.cards("8h 8c")
    d = T.enumerate_flops(H, V, cache_dir=str(tmp_path))
    assert len(d["flops"]) == 17296 and abs(d["eq"].mean() - 0.4428) < 1e-3
    for i in np.random.default_rng(0).choice(len(d["flops"]), 4, replace=False):
        pw, pt, n = poker.equity(H, list(d["flops"][i]), [tuple(V)])
        assert n == 990 and abs(pw - d["eq"][i]) < 1e-12 and abs(pt - d["ties"][i] / 990) < 1e-12


def test_sampled_targets_endpoints():
    w = np.array([0, 300, 990]); rng = np.random.default_rng(0)
    assert np.allclose(T.sampled_targets(w, 990, rng), w / 990)
    assert set(np.unique(T.sampled_targets(np.full(1000, 495), 1, rng))) <= {0.0, 1.0}
