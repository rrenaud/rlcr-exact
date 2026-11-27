"""Proper scores as a bettor facing random prices (Schervish 1989); see docs/poker_site.

A forecaster states c and is offered a ticket costing t that pays 1 if y = 1; it buys iff c > t. Against fair
value it loses t (bought, y = 0) or 1 - t (passed, y = 1). Averaging over t ~ w gives a proper score S_w(c, y):
  "uniform" = Brier / 2; "log" ∝ 1/(t(1-t)) on [0.001, 0.999] (clipped log score, rescaled);
  "caller" = bets b x pot, b ~ U[1/4, 2], price t = b / (1 + 2b) on [1/6, 0.4], w ∝ 1 / (1 - 2t)^2.
H(p): loss when stating the truth; D(p, c) = ∫_{t between c and p} w |p - t|: EV lost to the wrong decisions.
"""
import numpy as np

T = np.linspace(0, 1, 200001)


def weight(name):
    if name == "uniform":
        w = np.ones_like(T)
    elif name == "log":
        w = np.where((T >= 0.001) & (T <= 0.999), 1 / np.clip(T * (1 - T), 1e-12, None), 0)
    elif name == "caller":
        w = np.where((T >= 1 / 6) & (T <= 0.4), 1 / np.clip((1 - 2 * T) ** 2, 1e-12, None), 0)
    w = w / np.trapezoid(w, T)
    F = np.concatenate([[0], np.cumsum((w[1:] + w[:-1]) / 2 * np.diff(T))])
    G = np.concatenate([[0], np.cumsum((w[1:] * T[1:] + w[:-1] * T[:-1]) / 2 * np.diff(T))])
    return w, F, G


class Score:
    """Threshold-weighted score; units: loss per decision as a fraction of the ticket payout."""
    def __init__(self, name):
        self.name = name; self.w, self.F, self.G = weight(name)
    def F_(self, x): return np.interp(x, T, self.F)
    def G_(self, x): return np.interp(x, T, self.G)
    def H(self, p):  # ∫_0^p w t (1-p) + ∫_p^1 w (1-t) p
        p = np.asarray(p, float)
        return (1 - p) * self.G_(p) + p * ((self.F_(1) - self.F_(p)) - (self.G_(1) - self.G_(p)))
    def D(self, p, c):
        p, c = np.broadcast_arrays(np.asarray(p, float), np.asarray(c, float))
        a, b = np.minimum(p, c), np.maximum(p, c)
        return np.abs(p * (self.F_(b) - self.F_(a)) - (self.G_(b) - self.G_(a)))
    def S(self, c, y):  # loss of stating c when outcome is y
        return self.H(y) + self.D(y, c)
