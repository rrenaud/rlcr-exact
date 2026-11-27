"""Capacity experiments on the hard rungs (g4_river hand reading, g5_turn outs, g3_three): more data, deeper set
transformers, a looped (weight-tied, recurrent-depth) transformer evaluated at several loop counts, and auxiliary
hand-category targets. CPU only; --threads caps torch threads (default 8) so the host keeps CPU free.

Looped model: x0 = card + role embeddings; h_{t+1} = Block(h_t + x0) for t < T (input injection each loop); output
head on the masked mean of h_T. Trained with T drawn uniformly from 1..T_max per batch, evaluated at T = 1, 2, 4, 8, 16
(more loops at test than at train probes whether "thinking longer" generalizes).
Aux: on hold'em games, also predict the hero's and villain's hand category (9-way, cross-entropy, weight 0.3) from
the final cards (river: their 7-card hands; turn: their 6-card hands).

usage: easy_capacity.py --data data/easy_big --small data/easy_v1 --out data/easy_big/capacity.jsonl [--only NAME]
"""
import argparse, json, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rlcr_exact import easy_games as G, poker  # noqa: E402


def load(path, n=None):
    out = []
    with open(path) as f:
        for i, l in enumerate(f):
            if n and i >= n:
                break
            out.append(json.loads(l))
    return out


def toks(rows):
    c, r = zip(*[G.tokens(x) for x in rows])
    return torch.tensor(np.array(c)), torch.tensor(np.array(r))


def cats(rows):
    h = poker.eval7_any([x["hero"] + x["board"] for x in rows]); v = poker.eval7_any([x["villain"] + x["board"] for x in rows])
    return torch.tensor(poker.category(h)), torch.tensor(poker.category(v))


class Net(torch.nn.Module):
    def __init__(self, d=64, layers=2, heads=4, looped=False, aux=False):
        super().__init__()
        self.card = torch.nn.Embedding(53, d); self.role = torch.nn.Embedding(4, d)
        mk = lambda: torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.looped = looped
        self.blocks = torch.nn.ModuleList([mk()] if looped else [mk() for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(d); self.out = torch.nn.Linear(d, 1)
        self.aux = torch.nn.Linear(d, 18) if aux else None

    def forward(self, c, r, T=None):
        pad = c == 52; x0 = self.card(c) + self.role(r); h = x0
        if self.looped:
            for _ in range(T):
                h = self.blocks[0](h + x0, src_key_padding_mask=pad)
        else:
            for b in self.blocks:
                h = b(h, src_key_padding_mask=pad)
        h = self.norm((h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True))
        return self.out(h).squeeze(-1), (self.aux(h) if self.aux is not None else None)


def metrics(q, p):
    q, p = np.asarray(q, float), np.asarray(p, float)
    return dict(mse=float(((q - p) ** 2).mean()), corr=float(np.corrcoef(q, p)[0, 1]) if q.std() > 1e-9 else 0.0,
                acc=float(((q > 0.5) == (p > 0.5)).mean()), brier_oracle=float((p * (1 - q) ** 2 + (1 - p) * q ** 2).mean()))


def run(cfg, data, small, log):
    game = cfg["game"]
    src = small if cfg["n"] <= 20000 else data
    tr = load(os.path.join(src, game, "train.jsonl"), cfg["n"]); te = load(os.path.join(data, game, "test.jsonl"))
    ctr, rtr = toks(tr); cte, rte = toks(te)
    y = torch.tensor([r["p_win"] if cfg["target"] == "exact" else float(r["outcome"] > 0) for r in tr], dtype=torch.float32)
    p_te = np.array([r["p_win"] for r in te])
    aux = cfg.get("aux", False)
    if aux:
        ah, av = cats(tr)
    torch.manual_seed(0)
    if cfg["model"] == "mlp":
        Xtr, Xte = torch.tensor(G.onehot(tr)), torch.tensor(G.onehot(te))
        m = torch.nn.Sequential(torch.nn.Linear(156, cfg["d"]), torch.nn.ReLU(), torch.nn.Linear(cfg["d"], cfg["d"]),
                                torch.nn.ReLU(), torch.nn.Linear(cfg["d"], 1), torch.nn.Flatten(0))
    else:
        m = Net(cfg["d"], cfg.get("layers", 2), looped=cfg["model"] == "looped", aux=aux)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1e-3), total_steps=cfg["steps"], pct_start=0.05)
    g = torch.Generator().manual_seed(0); t0 = time.time(); bs = 512
    for s in range(cfg["steps"]):
        idx = torch.randint(0, len(y), (bs,), generator=g)
        opt.zero_grad()
        if cfg["model"] == "mlp":
            out, logits = m(Xtr[idx]), None
        else:
            T = int(torch.randint(1, cfg.get("tmax", 8) + 1, (1,), generator=g)) if cfg["model"] == "looped" else None
            out, logits = m(ctr[idx], rtr[idx], T)
        loss = ((torch.sigmoid(out) - y[idx]) ** 2).mean()
        if logits is not None:
            ce = torch.nn.functional.cross_entropy
            loss = loss + 0.3 * (ce(logits[:, :9], ah[idx]) + ce(logits[:, 9:], av[idx])) / 2
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    res = dict(cfg, seconds=round(time.time() - t0, 1))
    with torch.no_grad():
        if cfg["model"] == "mlp":
            res["test"] = metrics(torch.sigmoid(m(Xte)).numpy(), p_te)
        elif cfg["model"] == "looped":
            res["test_by_T"] = {T: metrics(torch.sigmoid(m(cte, rte, T)[0]).numpy(), p_te) for T in (1, 2, 4, 8, 16)}
            res["test"] = res["test_by_T"][cfg.get("tmax", 8)]
        else:
            res["test"] = metrics(torch.sigmoid(m(cte, rte)[0]).numpy(), p_te)
    log.write(json.dumps(res) + "\n"); log.flush()
    t = res["test"]
    extra = " | by T: " + " ".join(f"{k}:{v['acc']:.3f}/{v['mse']:.4f}" for k, v in res["test_by_T"].items()) if "test_by_T" in res else ""
    print(f"{cfg['name']:28s} mse {t['mse']:.4f} corr {t['corr']:.3f} acc {t['acc']:.3f} {res['seconds']}s{extra}", flush=True)


CONFIGS = [
    # g4 river: hand reading (exact == sampled, deterministic)
    dict(name="g4 settf64x2 20k", game="g4_river", model="settf", d=64, layers=2, n=20000, steps=4000, target="exact"),
    dict(name="g4 settf64x2 200k", game="g4_river", model="settf", d=64, layers=2, n=200000, steps=4000, target="exact"),
    dict(name="g4 settf128x4 200k", game="g4_river", model="settf", d=128, layers=4, n=200000, steps=6000, target="exact"),
    dict(name="g4 settf128x4 200k +aux", game="g4_river", model="settf", d=128, layers=4, n=200000, steps=6000, target="exact", aux=True),
    dict(name="g4 looped128 T1-8 200k", game="g4_river", model="looped", d=128, n=200000, steps=6000, target="exact", tmax=8),
    dict(name="g4 mlp1024 200k", game="g4_river", model="mlp", d=1024, n=200000, steps=6000, target="exact"),
    # g5 turn: outs (exact vs sampled)
    dict(name="g5 settf128x4 200k exact", game="g5_turn", model="settf", d=128, layers=4, n=200000, steps=6000, target="exact"),
    dict(name="g5 settf128x4 200k sampled", game="g5_turn", model="settf", d=128, layers=4, n=200000, steps=6000, target="sampled"),
    dict(name="g5 looped128 T1-8 200k exact", game="g5_turn", model="looped", d=128, n=200000, steps=6000, target="exact", tmax=8),
    # g3 three-card: does 10x data close the sampled gap?
    dict(name="g3 mlp256 200k exact", game="g3_three", model="mlp", d=256, n=200000, steps=6000, target="exact"),
    dict(name="g3 mlp256 200k sampled", game="g3_three", model="mlp", d=256, n=200000, steps=6000, target="sampled"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/easy_big"); ap.add_argument("--small", default="data/easy_v1")
    ap.add_argument("--out", default="data/easy_big/capacity.jsonl"); ap.add_argument("--only", default=None)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--steps-scale", type=float, default=1.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["name"] for l in open(a.out)}
    with open(a.out, "a") as log:
        for cfg in CONFIGS:
            if (a.only and a.only not in cfg["name"]) or cfg["name"] in done:
                continue
            cfg = dict(cfg, steps=max(10, int(cfg["steps"] * a.steps_scale)))
            run(cfg, a.data, a.small, log)


if __name__ == "__main__":
    main()
