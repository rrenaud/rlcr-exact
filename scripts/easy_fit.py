"""Fastest models on the easy-game ladder (scripts/easy_data.py output), CPU only, exact vs sampled targets.

Models: const (train mean), logreg (linear on role-tagged one-hot, 156 dims), MLP 256 (same input), and a small
set transformer (card + role embeddings, 2 layers, d 64, mean pool), all with a sigmoid output trained on squared
error (Brier). Targets: "exact" = p_win, "sampled" = the row's one sampled outcome (win = 1). Test metrics against
the exact p_win: MSE, correlation, Brier vs oracle, decision accuracy (p > 0.5 side), and the floor E[p(1-p)].

usage: easy_fit.py DATA_DIR [--games ...] [--n-train 20000] [--out results.json]
"""
import argparse, json, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rlcr_exact import easy_games as G  # noqa: E402


def load(path):
    return [json.loads(l) for l in open(path)]


def metrics(q, p):
    q, p = np.asarray(q, float), np.asarray(p, float)
    return dict(mse=float(((q - p) ** 2).mean()), corr=float(np.corrcoef(q, p)[0, 1]) if q.std() > 1e-9 and p.std() > 1e-9 else 0.0,
                brier_oracle=float((p * (1 - q) ** 2 + (1 - p) * q ** 2).mean()), floor=float((p * (1 - p)).mean()),
                acc=float(((q > 0.5) == (p > 0.5)).mean()))


class SetTransformer(torch.nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.card = torch.nn.Embedding(53, d); self.role = torch.nn.Embedding(4, d)
        enc = torch.nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(enc, layers); self.out = torch.nn.Linear(d, 1)

    def forward(self, c, r):
        pad = c == 52
        h = self.enc(self.card(c) + self.role(r), src_key_padding_mask=pad)
        h = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True)
        return self.out(h).squeeze(-1)


def train(model, inputs, y, steps, bs=256, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(y); g = torch.Generator().manual_seed(0)
    for s in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), generator=g) if bs < n else torch.arange(n)
        opt.zero_grad()
        loss = ((torch.sigmoid(model(*[x[idx] for x in inputs])) - y[idx]) ** 2).mean()
        loss.backward(); opt.step()
    return model


def predict(model, inputs):
    with torch.no_grad():
        return torch.sigmoid(model(*inputs)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--games", default=",".join(G.GAMES))
    ap.add_argument("--n-train", type=int, default=20000); ap.add_argument("--out", default=None)
    ap.add_argument("--mlp-steps", type=int, default=1500); ap.add_argument("--tf-steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    results = []
    for game in a.games.split(","):
        tr = load(os.path.join(a.data, game, "train.jsonl"))[: a.n_train]; te = load(os.path.join(a.data, game, "test.jsonl"))
        p_te = np.array([r["p_win"] for r in te])
        Xtr, Xte = torch.tensor(G.onehot(tr)), torch.tensor(G.onehot(te))
        tok = lambda rows: [torch.tensor(np.array(z)) for z in zip(*[G.tokens(r) for r in rows])]
        Ttr, Tte = tok(tr), tok(te)
        targets = {"exact": np.array([r["p_win"] for r in tr], np.float32),
                   "sampled": np.array([float(r["outcome"] > 0) for r in tr], np.float32)}
        res = dict(game=game, n_train=len(tr), n_test=len(te), floor=float((p_te * (1 - p_te)).mean()),
                   const=metrics(np.full(len(te), targets["exact"].mean()), p_te), models={})
        for tname, y in targets.items():
            yt = torch.tensor(y)
            for mname, build, inp_tr, inp_te, steps in [
                    ("logreg", lambda: torch.nn.Sequential(torch.nn.Linear(156, 1), torch.nn.Flatten(0)), [Xtr], [Xte], a.mlp_steps),
                    ("mlp256", lambda: torch.nn.Sequential(torch.nn.Linear(156, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                                                           torch.nn.ReLU(), torch.nn.Linear(256, 1), torch.nn.Flatten(0)), [Xtr], [Xte], a.mlp_steps),
                    ("settf", lambda: SetTransformer(), Ttr, Tte, a.tf_steps)]:
                torch.manual_seed(0); t = time.time()
                m = train(build(), inp_tr, yt, steps)
                r = metrics(predict(m, inp_te), p_te); r["seconds"] = round(time.time() - t, 1)
                res["models"][f"{mname}/{tname}"] = r
                print(f"{game:10s} {mname:7s} {tname:8s} mse {r['mse']:.4f} corr {r['corr']:.3f} acc {r['acc']:.3f} "
                      f"brierO {r['brier_oracle']:.4f} (floor {res['floor']:.4f}, const mse {res['const']['mse']:.4f}) "
                      f"{r['seconds']}s", flush=True)
        results.append(res)
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
