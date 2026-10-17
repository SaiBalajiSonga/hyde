"""Fit a win-probability model on features.py data and write it as model json for learned.py.

    python tools/train_value.py tools/data/ff4000.npz --kind linear --out model_lin.json
    python tools/train_value.py tools/data/ff4000.npz --kind mlp --hidden 32 --out model_mlp.json

Validation is split by GAME, never by row, so positions from one game never sit on both sides.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TFI)
from feats import NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--kind", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--not_my_turn", action="store_true",
                    help="train only on positions where the opponent is to move (end-of-turn judge)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Xs, ys, ts, gs = [], [], [], []
    goff = 0
    for path in a.data:
        d = np.load(path)
        Xs.append(d["X"].astype(np.float64))
        ys.append(d["y"])
        ts.append(d["t"])
        gs.append(d["g"] + goff)
        goff += int(d["g"].max()) + 1
    X, y, t, g = map(np.concatenate, (Xs, ys, ts, gs))
    if a.not_my_turn:
        keep = X[:, NAMES.index("my_turn")] == 0
        X, y, t, g = X[keep], y[keep], t[keep], g[keep]
    val = (g % 5) == 0
    mean = X[~val].mean(axis=0)
    scale = X[~val].std(axis=0) + 1e-6
    Z = (X - mean) / scale
    print(f"rows {len(y)}  train {int((~val).sum())}  val {int(val.sum())}  features {X.shape[1]}")

    if a.kind == "linear":
        m = LogisticRegression(C=a.C, max_iter=2000)
        m.fit(Z[~val], y[~val])
        model = dict(kind="linear", names=NAMES, mean=mean.tolist(), scale=scale.tolist(),
                     coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]))
        top = sorted(zip(NAMES, m.coef_[0]), key=lambda z: -abs(z[1]))
        print("largest standardized coefficients:")
        for n, c in top[:20]:
            print(f"  {n:16s} {c:+.3f}")
    else:
        m = MLPClassifier(hidden_layer_sizes=(a.hidden,), alpha=1e-3, max_iter=60,
                          early_stopping=True, random_state=0)
        m.fit(Z[~val], y[~val])
        model = dict(kind="mlp", names=NAMES, mean=mean.tolist(), scale=scale.tolist(),
                     W1=m.coefs_[0].tolist(), b1=m.intercepts_[0].tolist(),
                     W2=m.coefs_[1][:, 0].tolist(), b2=float(m.intercepts_[1][0]))

    p = m.predict_proba(Z[val])[:, 1]
    print(f"val log-loss {log_loss(y[val], p):.4f}  acc {accuracy_score(y[val], p > 0.5):.4f}")
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 1000)):
        s = val & (t >= lo) & (t < hi)
        if s.sum():
            ps = m.predict_proba(Z[s])[:, 1]
            print(f"  turns {lo:3d}-{hi:<4d} n={int(s.sum()):6d}  acc {accuracy_score(y[s], ps > 0.5):.3f}")
    with open(os.path.join(TFI, a.out) if not os.path.isabs(a.out) else a.out, "w") as f:
        json.dump(model, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
