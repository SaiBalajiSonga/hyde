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
    ap.add_argument("--identity", action="store_true",
                    help="last 2 columns are (me is F, opponent is F): train with them, then fold "
                         "them in as constants (me=our agent, opponent=F-like) so play ignores them")
    ap.add_argument("--target", choices=["win", "margin"], default="win",
                    help="win: logistic on the winner; margin: regression on the final VP margin")
    ap.add_argument("--nfeat", type=int, default=0, help="use only the first n feature columns")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Xs, ys, ts, gs, ms = [], [], [], [], []
    goff = 0
    for path in a.data:
        d = np.load(path)
        Xi = d["X"].astype(np.float64)
        Xs.append(Xi[:, :a.nfeat] if a.nfeat else Xi)
        ys.append(d["y"])
        ts.append(d["t"])
        gs.append(d["g"] + goff)
        ms.append(d["m"] if "m" in d else np.zeros(len(d["y"]), dtype=np.int8))
        goff += int(d["g"].max()) + 1
    X, y, t, g, mg = map(np.concatenate, (Xs, ys, ts, gs, ms))
    if a.nfeat:
        X = X[:, :a.nfeat]
    if a.not_my_turn:
        keep = X[:, NAMES.index("my_turn")] == 0
        X, y, t, g, mg = X[keep], y[keep], t[keep], g[keep], mg[keep]
    val = (g % 5) == 0
    names = list(NAMES)[:X.shape[1] - (2 if a.identity else 0)] + (["me_is_F", "opp_is_F"] if a.identity else [])
    assert X.shape[1] == len(names), (X.shape, len(names))
    mean = X[~val].mean(axis=0)
    scale = X[~val].std(axis=0) + 1e-6
    Z = (X - mean) / scale
    print(f"rows {len(y)}  train {int((~val).sum())}  val {int(val.sum())}  features {X.shape[1]}")

    if a.kind == "linear":
        m = LogisticRegression(C=a.C, max_iter=2000)
        m.fit(Z[~val], y[~val])
        k = X.shape[1] - (2 if a.identity else 0)
        model = dict(kind="linear", names=names[:k], mean=mean[:k].tolist(), scale=scale[:k].tolist(),
                     coef=m.coef_[0][:k].tolist(), intercept=float(m.intercept_[0]))
        top = sorted(zip(names, m.coef_[0]), key=lambda z: -abs(z[1]))
        print("largest standardized coefficients:")
        for n, c in top[:20]:
            print(f"  {n:16s} {c:+.3f}")
    elif a.target == "margin":
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(a.hidden,), alpha=1e-3, max_iter=60,
                         early_stopping=True, random_state=0)
        m.fit(Z[~val], mg[~val].astype(float))
        pred = m.predict(Z[val])
        print(f"margin regression: val corr {np.corrcoef(pred, mg[val])[0, 1]:.3f}  "
              f"sign acc vs winner {((pred > 0) == (y[val] == 1)).mean():.4f}")
        model = dict(kind="mlp", names=names, mean=mean.tolist(), scale=scale.tolist(),
                     W1=m.coefs_[0].tolist(), b1=m.intercepts_[0].tolist(),
                     W2=m.coefs_[1][:, 0].tolist(), b2=float(m.intercepts_[1][0]))
        with open(os.path.join(TFI, a.out) if not os.path.isabs(a.out) else a.out, "w") as f:
            json.dump(model, f)
        print("wrote", a.out)
        return
    else:
        m = MLPClassifier(hidden_layer_sizes=(a.hidden,), alpha=1e-3, max_iter=60,
                          early_stopping=True, random_state=0)
        m.fit(Z[~val], y[~val])
        W1, b1 = m.coefs_[0], m.intercepts_[0].copy()
        if a.identity:                         # fold me_is_F=0, opp_is_F=1 into the bias
            for j, val_ in ((-2, 0.0), (-1, 1.0)):
                b1 += W1[j] * ((val_ - mean[j]) / scale[j])
            W1 = W1[:-2]
        k = W1.shape[0]
        model = dict(kind="mlp", names=names[:k], mean=mean[:k].tolist(), scale=scale[:k].tolist(),
                     W1=W1.tolist(), b1=b1.tolist(),
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
