"""Learned evaluation: the logit of a win-probability model over feats.features (pure Python/numpy).

A won position always scores above any other, so the search never misses a winning move.
"""
import json

import numpy as np

from feats import BASE_NAMES, NAMES, features

WIN = 1e6


def make_learned_fn(path):
    with open(path) as f:
        m = json.load(f)
    names = m.get("names", NAMES[:len(m["mean"])])
    indices = [NAMES.index(name) for name in names]
    include_extras = any(i >= len(BASE_NAMES) for i in indices)
    identity = indices == list(range(len(indices)))
    mean = np.array(m["mean"])
    inv = 1.0 / np.array(m["scale"])
    if m["kind"] == "linear":
        w = np.array(m["coef"]) * inv
        b = m["intercept"] - float(np.dot(w, mean))
        wl = w.tolist()

        def score(x):
            return b + sum(a * c for a, c in zip(wl, x))
    else:
        W1 = np.array(m["W1"])
        b1 = np.array(m["b1"])
        W2 = np.array(m["W2"])
        b2 = m["b2"]

        def score(x):
            z = (np.asarray(x) - mean) * inv
            h = np.maximum(z @ W1 + b1, 0.0)
            return float(h @ W2 + b2)

    def fn(game, me):
        w = game.winning_color()
        if w is not None:
            return WIN if w == me else -WIN
        x = features(game, me, include_extras=include_extras)
        if not identity or len(x) != len(indices):
            x = [x[i] for i in indices]
        return score(x)

    return fn
