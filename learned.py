"""Learned evaluation: the logit of a win-probability model over feats.features (pure Python/numpy).

A won position always scores above any other, so the search never misses a winning move.
"""
import json

import numpy as np

from feats import features

WIN = 1e6


def make_learned_fn(path):
    with open(path) as f:
        m = json.load(f)
    mean = np.array(m["mean"])
    n = len(mean)                      # a model uses the first n features it was trained on
    inv = 1.0 / np.array(m["scale"])
    if m["kind"] == "linear":
        w = np.array(m["coef"]) * inv
        b = m["intercept"] - float(np.dot(w, mean))
        wl = w.tolist()

        def score(x):
            return b + sum(a * c for a, c in zip(wl, x))     # zip stops at the model's n
    else:
        W1 = np.array(m["W1"])
        b1 = np.array(m["b1"])
        W2 = np.array(m["W2"])
        b2 = m["b2"]

        def score(x):
            z = (np.asarray(x[:n]) - mean) * inv
            h = np.maximum(z @ W1 + b1, 0.0)
            return float(h @ W2 + b2)

    def fn(game, me):
        w = game.winning_color()
        if w is not None:
            return WIN if w == me else -WIN
        return score(features(game, me))

    return fn
