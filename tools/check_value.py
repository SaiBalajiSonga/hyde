"""Check that value.make_value_fn with default weights equals catanatron's base_fn, and time both."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import catan_kit  # noqa: F401
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, base_fn

from value import make_value_fn

ref = base_fn(DEFAULT_WEIGHTS)
ours = make_value_fn(DEFAULT_WEIGHTS)
states = []
for seed in range(30):
    random.seed(seed)
    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    for t in range(400):
        if g.winning_color() is not None:
            break
        g.play_tick()
        if t % 7 == 0:
            states.append(g.copy())
bad = 0
for g in states:
    for c in g.state.colors:
        a, b = ref(g, c), ours(g, c)
        if abs(a - b) > 1e-6 * max(1, abs(a)):
            bad += 1
print("states", len(states), "mismatches", bad)
for name, f in (("base_fn", ref), ("ours", ours)):
    t0 = time.perf_counter()
    for g in states:
        f(g, Color.RED)
    print(name, f"{(time.perf_counter() - t0) / len(states) * 1e6:.1f} us")
