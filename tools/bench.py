"""Micro-benchmark engine costs on this machine: copy, execute, value fn, sample_world."""
import random
import sys
import time

import catan_kit  # noqa: F401
from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


def main():
    random.seed(1)
    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=1)
    # advance to mid game
    for _ in range(140):
        if g.winning_color() is not None:
            break
        g.play_tick()
    print("turn", g.state.num_turns, "n actions", len(g.playable_actions))
    color = g.state.current_color()
    obs = Obs(g, color)
    rng = random.Random(0)

    def t(label, fn, n):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = (time.perf_counter() - t0) / n
        print(f"{label:20s} {dt*1e6:9.1f} us")

    t("copy", lambda: g.copy(), 2000)
    t("sample_world", lambda: obs.sample_world(rng), 2000)
    t("value_fn", lambda: VALUE_FN(g, color), 2000)
    acts = g.playable_actions

    def ex():
        c = g.copy()
        c.execute(acts[0])
    t("copy+execute", ex, 2000)
    t("Obs()", lambda: Obs(g, color), 2000)
    print(sorted(DEFAULT_WEIGHTS.items()))


if __name__ == "__main__":
    sys.exit(main())
