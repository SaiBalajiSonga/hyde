"""Data for an opening model: features of the position right after the placement phase ends,
from both seats, labelled with who went on to win. Placements are randomised with probability
--rand so the model sees openings F would not choose.

    python tools/gen_opening.py --games 16000 --workers 22 --out tools/data/open16k.npz
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TFI)


def play(args):
    seed, rand_p = args
    import catan_kit  # noqa: F401
    from catanatron.game import Game
    from catanatron.models.enums import ActionType
    from catanatron.models.player import Color
    from catanatron.players.value import ValueFunctionPlayer
    from feats import features

    random.seed(seed)
    rng = random.Random(seed * 7 + 1)

    class P(ValueFunctionPlayer):
        def decide(self, game, actions):
            if (game.state.is_initial_build_phase and len(actions) > 1
                    and actions[0].action_type == ActionType.BUILD_SETTLEMENT
                    and rng.random() < rand_p):
                return rng.choice(actions)
            return super().decide(game, actions)

    game = Game([P(Color.RED), P(Color.BLUE)], seed=seed, vps_to_win=10)
    rows = None
    while game.winning_color() is None and game.state.num_turns < 1000:
        game.play_tick()
        if rows is None and not game.state.is_initial_build_phase:
            rows = [(features(game, c), c) for c in game.state.colors]
    w = game.winning_color()
    if w is None or rows is None:
        return []
    return [(x, int(c == w)) for x, c in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--rand", type=float, default=0.3)
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    X, y, g = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, rows in enumerate(ex.map(play, [(a.seed0 + i, a.rand) for i in range(a.games)],
                                        chunksize=8)):
            for x, lab in rows:
                X.append(x)
                y.append(lab)
                g.append(i)
    np.savez_compressed(a.out, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int8),
                        t=np.zeros(len(y), dtype=np.int16), g=np.array(g, dtype=np.int32))
    print(f"{len(y)} rows from {a.games} games in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
