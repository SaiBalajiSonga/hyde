"""Generate (features, winner) training data from games between catanatron bots, in-process.

Positions are recorded after every action (with probability --p), from both seats, so the data
covers mid-turn afterstates, which is what the agent evaluates.

    python tools/gen_data.py --games 3000 --workers 8 --out tools/data/ff.npz
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


def _make_player(kind, color, eps):
    from catanatron.models.player import Player
    from catanatron.players.value import ValueFunctionPlayer
    if kind == "F":
        return ValueFunctionPlayer(color, epsilon=eps)
    if kind.startswith("AB"):          # catanatron AlphaBeta, e.g. AB2 or AB2p (pruned)
        from catanatron.players.minimax import AlphaBetaPlayer
        return AlphaBetaPlayer(color, depth=int(kind[2]), prunning=kind.endswith("p"))
    # Our agent, configured by the json file `kind`, wrapped as a catanatron Player.
    import agent as agent_mod
    from catan_kit.obs import Obs
    os.environ["TFI_CFG"] = kind      # Agent() reads its config from here
    ag = agent_mod.Agent()

    class P(Player):
        def decide(self, game, actions):
            if len(actions) == 1:
                return actions[0]
            if eps and random.random() < eps:
                return random.choice(actions)
            obs = Obs(game, self.color)
            obs.cpu_left = 1.0
            return actions[ag.decide(obs, list(actions))]
    return P(color)


def play(args):
    seed, p, eps, kinds = args
    import catan_kit  # noqa: F401
    from catanatron.game import Game
    from catanatron.models.player import Color
    from feats import features

    random.seed(seed)
    rng = random.Random(seed + 7)
    if seed % 2:
        kinds = kinds[::-1]
    players = [_make_player(kinds[0], Color.RED, eps), _make_player(kinds[1], Color.BLUE, eps)]
    game = Game(players, seed=seed, vps_to_win=10)
    rows = []
    while game.winning_color() is None and game.state.num_turns < 1000:
        game.play_tick()
        if rng.random() < p:
            for c in game.state.colors:
                rows.append((features(game, c), c, game.state.num_turns))
    w = game.winning_color()
    if w is None:
        return [], [], []
    X = [r[0] for r in rows]
    y = [int(r[1] == w) for r in rows]
    t = [r[2] for r in rows]
    return X, y, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--p", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=0.03, help="random-move rate, for diversity")
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--a", default="F", help='"F" or an agent config json')
    ap.add_argument("--b", default="F")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    X, y, t, g = [], [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        jobs = [(a.seed0 + i, a.p, a.eps, (a.a, a.b)) for i in range(a.games)]
        for i, (xi, yi, ti) in enumerate(ex.map(play, jobs, chunksize=4)):
            X += xi
            y += yi
            t += ti
            g += [i] * len(yi)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int8),
                        t=np.array(t, dtype=np.int16), g=np.array(g, dtype=np.int32))
    print(f"{len(y)} rows from {a.games} games in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
