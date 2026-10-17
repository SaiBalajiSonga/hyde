"""How often does our agent's move differ from F's? Plays in-process games of our agent (config
json) against F and, at every decision we are asked, compares three choices on the same position:
  F       one-ply argmax of F's own evaluation (what F would play here)
  hybrid  one-ply argmax of our hybrid evaluation (F + alpha * learned)
  final   what our agent actually plays (hybrid + lookahead + opening simulation)
Deterministic moves only are compared (moves with a random outcome are scored on one sample).

    python tools/diverge.py --cfg tools/cfg/v3_mlp.json --games 100
"""
import argparse
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TFI)


def play(args):
    seed, cfg = args
    os.environ["TFI_CFG"] = cfg
    import catan_kit  # noqa: F401
    import agent as agent_mod
    from catan_kit.obs import Obs
    from catanatron.game import Game
    from catanatron.models.player import Color, Player
    from catanatron.players.value import ValueFunctionPlayer

    ag = agent_mod.Agent()
    core = ag.cores[0]
    stats = Counter()

    def argmax(world, actions, value, me):
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a, validate_action=False)
                v = value(g, me)
            except Exception:
                continue
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    class P(Player):
        def decide(self, game, actions):
            actions = list(actions)
            if len(actions) == 1:
                return actions[0]
            obs = Obs(game, self.color)
            obs.cpu_left = 1.0
            world = obs.sample_world(random.Random(seed))
            i_f = argmax(world, actions, core.cheap_value, self.color)
            i_h = argmax(world, actions, core.value, self.color)
            i_final = ag.decide(obs, actions)
            kind = actions[i_f].action_type.name
            stats["n"] += 1
            stats[f"n:{kind}"] += 1
            if actions[i_h] != actions[i_f]:
                stats["hyb"] += 1
            if actions[i_final] != actions[i_f]:
                stats["final"] += 1
                stats[f"final:{kind}"] += 1
            if game.state.is_initial_build_phase:
                stats["n:OPENING"] += 1
                stats["final:OPENING"] += actions[i_final] != actions[i_f]
            return actions[i_final]

    random.seed(seed)
    me_first = seed % 2 == 0
    ours, f = P(Color.RED if me_first else Color.BLUE), None
    f = ValueFunctionPlayer(Color.BLUE if me_first else Color.RED)
    players = [ours, f] if me_first else [f, ours]
    game = Game(players, seed=seed)
    game.play()
    stats["won"] += game.winning_color() == ours.color
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tot = Counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for s in ex.map(play, [(810000 + i, os.path.abspath(a.cfg)) for i in range(a.games)]):
            tot.update(s)
    n = tot["n"]
    print(f"games {a.games}  our wins {tot['won']}  decisions {n}")
    print(f"hybrid one-ply differs from F: {tot['hyb'] / n:.1%}")
    print(f"final move differs from F:     {tot['final'] / n:.1%}")
    kinds = sorted({k.split(':', 1)[1] for k in tot if k.startswith("n:")},
                   key=lambda k: -tot[f"n:{k}"])
    for k in kinds:
        m = tot[f"n:{k}"]
        print(f"  F would play {k:22s} n={m:6d}  we differ {tot[f'final:{k}'] / m:6.1%}")


if __name__ == "__main__":
    main()
