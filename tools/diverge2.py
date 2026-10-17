"""Fraction of decisions where agent config A and config B choose different moves (same positions)."""
import os, random, sys
from concurrent.futures import ProcessPoolExecutor
TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TFI)


def play(args):
    seed, ca, cb = args
    import catan_kit  # noqa
    import agent as agent_mod
    from catan_kit.obs import Obs
    from catanatron.game import Game
    from catanatron.models.player import Color, Player
    from catanatron.players.value import ValueFunctionPlayer
    os.environ["TFI_CFG"] = ca; A = agent_mod.Agent()
    os.environ["TFI_CFG"] = cb; B = agent_mod.Agent()
    st = {"n": 0, "diff": 0, "end_n": 0, "end_diff": 0}

    class P(Player):
        def decide(self, game, actions):
            actions = list(actions)
            if len(actions) == 1:
                return actions[0]
            obs = Obs(game, self.color); obs.cpu_left = 1.0
            ia = A.decide(obs, actions); ib = B.decide(obs, actions)
            st["n"] += 1; st["diff"] += actions[ia] != actions[ib]
            if any(a.action_type.name == "END_TURN" for a in actions):
                st["end_n"] += 1
                st["end_diff"] += (actions[ia].action_type.name == "END_TURN") != (actions[ib].action_type.name == "END_TURN")
            return actions[ia]
    random.seed(seed)
    me = Color.RED if seed % 2 == 0 else Color.BLUE
    pl = [P(me), ValueFunctionPlayer(Color.BLUE if me == Color.RED else Color.RED)]
    if me == Color.BLUE: pl = pl[::-1]
    Game(pl, seed=seed).play()
    return st


if __name__ == "__main__":
    ca, cb, n = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), int(sys.argv[3])
    tot = {"n": 0, "diff": 0, "end_n": 0, "end_diff": 0}
    with ProcessPoolExecutor(6) as ex:
        for s in ex.map(play, [(820000 + i, ca, cb) for i in range(n)]):
            for k in tot: tot[k] += s[k]
    print(f"decisions {tot['n']}  moves differ {tot['diff']/tot['n']:.1%}  "
          f"end-turn decisions {tot['end_n']}  differ on whether to end the turn {tot['end_diff']/max(1,tot['end_n']):.1%}")
