"""TFI Catan agent.

F (catanatron's ValueFunctionPlayer) scores every legal move with a 13-feature value function and
plays the best, one move ahead. We keep that evaluation but fix its main blind spot: a move that
looks bad on its own but enables a better position later in the SAME turn (a bank trade that
completes a settlement, for example). For the most promising moves we roll the turn forward
greedily and score the best position reachable before ending it.

Everything runs under a CPU budget derived from obs.cpu_left, and every decision is wrapped so an
error falls back to a plain one-ply choice, then to index 0.
"""
import json
import os
import random
import time

from catanatron.models.enums import ActionType
from catanatron.players.value import DEFAULT_WEIGHTS

from value import make_value_fn

HERE = os.path.dirname(os.path.abspath(__file__))

# Search settings. Times are in seconds of the per-game CPU allowance.
CFG = dict(
    top_k=4,            # moves whose turn we roll forward
    rollout_depth=4,    # extra own moves per rollout
    cap_normal=0.020,   # most we spend on one ordinary decision
    cap_opening=0.150,  # initial placement: most candidates, and the 300 ms base pays for it
    spend_fraction=0.25,  # of the clock above the reserve
    reserve=0.200,      # clock kept back for the rest of the game; below it, one-ply only
    panic=0.040,        # below this, one-ply with the cheap hand-written value only
    open_m=6,           # first-settlement candidates whose placement phase we simulate
    open_fraction=0.6,  # opening: share of the clock above open_reserve we may spend
    open_reserve=0.100,
    weights={},
    model=None,         # json file beside agent.py: learned evaluation
    model_alpha=None,   # None: learned only. A number: F's value + alpha * learned logit
)


def load_cfg():
    """config.json beside this file, then TFI_CFG (a path, for local experiments only)."""
    cfg = dict(CFG)
    for path in (os.path.join(HERE, "config.json"), os.environ.get("TFI_CFG")):
        if path and os.path.exists(path):
            with open(path) as f:
                cfg.update(json.load(f))
    w = dict(DEFAULT_WEIGHTS)
    w.update(cfg["weights"])
    cfg["weights"] = w
    return cfg


class Agent:
    def __init__(self):
        self.rng = random.Random(0)
        self.cfg = load_cfg()
        self.value = make_value_fn(self.cfg["weights"])
        self.cheap_value = self.value
        if self.cfg.get("model"):
            from learned import make_learned_fn
            learned = make_learned_fn(os.path.join(HERE, self.cfg["model"]))
            alpha = self.cfg.get("model_alpha")
            if alpha is None:
                self.value = learned
            else:
                base = self.value

                def hybrid(game, me):
                    return base(game, me) + alpha * learned(game, me)
                self.value = hybrid
        self.end_turn = ActionType.END_TURN

    # ------------------------------------------------------------------ entry point
    def decide(self, obs, playable_actions) -> int:
        t0 = time.perf_counter()
        n = len(playable_actions)
        try:
            i = self._decide(obs, playable_actions, t0)
            if isinstance(i, int) and 0 <= i < n:
                return i
        except Exception:
            pass
        try:
            return self._one_ply_only(obs, playable_actions)
        except Exception:
            return 0

    # ------------------------------------------------------------------ search
    def _budget(self, obs, playable_actions, placing):
        left = obs.cpu_left if obs.cpu_left is not None else 1.0
        if placing:
            spare = max(0.0, left - self.cfg["open_reserve"])
            return min(self.cfg["cap_opening"], self.cfg["open_fraction"] * spare), left
        spare = max(0.0, left - self.cfg["reserve"])
        return min(self.cfg["cap_normal"], self.cfg["spend_fraction"] * spare), left

    def _decide(self, obs, actions, t0):
        world = obs.sample_world(self.rng)
        placing = (actions[0].action_type == ActionType.BUILD_SETTLEMENT
                   and world.state.is_initial_build_phase)
        budget, left = self._budget(obs, actions, placing)
        if left < self.cfg["panic"]:
            return self._one_ply_only(obs, actions, self.cheap_value)
        deadline = t0 + budget
        me = obs.me

        # One ply: the value of the position right after each move.
        children = []
        scores = []
        for a in actions:
            g = world.copy()
            try:
                g.execute(a, validate_action=False)
                v = self.value(g, me)
            except Exception:
                g, v = None, float("-inf")
            children.append(g)
            scores.append(v)

        if budget <= 0:
            return max(range(len(actions)), key=scores.__getitem__)

        if placing:
            return self._placement(actions, children, scores, me, deadline)
        if self.cfg["top_k"] == 0:
            return max(range(len(actions)), key=scores.__getitem__)

        # Same-turn rollouts for the best few moves that keep the turn going.
        order = sorted(range(len(actions)), key=scores.__getitem__, reverse=True)
        final = list(scores)
        for i in order[:self.cfg["top_k"]]:
            if time.perf_counter() > deadline:
                break
            g = children[i]
            if g is None or actions[i].action_type == self.end_turn:
                continue
            final[i] = max(scores[i], self._rollout(g, me, deadline))
        return max(range(len(actions)), key=final.__getitem__)

    def _placement(self, actions, children, scores, me, deadline):
        """Initial settlement: simulate the rest of the placement phase for the best few candidates,
        both seats choosing greedily by our evaluation from their own side, and keep the candidate
        whose finished placement we like best."""
        order = sorted(range(len(actions)), key=scores.__getitem__, reverse=True)
        best_i, best_v = order[0], None
        for i in order[:self.cfg["open_m"]]:
            g = children[i]
            if g is None:
                continue
            while g.state.is_initial_build_phase and g.winning_color() is None:
                if time.perf_counter() > deadline:
                    return best_i if best_v is not None else order[0]
                who = g.state.current_color()
                nxt, nxt_v = None, None
                for a in g.playable_actions:
                    c = g.copy()
                    try:
                        c.execute(a, validate_action=False)
                        v = self.value(c, who)
                    except Exception:
                        continue
                    if nxt_v is None or v > nxt_v:
                        nxt, nxt_v = c, v
                if nxt is None:
                    break
                g = nxt
            v = self.value(g, me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    def _rollout(self, g, me, deadline):
        """Greedy ascent through our own remaining moves this turn. Returns the best value seen."""
        best = float("-inf")
        for _ in range(self.cfg["rollout_depth"]):
            if g.winning_color() is not None:
                return self.value(g, me)
            if g.state.current_color() != me:
                break
            acts = g.playable_actions
            nxt, nxt_v = None, float("-inf")
            for a in acts:
                if time.perf_counter() > deadline:
                    return best
                if a.action_type == self.end_turn:
                    continue
                c = g.copy()
                try:
                    c.execute(a, validate_action=False)
                    v = self.value(c, me)
                except Exception:
                    continue
                if v > nxt_v:
                    nxt, nxt_v = c, v
            if nxt is None:
                break
            best = max(best, nxt_v)
            g = nxt
        return best

    def _one_ply_only(self, obs, actions, value=None):
        value = value or self.value
        world = obs.sample_world(self.rng)
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            g = world.copy()
            try:
                g.execute(a, validate_action=False)
                v = value(g, obs.me)
            except Exception:
                continue
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i
