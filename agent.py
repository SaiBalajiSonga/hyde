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

from catanatron.models.enums import RESOURCES, ActionType
from catanatron.state_functions import player_key
from catanatron.players.value import DEFAULT_WEIGHTS

from value import make_value_fn

HERE = os.path.dirname(os.path.abspath(__file__))
# Last-resort move order when the clock is almost gone: build if possible, never trade or wander.
PRIORITY = {ActionType.BUILD_CITY: 0, ActionType.BUILD_SETTLEMENT: 1, ActionType.ROLL: 2,
            ActionType.BUILD_ROAD: 3, ActionType.PLAY_KNIGHT_CARD: 4,
            ActionType.BUY_DEVELOPMENT_CARD: 5, ActionType.END_TURN: 10}
# Build costs and how much we want each, for discard decisions.
BUILDS = [({"WHEAT": 2, "ORE": 3}, 3.0),
          ({"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1}, 3.0),
          ({"SHEEP": 1, "WHEAT": 1, "ORE": 1}, 1.5),
          ({"WOOD": 1, "BRICK": 1}, 1.0)]
CHANCE_TYPES = (ActionType.BUY_DEVELOPMENT_CARD, ActionType.MOVE_ROBBER, ActionType.ROLL)

# Search settings. Times are in seconds of the per-game CPU allowance.
CFG = dict(
    top_k=4,            # moves whose turn we roll forward
    rollout_depth=4,    # extra own moves per rollout
    cap_normal=0.020,   # most we spend on one ordinary decision
    cap_opening=0.150,  # initial placement: most candidates, and the 300 ms base pays for it
    spend_fraction=0.25,  # of the clock above the reserve
    reserve=0.200,      # clock kept back for the rest of the game; below it, one-ply only
    panic=0.040,        # below this, one-ply with the cheap hand-written value only
    last_resort=0.015,  # below this, a fixed priority by move type: no copies, no evaluation
    chance_n=1,         # samples for moves with a random outcome (4 measured no better)
    open_roads=False,   # also simulate the placement phase for initial road choices
    open_opp="self",    # placement simulation: opponent picks by "self" (our eval) or "F" (F's eval)
    open_m=6,           # first-settlement candidates whose placement phase we simulate
    open_fraction=0.6,  # opening: share of the clock above open_reserve we may spend
    open_reserve=0.100,
    weights={},
    model=None,         # json file beside agent.py: learned evaluation
    model_alpha=None,   # None: learned only. A number: F's value + alpha * learned logit
    judge_model=None,   # json: judge each candidate turn by this model at the END of the turn
    robber_exact=False,  # robber: exact expectation over the card stolen (opponent hand is known)
    discard_smart=False, # discard: keep the hand closest to our best builds
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
        self.judge = None
        if self.cfg.get("judge_model"):
            from learned import make_learned_fn
            self.judge = make_learned_fn(os.path.join(HERE, self.cfg["judge_model"]))

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
        placing = (world.state.is_initial_build_phase and (
            actions[0].action_type == ActionType.BUILD_SETTLEMENT
            or (self.cfg["open_roads"] and actions[0].action_type == ActionType.BUILD_ROAD)))
        budget, left = self._budget(obs, actions, placing)
        if left < self.cfg["last_resort"]:
            return min(range(len(actions)),
                       key=lambda i: PRIORITY.get(actions[i].action_type, 50))
        if left < self.cfg["panic"]:
            return self._one_ply_only(obs, actions, self.cheap_value)
        deadline = t0 + budget
        me = obs.me
        kind = actions[0].action_type
        if self.cfg["discard_smart"] and kind == ActionType.DISCARD_RESOURCE:
            return self._discard(obs, actions)
        if self.cfg["robber_exact"] and kind == ActionType.MOVE_ROBBER:
            return self._robber(world, actions, me)

        # One ply: the value of the position right after each move. A move with a random outcome
        # (dev-card draw, robber steal, dice) is scored by its mean over chance_n samples.
        children = []   # per move: list of (position, value), one per sample
        scores = []
        n_chance = self.cfg["chance_n"]
        for a in actions:
            samples = []
            reps = n_chance if a.action_type in CHANCE_TYPES else 1
            for r in range(reps):
                if r == 0:
                    g = world.copy()
                elif a.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                    g = obs.sample_world(self.rng)     # a fresh deck order: a different card
                else:
                    g = world.copy()                   # the engine's own random differs per call
                try:
                    g.execute(a, validate_action=False)
                    samples.append((g, self.value(g, me)))
                except Exception:
                    pass
            children.append(samples)
            scores.append(sum(v for _, v in samples) / len(samples) if samples else float("-inf"))

        if budget <= 0:
            return max(range(len(actions)), key=scores.__getitem__)

        if placing:
            return self._placement(actions, children, scores, me, deadline)
        if self.cfg["top_k"] == 0:
            return max(range(len(actions)), key=scores.__getitem__)

        order = sorted(range(len(actions)), key=scores.__getitem__, reverse=True)
        if self.judge is not None:
            i = self._judge_turns(actions, children, order, me, deadline)
            if i is not None:
                return i

        # Same-turn rollouts for the best few moves that keep the turn going.
        final = list(scores)
        for i in order[:self.cfg["top_k"]]:
            if time.perf_counter() > deadline:
                break
            if not children[i] or actions[i].action_type == self.end_turn:
                continue
            final[i] = sum(max(v, self._rollout(g, me, deadline))
                           for g, v in children[i]) / len(children[i])
        return max(range(len(actions)), key=final.__getitem__)

    def _placement(self, actions, children, scores, me, deadline):
        """Initial settlement: simulate the rest of the placement phase for the best few candidates,
        both seats choosing greedily by our evaluation from their own side, and keep the candidate
        whose finished placement we like best."""
        order = sorted(range(len(actions)), key=scores.__getitem__, reverse=True)
        best_i, best_v = order[0], None
        for i in order[:self.cfg["open_m"]]:
            if not children[i]:
                continue
            g = children[i][0][0]
            while g.state.is_initial_build_phase and g.winning_color() is None:
                if time.perf_counter() > deadline:
                    return best_i if best_v is not None else order[0]
                who = g.state.current_color()
                val = self.cheap_value if (who != me and self.cfg["open_opp"] == "F") else self.value
                nxt, nxt_v = None, None
                for a in g.playable_actions:
                    c = g.copy()
                    try:
                        c.execute(a, validate_action=False)
                        v = val(c, who)
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

    def _robber(self, world, actions, me):
        """Each robber move scored by its exact expected value: every card the victim could lose,
        weighted by how many of it they hold (their hand is public information here)."""
        from catanatron.models.enums import ActionRecord
        ps = world.state.player_state
        best_i, best_v = 0, None
        for i, a in enumerate(actions):
            victim = a.value[1]
            outcomes = []
            if victim is not None:
                vk = player_key(world.state, victim)
                hand = {r: ps[f"{vk}_{r}_IN_HAND"] for r in RESOURCES}
                total = sum(hand.values())
                outcomes = [(r, n / total) for r, n in hand.items() if n] if total else []
            try:
                if not outcomes:
                    g = world.copy()
                    g.execute(a, validate_action=False)
                    v = self.value(g, me)
                else:
                    v = 0.0
                    for r, pr in outcomes:
                        g = world.copy()
                        g.execute(a, validate_action=False, action_record=ActionRecord(action=a, result=r))
                        v += pr * self.value(g, me)
            except Exception:
                continue
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    def _discard(self, obs, actions):
        """Discard the card whose loss leaves the hand closest to our most valuable builds; on ties,
        the resource we hold most of."""
        hand = dict(obs.my_resources)

        def score(h):
            s = 0.0
            for cost, w in BUILDS:
                need = sum(cost.values())
                have = sum(min(h.get(r, 0), n) for r, n in cost.items())
                s += w * have / need
            return s

        best_i, best_key = 0, None
        for i, a in enumerate(actions):
            r = a.value
            if r is None or hand.get(r, 0) <= 0:
                continue
            h = dict(hand)
            h[r] -= 1
            key = (score(h), hand[r])
            if best_key is None or key > best_key:
                best_i, best_key = i, key
        return best_i

    def _judge_turns(self, actions, children, order, me, deadline):
        """Plan the rest of the turn greedily for the best few moves (and for ending the turn now),
        end the turn, and pick the move whose end-of-turn position the judge model likes best."""
        cands = list(order[:self.cfg["top_k"]])
        for i, a in enumerate(actions):
            if a.action_type == self.end_turn and i not in cands:
                cands.append(i)
        best_i, best_v = None, None
        for i in cands:
            if time.perf_counter() > deadline:
                break
            if not children[i]:
                continue
            g = children[i][0][0]
            if actions[i].action_type != self.end_turn:
                g = self._rollout_state(g, me, deadline)
                if g.state.current_color() == me and g.winning_color() is None:
                    for a in g.playable_actions:
                        if a.action_type == self.end_turn:
                            g = g.copy()
                            g.execute(a, validate_action=False)
                            break
            v = self.judge(g, me)
            if best_v is None or v > best_v:
                best_i, best_v = i, v
        return best_i

    def _rollout_state(self, g, me, deadline):
        """Greedy ascent through our own moves this turn; returns the best position on the path."""
        best_g, best_v = g, self.value(g, me)
        for _ in range(self.cfg["rollout_depth"]):
            if g.winning_color() is not None or g.state.current_color() != me:
                break
            nxt, nxt_v = None, float("-inf")
            for a in g.playable_actions:
                if time.perf_counter() > deadline:
                    return best_g
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
            g = nxt
            if nxt_v > best_v:
                best_g, best_v = nxt, nxt_v
        return best_g

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
