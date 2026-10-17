"""Play games in-process and summarise how each side scores: VP sources, dev cards, trades, length.

    python tools/stats.py --a tools/cfg/v2base.json --b F --games 200
"""
import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TFI)
sys.path.insert(0, os.path.join(TFI, "tools"))


def play(args):
    seed, kinds = args
    import catan_kit  # noqa: F401
    from catanatron.game import Game
    from catanatron.models.enums import CITY, SETTLEMENT, ActionType
    from catanatron.models.player import Color
    from catanatron.state_functions import player_key
    from gen_data import _make_player

    random.seed(seed)
    swap = seed % 2 == 1
    order = kinds[::-1] if swap else kinds
    players = [_make_player(order[0], Color.RED, 0), _make_player(order[1], Color.BLUE, 0)]
    label = {Color.RED: "b" if swap else "a", Color.BLUE: "a" if swap else "b"}
    game = Game(players, seed=seed, vps_to_win=10)
    acts = defaultdict(Counter)
    first_city_turn = {}
    while game.winning_color() is None and game.state.num_turns < 1000:
        rec = game.play_tick()
        a = rec.action if hasattr(rec, "action") else rec
        acts[label[a.color]][a.action_type.name] += 1
        if a.action_type == ActionType.BUILD_CITY and label[a.color] not in first_city_turn:
            first_city_turn[label[a.color]] = game.state.num_turns
    w = game.winning_color()
    st = game.state
    out = {"winner": label.get(w), "turns": st.num_turns, "first_city": first_city_turn}
    for c in st.colors:
        k = player_key(st, c)
        ps = st.player_state
        out[label[c]] = dict(
            vp=ps[f"{k}_ACTUAL_VICTORY_POINTS"],
            sett=len(st.buildings_by_color[c][SETTLEMENT]), city=len(st.buildings_by_color[c][CITY]),
            lr=ps[f"{k}_HAS_ROAD"], la=ps[f"{k}_HAS_ARMY"],
            vpcards=ps[f"{k}_VICTORY_POINT_IN_HAND"],
            acts=dict(acts[label[c]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default="F")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=800000)
    a = ap.parse_args()
    kinds = (a.a, a.b)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(play, [(a.seed0 + i, kinds) for i in range(a.games)], chunksize=2))
    n = len(res)
    wins = Counter(r["winner"] for r in res)
    print(f"games {n}  A wins {wins['a']}  B wins {wins['b']}  none {wins[None]}  "
          f"mean turns {sum(r['turns'] for r in res) / n:.1f}")
    keys = ["vp", "sett", "city", "lr", "la", "vpcards"]
    acts = ["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT_CARD",
            "PLAY_KNIGHT_CARD", "MARITIME_TRADE", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
            "PLAY_ROAD_BUILDING", "END_TURN"]
    for side in ("a", "b"):
        for subset, name in ((res, "all"), ([r for r in res if r["winner"] == side], "won"),
                             ([r for r in res if r["winner"] not in (side, None)], "lost")):
            if not subset:
                continue
            m = len(subset)
            line = "  ".join(f"{k}={sum(float(r[side][k]) for r in subset) / m:.2f}" for k in keys)
            al = "  ".join(f"{x.split('_')[-1][:6]}={sum(r[side]['acts'].get(x, 0) for r in subset) / m:.1f}"
                           for x in acts)
            fc = [r["first_city"].get(side) for r in subset if r["first_city"].get(side)]
            print(f"{side.upper()} {name:4s} n={m:3d}  {line}  first_city_turn={sum(fc) / max(1, len(fc)):.0f}")
            print(f"           {al}")


if __name__ == "__main__":
    main()
