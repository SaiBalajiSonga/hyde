"""Features of a position from one seat, for the learned evaluation.

Everything here is visible to that seat: the opponent's dev cards are used only as a COUNT, and
victory points are public ones (plus our own hidden ones, which we know).
"""
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.state_functions import get_longest_road_length, player_key

from value import _production, _reachable

NAMES = (
    ["my_vps", "opp_vps", "my_turn"]
    + [f"my_prod_{r}" for r in RESOURCES] + [f"opp_prod_{r}" for r in RESOURCES]
    + ["my_variety", "opp_variety", "my_r0", "my_r1", "my_nodes", "my_tiles"]
    + [f"my_hand_{r}" for r in RESOURCES]
    + ["my_hand", "opp_hand", "my_over7", "opp_over7", "my_synergy", "my_d_city", "my_d_sett"]
    + ["my_road", "opp_road", "my_lr", "opp_lr", "my_knights", "opp_knights", "my_la", "opp_la"]
    + ["my_devs", "opp_devs", "my_sett", "opp_sett", "my_city", "opp_city"]
    + ["my_port2", "opp_port2", "my_port3", "opp_port3", "my_hidden_vp"]
)


def _ports(board, color, prod):
    ports = board.get_player_port_resources(color)
    if not ports:
        return 0.0, 0.0
    p2 = sum(prod[i] for i, r in enumerate(RESOURCES) if r in ports)
    return p2, float(None in ports)


def features(game, me):
    st = game.state
    board = st.board
    ps = st.player_state
    colors = st.colors
    opp = colors[1] if colors[0] == me else colors[0]
    k, o = player_key(st, me), player_key(st, opp)
    robber = board.robber_coordinate

    mine = _production(st, me, robber)
    theirs = _production(st, opp, robber)
    r0, r1 = _reachable(game, me)

    hand = [ps[f"{k}_{r}_IN_HAND"] for r in RESOURCES]
    n_hand = sum(hand)
    o_hand = sum(ps[f"{o}_{r}_IN_HAND"] for r in RESOURCES)
    wood, brick, sheep, wheat, ore = hand
    d_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
    d_sett = (max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0

    b, ob = st.buildings_by_color[me], st.buildings_by_color[opp]
    tiles = set()
    adj = board.map.adjacent_tiles
    for n in b[SETTLEMENT] + b[CITY]:
        tiles.update(adj[n])

    my_p2, my_p3 = _ports(board, me, mine)
    op_p2, op_p3 = _ports(board, opp, theirs)
    devs = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
    my_devs = sum(ps[f"{k}_{d}_IN_HAND"] for d in devs)
    opp_devs = sum(ps[f"{o}_{d}_IN_HAND"] for d in devs)

    return [
        ps[f"{k}_VICTORY_POINTS"], ps[f"{o}_VICTORY_POINTS"], float(st.current_color() == me),
        *mine, *theirs,
        sum(1 for x in mine if x), sum(1 for x in theirs if x),
        r0, r1, len(board.buildable_node_ids(me)), len(tiles),
        *hand,
        n_hand, o_hand, float(n_hand > 7), float(o_hand > 7), (2 - d_city - d_sett) / 2,
        d_city, d_sett,
        get_longest_road_length(st, me), get_longest_road_length(st, opp),
        float(ps[f"{k}_HAS_ROAD"]), float(ps[f"{o}_HAS_ROAD"]),
        ps[f"{k}_PLAYED_KNIGHT"], ps[f"{o}_PLAYED_KNIGHT"],
        float(ps[f"{k}_HAS_ARMY"]), float(ps[f"{o}_HAS_ARMY"]),
        my_devs, opp_devs,
        len(b[SETTLEMENT]), len(ob[SETTLEMENT]), len(b[CITY]), len(ob[CITY]),
        my_p2, op_p2, my_p3, op_p3,
        ps[f"{k}_ACTUAL_VICTORY_POINTS"] - ps[f"{k}_VICTORY_POINTS"],
    ]
