"""Our value function: catanatron's base_fn (F's evaluation), computed once instead of twice, plus
features F does not have. With every extra weight at 0 it returns exactly what base_fn returns.

Extra features (all from our seat):
  port_2to1      our production of each resource we hold a 2:1 port for (dice-probability units)
  port_3to1      1 if we hold a 3:1 port
  enemy_vps      the opponent's public victory points
  city_ready     number of our settlements, times our wheat+ore production (city potential)
"""
from catanatron.features import (
    count_production,
    get_node_production,
    get_owned_or_buildable,
    get_zero_nodes,
    iter_level_nodes,
)
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.state_functions import (
    get_longest_road_length,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

PROBA_POINT = 2.778 / 100
TRANSLATE_VARIETY = 4

EXTRA_DEFAULTS = {"port_2to1": 0.0, "port_3to1": 0.0, "enemy_vps": 0.0, "city_ready": 0.0}


def _production(state, color, robber):
    m = state.board.map
    b = state.buildings_by_color[color]
    out = []
    for r in RESOURCES:
        p = 0.0
        for n in b[SETTLEMENT]:
            p += get_node_production(m, n, r, robber)
        for n in b[CITY]:
            p += 2 * get_node_production(m, n, r, robber)
        out.append(p)
    return out


def _reachable(game, color):
    """P0_0 and P0_1 road-reachable production sums, as in catanatron's reachability_features."""
    board = game.state.board
    board_buildable = board.buildable_node_ids(color, True)
    owned_or_buildable = get_owned_or_buildable(game, color, board_buildable)
    zero_nodes = get_zero_nodes(game, color)
    p0 = count_production(frozenset(owned_or_buildable.intersection(zero_nodes)), board.map)
    r0 = sum(p0[r] for r in RESOURCES)
    enemy_nodes = frozenset(k for k, v in board.buildings.items() if v is not None and v[0] != color)
    enemy_roads = frozenset(k for k, v in board.roads.items() if v is not None and v != color)
    r1 = 0
    for level, level_nodes, _paths in iter_level_nodes(enemy_nodes, enemy_roads, 1,
                                                      frozenset(zero_nodes)):
        if level == 1:
            p1 = count_production(frozenset(owned_or_buildable.intersection(level_nodes)),
                                  board.map)
            r1 = sum(p1[r] for r in RESOURCES)
    return r0, r1


def make_value_fn(w):
    w = dict(EXTRA_DEFAULTS, **w)
    W_VP, W_PROD, W_EPROD = w["public_vps"], w["production"], w["enemy_production"]
    W_R0, W_R1 = w["reachable_production_0"], w["reachable_production_1"]
    W_SYN, W_NODES, W_TILES = w["hand_synergy"], w["buildable_nodes"], w["num_tiles"]
    W_HAND, W_DISC, W_LR = w["hand_resources"], w["discard_penalty"], w["longest_road"]
    W_DEVS, W_ARMY = w["hand_devs"], w["army_size"]
    W_P2, W_P3, W_EVP, W_CITY = w["port_2to1"], w["port_3to1"], w["enemy_vps"], w["city_ready"]
    extras = any((W_P2, W_P3, W_EVP, W_CITY))

    def fn(game, me):
        state = game.state
        board = state.board
        ps = state.player_state
        colors = state.colors
        opp = colors[1] if colors[0] == me else colors[0]
        key = player_key(state, me)
        robber = board.robber_coordinate

        mine = _production(state, me, robber)
        theirs = _production(state, opp, robber)
        production = sum(mine) + sum(1 for x in mine if x != 0) * TRANSLATE_VARIETY * PROBA_POINT
        enemy_production = sum(theirs)

        r0, r1 = _reachable(game, me)

        wheat = ps[key + "_WHEAT_IN_HAND"]
        ore = ps[key + "_ORE_IN_HAND"]
        sheep = ps[key + "_SHEEP_IN_HAND"]
        brick = ps[key + "_BRICK_IN_HAND"]
        wood = ps[key + "_WOOD_IN_HAND"]
        d_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
        d_sett = (max(1 - wheat, 0) + max(1 - sheep, 0) + max(1 - brick, 0) + max(1 - wood, 0)) / 4.0
        synergy = (2 - d_city - d_sett) / 2

        n_hand = player_num_resource_cards(state, me)
        discard = W_DISC if n_hand > 7 else 0

        b = state.buildings_by_color[me]
        owned_tiles = set()
        adj = board.map.adjacent_tiles
        for n in b[SETTLEMENT]:
            owned_tiles.update(adj[n])
        for n in b[CITY]:
            owned_tiles.update(adj[n])

        n_nodes = len(board.buildable_node_ids(me))
        lr_factor = W_LR if n_nodes == 0 else 0.1

        v = (ps[key + "_VICTORY_POINTS"] * W_VP
             + production * W_PROD
             + enemy_production * W_EPROD
             + r0 * W_R0 + r1 * W_R1
             + synergy * W_SYN
             + n_nodes * W_NODES
             + len(owned_tiles) * W_TILES
             + n_hand * W_HAND
             + discard
             + get_longest_road_length(state, me) * lr_factor
             + player_num_dev_cards(state, me) * W_DEVS
             + ps[key + "_PLAYED_KNIGHT"] * W_ARMY)

        if extras:
            ports = board.get_player_port_resources(me)
            if ports:
                p2 = 0.0
                for i, r in enumerate(RESOURCES):
                    if r in ports:
                        p2 += mine[i]
                v += p2 * W_P2 + (None in ports) * W_P3
            okey = player_key(state, opp)
            v += ps[okey + "_VICTORY_POINTS"] * W_EVP
            v += len(b[SETTLEMENT]) * (mine[3] + mine[4]) * W_CITY
        return float(v)

    return fn
