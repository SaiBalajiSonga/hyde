# Experiment log (TFI)

All runs use the kit's harness (`catan_kit.match`, which `evaluate` also uses), with seats alternated and each board played from both seats with the same dice.
- **Dev boards:** seed0 = 990000, the kit's default.
- **Holdout boards:** seed0 = 700000, used only to confirm.
- **Weight-tuning boards:** 100000+.
- **Model training games:** 500000 to 650000.

The raw rows are in `tools/results.csv`. All runs were on a 24-core laptop. F measures 186 to 223 ms per game here, against 158 ms on the scoring machine, so CPU numbers here are conservative.

## Engine costs (laptop, per call)

| Operation | Cost |
|---|---|
| `game.copy()` | 19 µs |
| copy + `execute` | 25 µs |
| `sample_world` | 33 µs |
| F's `base_fn` | 65 µs |
| our `value.py` (identical output on 1,740 positions) | 22 µs |

## Results against F (win rate ± 95% CI)

| # | Agent | Games | Dev | Holdout | Mean / worst CPU (ms) | Finding |
|---|---|---|---|---|---|---|
| 0 | F vs F (sanity) | 400 | 0.483 ± .049 | | 223 / 578 | The harness is fair |
| 0b | Ours, one-ply with F's weights | 400 | 0.483 | | 221 / 516 | Reproduces F exactly (193/400) |
| 1 | F eval + same-turn lookahead, top 4 | 400 | 0.515 ± .049 | | 409 / 1266 | Lookahead alone gives little |
| 2 | catanatron CONTENDER weights | 400 | 0.450 / 0.488 (+lookahead) | | | Not better |
| 3 | Weight tuning, CEM over 12 log-weights, 8 gens × 10 × 200 games | | ~0.50 flat | | | **Negative:** F's weights are near a local optimum, or gains are below the noise |
| 4 | Learned linear model only, one-ply | 400 | 0.263 ± .043 | | 242 / 828 | Model alone is bad within a turn |
| 5 | Learned linear model only + lookahead | 400 | 0.450 | | 379 / 1031 | Lookahead rescues it a lot |
| 6 | **Hybrid** F + α·logit, α = 1e6, one-ply | 400 | 0.573 ± .049 | | 222 / 703 | Hybrid works |
| 6b | α = 3e5 / 3e6 / 1e7 / 1e8 | 400 | 0.523 / 0.550 / 0.550 / 0.465 | | | α ≈ 1e6 is best. Too large breaks F's move ordering |
| 7 | **Hybrid α = 1e6 + lookahead top 4 (v1)** | 400 / 1000 | 0.643 ± .047 | **0.622 ± .030** | 346 / 891 | Hybrid and lookahead reinforce each other |
| 7b | top 8 / depth 7 / α = 3e5, 3e6 | 400 | 0.635 / 0.635 / 0.643 / 0.650 | | | Plateau: more search within a turn doesn't help |
| 8 | Linear model retrained on F + own games (DAgger-style) | 600 | 0.562 ± .040 | | 367 / 1453 | **Negative**: linear model got worse |
| 9 | MLP (32 hidden units) on the same data | 600 / 1000 | 0.658 ± .038 | 0.598 ± .030 | 414 / 1516 (1 over) | Not better on holdout, and costs more CPU |
| 10 | Linear model on 16k F-vs-F games | 1000 | 0.627 ± .030 | 0.618 ± .030 | 345 / 1391 | More data gives the same result |
| 11 | **v2 = v1 + opening placement simulation (6 candidates)** | 1000 | **0.684 ± .029** | **0.672 ± .029** | 460 / 1141 | **+3.7 pp over the same agent with the plain opening** (0.642 / 0.640) |

Hybrid α: the learned model's logit is scaled by α and added to F's value. F's own scale is 3e14 per VP and 1e8 per unit of production.

## Afternoon: trying to improve on v2 (all against F, v2 settings unless stated)

| # | Change on top of v2 | Games | Dev | Holdout | Finding |
|---|---|---|---|---|---|
| 12 | Opening simulation with 3 / 10 / 16 candidates (v2 uses 6) | 1000 | 0.663 / 0.690 / 0.678 | | Plateau from 6 up |
| 13 | α = 3e5 / 3e6 / 1e7 (v2 uses 1e6) | 1000 | 0.682 / 0.686 / 0.694 | | Insensitive once lookahead is on |
| 14 | Chance moves averaged over 4 samples (expectimax at depth 1) | 1000 | 0.663 | 0.658 | **Negative**: no gain, +100 ms per game |
| 15 | Opening also simulates initial roads | 1000 | 0.675 | 0.691 | Within noise of v2 (0.675 / 0.671 in the same session) |
| 16 | F's lower weights × up to 1e6 (dev cards in hand, army, synergy), one-ply | 400 | 0.20 to 0.48 | | **Negative**: dev-card weight makes it hoard, the rest changes nothing |
| 17 | Own hidden VP cards counted as VPs + expected-value dev buys | 600 | 0.672 | | Same as v2 |
| 18 | + value on knights in hand | 600 | 0.50 to 0.53 | | **Negative**, and 21 to 23 games over the clock |
| 19 | Wide CEM over 17 weights (σ = 2, ×7 per step), hybrid one-ply | 12 × 240 | pop. mean 0.562, best 0.588 | | Locally flat landscape. Run stopped (out of memory) |
| 20 | "Judge" mode: plan the turn with the hybrid, score the end-of-turn position with a model trained on opponent-to-move positions | 400 | 0.625 (top 4) / 0.673 (top 6) | | No gain |
| 21 | Opening simulation models the opponent with F's evaluation | 1000 | | 0.639 | **Negative** (v2: 0.672) |
| 22 | Expansion in the production tier: own reach ×1e3 / opponent reach −1e7 / opponent spots −1e6 | 600 | 0.518 / 0.652 / 0.685 | | Own-reach upweighting is strongly negative; blocking terms neutral |
| 23 | MLP (32 units) on 16k F-vs-F games: val log-loss 0.472 vs 0.491 for the linear model | 1000 | 0.678 | 0.667 | **Better prediction did not mean better play** |

### Ablation of v2 (1000 games, dev boards)

| Variant | Win rate vs F | Δ vs full v2 |
|---|---|---|
| Full v2 | 0.684 | |
| Without opening simulation | 0.642 | −4.2 pp |
| Without same-turn lookahead | 0.600 | −8.4 pp |
| Without learned model (F eval + lookahead + opening) | 0.517 | −16.7 pp |

### How v2 wins (300 in-process games vs F, end-of-game averages)

| | v2 | F |
|---|---|---|
| Holds longest road | 0.69 | 0.30 |
| Holds largest army | 0.28 | 0.03 |
| Dev cards bought | 3.3 | 0.9 |
| Knights played | 1.8 | 0.5 |
| VP cards | 0.71 | 0.17 |
| Settlements + cities | 4.0 | 4.5 |

F almost never contests largest army, and loses longest road most of the time. v2 builds slightly *fewer* buildings than F and wins on the 4 VP from longest road and largest army.

### Team machine check (Linux, official `catan_kit.evaluate`, 200 games)

v2: 0.700 ± 0.064 vs F. CPU 382 ms mean, 710 ms worst. 0 over the clock, 0 errors over 9,950 decisions. Load time 0.18 s.

## Head-to-head and robustness (row A wins)

| A | B | Games | Win rate |
|---|---|---|---|
| v1 | MLP hybrid | 400 | 0.533 ± .049 |
| v1 | learned-only + lookahead | 400 | 0.663 ± .046 |
| v1 | F + lookahead | 400 | 0.645 ± .047 |
| v1 | random | 400 | 1.000 |
| **v2** | **v1** | 1000 | **0.546 ± .031** |

## Learned model (linear, 50 features, 261k positions from 4,000 F-vs-F games)

- **Held-out accuracy**, split by game: 0.743 overall. By game turn: 0.62 (turns 0 to 20), 0.75 (20 to 40), 0.80 (40 and later).
- **Largest standardized coefficients:**

  | Feature | Coefficient |
  |---|---|
  | `my_city` | +0.71 |
  | `opp_vps` | −0.71 |
  | `opp_city` | −0.70 |
  | `my_vps` | +0.63 |
  | `opp_road` length | −0.56 |
  | `my_r0` (reachable production) | +0.55 |
  | `my_road` | +0.30 |
  | `my_nodes` (buildable spots) | −0.23 |

  The sign of `my_nodes` is the opposite of F's hand-set +1e3.

## Submissions

| Zip | What it is | Local result vs F |
|---|---|---|
| `tfi_v0_safeF.zip` | F + safety wrapper (pipeline test, baseline vs O) | ≈ 0.48 |
| `tfi_v1_hybrid_k4.zip` | Exp. 7 | 0.622 holdout |
| `tfi_v2_open6.zip` | Exp. 11 | 0.672 holdout |
