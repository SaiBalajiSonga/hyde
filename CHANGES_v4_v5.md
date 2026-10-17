# Changes from v4 to v5

**Short version:** v5 is **v3 with one safety change**, the long-game CPU fallback. The v4 bandit is **removed**, because a cross-check tournament showed it adds nothing.

| | v3 (`tfi_v3_mlp.zip`) | v4 (`tfi_v4_bandit.zip`) | **v5 (`tfi_v5_safe.zip`)** |
|---|---|---|---|
| Win model | MLP 32 (`model_mlpff32.json`) | 3 arms: MLP 32 / linear / MLP 64, picked per game by Thompson sampling | **MLP 32 only** (same as v3) |
| Same-turn lookahead | top 4, depth 4 | same | same |
| Opening simulation | 6 candidates | same | same |
| Long-game CPU safety (`cheap_in_reserve`) | no | yes | **yes** |
| Uploaded | yes (#2) | **no** (the upload failed: wrong file name, and the zip was never copied) | yes (#4) |

v5 config:
```json
{"top_k":4, "model":"model_mlpff32.json", "model_alpha":1e6, "open_m":6, "cheap_in_reserve":true}
```

---

## Portal results (public boards, 1,000 games per cell)

| # | Upload | vs F | vs O | Mean | CPU mean / worst | Over clock |
|---|---|---|---|---|---|---|
| 1 | v2 | 0.649 ± .030 | 0.487 ± .031 | 0.568 | 368 / 1029 ms | 0 |
| 2 | v3 | 0.698 ± .029 | 0.505 ± .031 | 0.602 | 450 / 1501 ms | 1 |
| 3 | tfi_mixed8000_extra *(not built in this session; contents to be documented by whoever uploaded it)* | 0.694 ± .029 | 0.538 ± .031 | 0.616 | 484 / 1453 ms | 0 |
| 4 | **v5** | 0.680 ± .029 | 0.480 ± .031 | 0.580 | 456 / 1412 ms | **0** |

- **The safety change did its job:** v5 had 0 over-clock games, against 1 for v3, and its worst game dropped from 1,501 to 1,412 ms.
- **v3 and v5 are the same agent apart from the end of very long games**, yet the portal shows them 1.8 pp apart against F and 2.5 pp apart against O. That's the portal's noise made visible: differences of that size between our uploads are not signal.

## Why the bandit was dropped

A cross-check tournament (36 matches × 500 games, same boards for every match, each block of 50 games played in one agent process so the bandit can learn) gave:

| | Overall vs field | **Worst half** (how groups are scored) | Worst opponent | Over-clock games |
|---|---|---|---|---|
| v2 | 0.683 | 0.624 | mixed-data MLP, 0.502 | 0 |
| **v3** | **0.698** | **0.644** | v1, 0.538 | 3 |
| v4 | 0.685 | 0.623 | mixed-data MLP, 0.534 | 1 |

The field is F, F_build, F_dev, F_road, F_city, F_contender, F with lookahead, v1, learned-only and the mixed-data MLP.

- The arms all used F's evaluation with the same search and differed only in the predictor on top. That made them nearly identical, so there was nothing to adapt between.
- Head-to-head, v3 and v4 tie: 0.500 in the tournament and 0.509 in an earlier 1,000-game run.
- The self-labelled win signal (61% detected against 67% true) was also a source of bias. A later fix gives VP-card draws a fractional label, but it no longer matters with a single strategy.

## Why the CPU safety stays

- v3 went over the clock in 1 of 2,000 games on the scoring machine (worst game 1,501 ms) and in 3 tournament games locally.
- The cause is long games. Past 96 decisions the clock stops growing (1.5 s ceiling), but each decision still costs about 1 ms to receive the game state, plus about 100 µs per legal move for the MLP hybrid.
- **The change:** once the clock drops into the 200 ms reserve, the agent plays one-ply with **F's cheap evaluation** instead of the hybrid. It's the only behavioural difference from v3, and it only applies at the very end of long games.
- Official `catan_kit.evaluate` smoke test of the v5 zip: 0 errors in 1,190 decisions, 0 over the clock, 0.33 s load time.

## Code in the v5 zip that is present but switched off

These are in `agent.py`, but v5's config doesn't enable them, so they don't affect play:

- `win_search`: an exhaustive search for a winning line at 7+ actual VP. Tested against v3: **0.499 head-to-head** (no gain), so it is off.
- `arms` / bandit machinery: unused with a single strategy.
- `robber_exact`, `discard_smart`, `open_model`, `judge_model`, `open_roads`, `open_opp`, `chance_n` > 1: all tested earlier, none better (see `EXPERIMENTS.md`).
- `feats.py`'s 3 extra features (robber exposure and road margin) are **not computed** unless a model uses more than 50 features. v5's model uses 50.

---

## Tested after v4 and not shipped in v5

| Attempt | Test | Result | Reading |
|---|---|---|---|
| Winning-line search (exhaustive own-turn search at 7+ actual VP) | vs v3, 1,000 games | 0.499 | Tie |
| CPU cut: `top_k` 2, 12 ms cap | vs F, 1,000 games | 0.652 (v3: 0.678) | −2.6 pp for −8% CPU; not worth it |
| **On-policy retrain:** v3 self-play only (4,000 games), win labels, 50 features | vs v3, 1,000 games | **0.442** | **Worse**, beyond the ±4 pp noise |
| Same data, win labels, 53 features (+robber exposure, road margin) | vs v3, 1,000 games | **0.443** | Worse; the extra features change nothing |
| Same data, **final VP-margin labels**, 53 features | vs v3, 1,000 games | **0.515** | Tie with v3, and **+7 pp over win labels on identical data** |
| F-vs-F data (10,000 games), VP-margin labels, 50 features | vs v3 / vs F (holdout), 1,000 games each | 0.485 / 0.663 | Tie with v3 |
| v3 self-play data, VP-margin labels, 53 features | vs F (holdout), 1,000 games | 0.668 | Tie with v3 |

### Diagnostic that shaped these choices

How often does our move differ from F's? Over 5,721 decisions in 120 games, our final move differs from F's in **47%** of decisions:

| F's choice | We differ |
|---|---|
| road | 70% |
| bank trade | 64% |
| discard | 55% |
| end turn | 46% |
| opening placement | 32% |
| settlement | 22% |
| robber move | 19% |
| city | 12% |

So the learned term clearly steers play. The plateau comes from how well it ranks sibling moves, not from too small a scale.

## Testing rules used

- Every comparison is head-to-head against the previous version on the same seeds, or against F on the same boards.
- Anything under about **4 pp** is treated as noise. That's ±3 pp binomial at 1,000 games plus up to 2.5 pp of run-to-run spread for the same agent, which comes from wall-clock search cutoffs and laptop throttling.
- Each upload changes one thing. v5 differs from v3 only by `cheap_in_reserve`.
