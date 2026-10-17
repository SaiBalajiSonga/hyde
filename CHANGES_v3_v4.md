# Changes in v3 and v4

All versions share one code base: `agent.py`, `value.py`, `feats.py` and `learned.py`. A version is fixed by the `config.json` shipped in its zip, plus the model files that config names.

## Baseline: v2 (`tfi_v2_open6.zip`)

- F's evaluation, re-implemented exactly and 3× faster (`value.py`)
- \+ learned win-probability **linear** model (`model_lin.json`, 50 features, trained on F-vs-F games), added as α·logit with α = 1e6
- \+ same-turn greedy lookahead on the top 4 moves (`top_k` 4, `rollout_depth` 4)
- \+ opening simulation over the top 6 settlement candidates (`open_m` 6)

Config: `{"top_k":4, "model":"model_lin.json", "model_alpha":1e6, "open_m":6}`

| | vs F | vs O | Mean | CPU mean / worst | Over clock |
|---|---|---|---|---|---|
| Local (1,000 games, dev / holdout) | 0.684 / 0.672 | not available | | 455 / 1141 ms | 0 |
| **Portal, public boards** | **0.649** | **0.487** | **0.568** | 368 / 1029 ms | 0 |

---

## v3 (`tfi_v3_mlp.zip`): one change from v2

**Change:** the linear win model is replaced by a small neural network (MLP, one hidden layer of 32 ReLU units), `model_mlpff32.json`.

- Training data: 16,000 F-vs-F games, about 1.04M positions recorded after every action from both seats, labelled with the eventual winner (`tools/data/ff4000.npz` + `ff12000.npz`).
- The 50 features are the same as v2's (`feats.py`).
- Prediction quality on held-out games: log-loss **0.472** against 0.491 for the linear model, accuracy 76.1% against 75.0%. The biggest gain is early in the game (turns 0 to 20: 65.3% against 62.4%).
- Everything else (α, lookahead, opening) is identical to v2.

Config: `{"top_k":4, "model":"model_mlpff32.json", "model_alpha":1e6, "open_m":6}`

| | vs F | vs O | Mean | CPU mean / worst | Over clock |
|---|---|---|---|---|---|
| Local (1,000 games, dev / holdout) | 0.678 / 0.667 | not available | | 510 / 1406 ms | 0 |
| **Portal, public boards** | **0.698** | **0.505** | **0.602** | 450 / **1501** ms | **1** |

**Reading:**
- Locally, v3 equals v2. On the public boards it is +4.9 pp against F, at the edge of the intervals, and +1.8 pp against O (noise).
- The MLP costs about 20% more CPU per game. That caused one game over the clock on the scoring machine.

---

## v4 (`tfi_v4_bandit.zip`): two changes from v3

### Change 1: in-run strategy selection (bandit)

The agent now holds **three complete strategies ("arms")** and picks one per game:

| Arm | Win model | Otherwise |
|---|---|---|
| 0 | `model_mlpff32.json` (v3's MLP) | v3 settings |
| 1 | `model_lin.json` (v2's linear model) | v3 settings |
| 2 | `model_mlpff64.json` (MLP, 64 units, same F-vs-F data) | v3 settings |

- **Why:** one agent process plays a whole match against one opponent (problem statement, Section 4), and adapting to the opponent is allowed. Arms that are equally strong against F may differ against O or other teams. The group stage scores our *worst half* of opponents.
- **How:** before each game it chooses an arm by **Thompson sampling** from a Beta(wins+1, losses+1) posterior per arm.
- **Outcome labels:** a game counts as won if our chosen move produced the 10th VP. The agent checks this on a sampled copy of the game when we have 7 or more VP. A new game is detected when the turn counter goes backwards.
- **Cost:** nothing measurable against F, where the arms are equal. 1,000 games in 5 processes scored 0.671.
- Self-labelling counts 61% wins against the true 67%, because it misses wins completed by a forced move. The bias is the same for every arm, so the arms are still ranked correctly.
- `TFI_BANDIT_LOG` (an environment variable, for local runs only) writes the per-arm win/loss counts to a file. It does nothing on the scoring machine.

### Change 2: CPU safety for long games (`cheap_in_reserve`)

- **Why:** v3's one over-clock game. Past 96 decisions the clock stops growing (ceiling 1.5 s), but each decision still costs about 1 ms just to receive the game state. Our one-ply pass with the MLP hybrid costs about 100 µs per legal move.
- **How:** once the clock drops into the 200 ms reserve, the agent plays one-ply with **F's cheap evaluation** instead of the hybrid. Lower tiers are unchanged: below 40 ms, cheap one-ply; below 15 ms, a fixed move-type priority with no copying.

### Smaller changes in the same build (no effect on play)

- **Warm-up in `__init__`:** each strategy evaluates a throwaway position at load time, so import and first-call costs are not charged to the first game. Load time is 0.22 s against a 60 s limit.
- `feats.py` gained 3 appended features (production lost to the robber, both sides; longest-road margin). They are **computed only when a model uses more than 50 features**, so no model in v4 pays for them.
- `tools/package.py` now bundles the model files of every arm.

Config:
```json
{"top_k":4, "model":"model_mlpff32.json", "model_alpha":1e6, "open_m":6,
 "cheap_in_reserve":true,
 "arms":[{}, {"model":"model_lin.json"}, {"model":"model_mlpff64.json"}]}
```

| | vs F | vs v3 | CPU mean / worst | Over clock | Errors |
|---|---|---|---|---|---|
| Local, holdout boards (1,000 games, 6 long-running processes) | 0.640 ± .030 | 0.509 ± .031 | 437 / 1297 ms | 0 | 0 |
| Official `catan_kit.evaluate`, 20-game smoke test | 0.65 | | 442 / 719 ms | 0 | 0 (954 decisions) |
| **Portal, public boards** | *pending* | | | | |

**Reading:** v4 is as strong as v3 (head-to-head 50.9%). The F number is within the noise of v3's 0.667 on the same holdout boards, because repeat runs of the same agent vary by about ±2.5 pp on top of the binomial ±3 pp. v4's expected gains are against O and in the group stage (adaptation), and fewer over-clock games.

---

## Tried between v3 and v4 and rejected (details in `EXPERIMENTS.md`)

| Attempt | Result vs F (1,000 games) | Why rejected |
|---|---|---|
| CPU cut: `top_k` 2, 12 ms cap, 250 ms reserve | 0.652 (v3: 0.678) | −2.6 pp for only −8% CPU |
| MLP retrained on F-vs-F + v3-vs-F + v3-vs-v2, player identity controlled, +3 features | 0.640 / 0.640 | Not better than v3 |
| MLP with 64 hidden units, F-vs-F data, as the only model | 0.655 / 0.645 | Not better alone; kept as a bandit arm |
| Opening judged by a dedicated opening model (linear / MLP) | 0.693 / 0.677 dev, 0.644 / 0.680 holdout | No gain on holdout |
| Exact robber steal expectation | 0.679 | Neutral |
| Hand-aware discard | 0.663 | Neutral or slightly negative |
| Army race, zero-sum VP, road race, blocking terms | 0.64 to 0.69 | Neutral or harmful |

## Upload status

| Zip | Scored upload | Status |
|---|---|---|
| `tfi_v2_open6.zip` | #1 | done |
| `tfi_v3_mlp.zip` | #2 | done |
| `tfi_v4_bandit.zip` | (#3) | built and tested; **check the portal to confirm it was uploaded** |
