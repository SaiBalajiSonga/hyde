# Catan Agent: Overview, Goals and Implementation Plan

Hackathon day: **17 Oct 2026**. Submissions 11:00 to 17:00 IST. Report due 17:30. Team: 3 to 4 people.

---

## 1. Overview: what we are actually solving

We submit one Python `Agent` whose `decide(obs, playable_actions) -> int` plays **2-player Catan, 10 VP,
no player trading**, on the organisers' patched catanatron engine.

The constraints that shape the whole design:

| Fact | What it means for us |
|---|---|
| **CPU clock**: 300 ms to start, +12.5 ms per decision, ceiling 1.5 s per game. About 1 s per game on average. Out of clock means random moves for the rest of the game, which is a certain loss. | The budget is about **4x F per decision**. Deep search (MCTS, alpha-beta depth 3) is out. Our edge must come from a **better evaluation**, **better special-case decisions**, and **selective, budgeted lookahead**. |
| Their machine is about **2x slower** than a laptop. | Locally, target **at most 500 ms mean and 700 ms worst CPU per game**. |
| **47% of decisions are forced** and cost nothing. | The time that counts goes to the other ~56 decisions per game. |
| **Hidden information is tiny**: the opponent's resource hand is exact. Only their unplayed dev cards are hidden. | One `sample_world` per decision is nearly enough. Imperfect-information machinery is not where the wins are. |
| **Opening placement** has the most legal moves, and the 300 ms starting bank exists for it. | The two settlement placements decide most of a 2-player game. Spend time here. |
| Faults above 2% means disqualification. | Every code path is wrapped, and there is always a fallback. |
| Scoring: strength = 0.5 x WR(F) + 0.5 x WR(O) on **private** boards. Robustness = mean WR against the **worst half** of your group. Report + presentation weigh as much as both together. | Do not overfit to F. Test against a **pool** of different opponents. Keep a clean experiment log, because it becomes the report. |

**F** = catanatron `ValueFunctionPlayer`: a 1-ply greedy over a 13-feature linear value function
(`base_fn`, `DEFAULT_WEIGHTS`), 2.9 ms per decision. **O** = the organisers' trained bot, stronger than F.
We cannot run O locally, so the only signal on O is the leaderboard.

---

## 2. Goals

| Priority | Goal | Measured by |
|---|---|---|
| **G0, must** | A legal, fault-free, in-clock agent **submitted by 11:30** | First upload passes the two test games. 0 faults. 0 games over the clock. |
| **G1, must** | Beat F clearly: **WR(F) of 0.60 or more** locally over 1,000+ paired games | `catan_kit.evaluate`, with the 95% CI lower bound above 0.55 |
| **G2, should** | WR(F) of 0.65 or more, and **no opponent in our test pool below 0.50** (robustness) | Pool tournament (section 6) |
| **G3, should** | Beat O | Leaderboard O column |
| **G4, must** | A 4-page report where every number comes from a logged run, including failed experiments and ablations | `results.csv` plus the report |
| **Non-goal** | Big neural nets, RL from scratch, MCTS | Too slow for the clock and too risky for one day. Mention them in the report as considered and rejected, with the reasons. |

---

## 3. Agent architecture (target for the final submission)

```
my_team/
  agent.py          # Agent class: dispatch, time budget, safety wrapper
  features.py       # fast features of a Game state (F's 13 + our extras)
  value.py          # linear value function, weights loaded from weights.json
  opening.py        # initial-placement policy (settlements + roads)
  specials.py       # robber, discard, monopoly, year of plenty, knight timing
  search.py         # candidate pruning + budgeted same-turn lookahead
  weights.json      # tuned weights (versioned: weights_v1.json, ...)
```

`decide()` flow:

```
1. t0 = time.process_time()
   budget = choose_budget(obs.cpu_left, phase)      # section 3.2
2. try:
     phase dispatch:
       initial placement -> opening.py
       MOVE_ROBBER / DISCARD / MONOPOLY / YoP -> specials.py (exact: opp hand is known)
       otherwise -> search.py:
         a. world = obs.sample_world(rng)
         b. candidates = prune(actions)             # dedupe/drop dominated maritime trades
         c. 1-ply: score every candidate with value.py (this is F with our weights)
         d. while time < budget: refine top-k by a greedy same-turn rollout to END_TURN
         e. return argmax
   except Exception:
     return f_fallback(obs, actions)   # plain F; if that also fails -> 0
```

### 3.1 Why same-turn lookahead, not opponent search
F's myopia is mostly **inside our own turn**. A maritime trade, or a dev-card buy, scores badly by
itself, even when it enables a build on the next action. A greedy rollout from each top-k candidate
until `END_TURN` makes the evaluation see the whole turn. It is nearly deterministic, so it needs no
dice sampling, and it costs about k x (turn length) evaluations.
`SameTurnAlphaBetaPlayer` got 0.08 only because it ran out of clock in 415 of 1,000 games, not
because the idea is wrong. Ours is budget-capped.

### 3.2 Time management
* Spend inside the agent at most **`min(0.5 * obs.cpu_left, cap)`**. The cap is about 120 ms in the
  opening and about 8 ms otherwise, measured with `time.process_time()`.
* The opening (2 settlement decisions) may use up to about 200 ms locally, out of the 300 ms bank.
* If `obs.cpu_left` drops below 150 ms, switch to **pure 1-ply** (F cost). Below 40 ms, use a cheap
  heuristic with no `copy()`.
* Do all imports, weight loads and a warm-up evaluation in `__init__`. It is free, up to 60 s.
* No threads and no multiprocessing. If torch is imported, call `torch.set_num_threads(1)`.

### 3.3 Safety wrapper (built first, never removed)
* Every `decide` is in `try/except` with an F fallback, and then `return 0`.
* Assert that the index is in range before returning.
* Seed one `random.Random` in `__init__`.
* A local test plays 200 games and asserts **0 faults and 0 over the clock**. Run it before every upload.

---

## 4. The four improvement levers (ranked by expected gain per hour)

### L1. Opening placement: high leverage, cheap
* Score each candidate node by: pip sum (dice probability) of adjacent tiles, resource diversity,
  scarcity (pips per resource on this board), complementarity with the first settlement (in 2P we
  place 2), port synergy (a 2:1 port on a resource we produce heavily), and blocking the opponent's
  best remaining spot.
* For the 1st settlement, use a 2-ply check: for the top 8 nodes, assume the opponent takes its best
  reply node by our score, then pick our best 2nd node. Maximise our total minus theirs.
* Roads: point toward the best reachable free node (distance-2 rule).
* Tune the handful of opening coefficients against F (section 5).

### L2. Value-function weights: no extra runtime cost
* **Step 1, regression (Buro 2002).** Log afterstate features at every decision of about 5k games
  (F vs F, plus F vs perturbed-F for diversity), labelled with the final winner. Fit
  `LogisticRegression` and use the coefficients as initial weights.
* **Step 2, CMA-ES (Hansen 2001)** on win rate vs F. Population about 12, 200 paired games per
  candidate, 15 to 25 generations. Run it on Kaggle/Colab/spare laptops in parallel.
* Add features to the 13 only if they are cheap: VP gap, turns-to-next-build, longest-road and
  largest-army **race margins**, robber-on-us, dev-card count, resource diversity of production,
  port-adjusted trade efficiency, hand size above 7 risk.

### L3. Special decisions: exact information makes them easy wins
* **Robber**: we know the opponent's exact hand. Choose the tile that maximises
  (opponent pips blocked) + (expected value of the stolen card for *our* next build). Never block
  ourselves.
* **Discard**: keep the subset that maximises builds reachable next turn (enumerate, it is small).
* **Monopoly**: argmax over resources of the opponent's count, weighted by our need.
* **Year of plenty**: take what completes the best build now.
* **Knight timing**: play before rolling if the robber is on our tile or if it wins/defends largest army.

### L4. Budgeted same-turn lookahead (section 3.1)
* Start with k = 3. Increase k only while CPU stays under target.
* **Candidate pruning** first: collapse equivalent maritime trades and drop trades that do not
  enable a build within the turn. This alone frees time.

**Stretch only, if L1 to L4 are done:** opponent adaptation inside a process (track the opponent's
style and adjust robber/blocking), or a small learned eval distilled into a linear or tiny MLP model.

---

## 5. Experiment protocol (this becomes the report's method section)

* **Seeds**: fix a **dev seed set** for tuning and a separate **holdout seed set** used only to
  confirm before an upload (problem statement, advice 6). Check `python -m catan_kit.evaluate --help`
  for the seed option.
* **Statistics**: ±3.1 pp is the 95% CI at 1,000 games, and ±4.9 pp at 400. Tune at 400. Confirm at
  1,000 to 2,000. **A change counts only if it beats the previous version beyond the CI.**
* **One change at a time**, always against the last measured version, on the same seeds.
* **Log every run** to `results.csv`:
  `time, version, change, opponent, games, seeds, WR, CI_lo, CI_hi, mean_cpu_ms, worst_cpu_ms, over_clock, faults, notes`.
* **Loss forensics**: replay 10 lost games per version (runs replay exactly). Classify the cause:
  bad opening, starved resource, lost road race, robbed, idle hand. Fix the biggest class next.
* Use `catan_kit._patch.ensure_hash_seed()` plus an `if __name__ == "__main__":` guard in any custom
  driver script.

---

## 6. Robustness pool (protects the 9.1 score)

Keep every version, and build cheap opponent variants. Each new candidate plays all of them:

| Opponent | Why |
|---|---|
| F | The anchor |
| F with build-heavy, dev-heavy and road-heavy weights | Other teams will be F variants. Stresses different strategies. |
| Our v1, v2, ... | Catches regressions and self-exploits |
| Opening-only agent (L1 plus F) | Isolates the opening's contribution, as an ablation for the report |

Selection rule: **maximise mean WR over the worst half of the pool**, which mirrors 9.1, with WR(F) as the tiebreak.

---

## 7. Timeline and owners (IST)

Roles: **A** infra and eval, **B** search and time, **C** evaluation and tuning, **D** domain
heuristics and report. With 3 people, D's work splits between B and C, and A owns the report.

| Time | A: infra/eval | B: search/time | C: eval/tuning | D: heuristics/report |
|---|---|---|---|---|
| **10:00 to 10:45** | Install, `selfcheck`, run F vs F for 100 games, check throughput | Read `obs`, action lists, phases. Micro-benchmark `copy`, `execute`, value fn, `sample_world`. | Read `base_fn` and the 13 features. Inspect `create_sample_vector`. | Read the rules. Check how F plays the opening, robber and discards. |
| **10:45 to 11:30** | Parallel eval runner, `results.csv`, upload script. **Upload S1 = F + safety wrapper.** | Safety wrapper and time budget skeleton | Feature logger for the regression dataset (start generating games) | Opening scorer v0 |
| **11:30 to 13:00** | Opponent pool, holdout seeds, CPU dashboard | Candidate pruning, then 1-ply with pluggable weights | Logistic regression weights, then start CMA-ES on Kaggle/Colab | Opening v1 plus robber, discard, monopoly, YoP |
| **13:00 to 13:15** | **Integration and gate**. Merge whatever passes. **Upload S2.** | | | Start the report skeleton |
| **13:15 to 15:00** | Pool tournaments, loss forensics | Same-turn lookahead (L4), tune k against CPU | CMA-ES continues. Add extra features. | Tune opening coefficients. Knight timing. |
| **15:00** | **Upload S3** (best combined) before the 15:30 checkpoint | | | |
| **15:00 to 16:15** | Holdout confirmation at 2,000 games | Squeeze CPU, tighten budget | Final weights from CMA-ES | Ablation runs for the report |
| **~16:00** | **Upload S4/S5** (strongest, and most robust if they differ) | | | |
| **16:30** | **Last upload, S6.** Keep S7 and spare uploads for an emergency fix. **Code freeze.** | | | |
| **16:30 to 17:30** | Final tables | Help with the report | Help with the report | **Report to PDF, uploaded by 17:25** |
| **17:30 to 18:00** | Slides plus a Q&A drill | | | |

**Why upload by 16:30**: only 2 uploads can wait for scoring at once, and each scores 2,000 games.
A 16:55 upload might never be scored in a useful order.

### Submission gate (all must hold before any upload)
1. 1,000 or more holdout games vs F: WR above the previous best beyond the CI, or same WR and better robustness.
2. Local mean CPU at most 500 ms per game, worst at most 700 ms, **0 over the clock, 0 faults**.
3. Pool worst-half WR is not lower than the previous version's.
4. The zip contains only `.py` and data files, uses relative paths, writes only to `tempfile.gettempdir()`, and is under 100 MB.

### Kill switches
* **13:30**: if lookahead does not beat 1-ply by 3 pp or more at equal CPU, drop it and spend B on opening and pruning.
* **15:00**: if the leaderboard reports games over the clock, halve all caps immediately.
* After 16:30: no new features, only fixes.

---

## 8. Report outline (4 pages, written as we go)

1. **Problem and constraints** (½ page): the clock, why hidden information is small, and why this rules out deep search.
2. **Method** (1¼ pages): architecture, opening model, special decisions, value-function fitting
   (regression then CMA-ES), budgeted same-turn lookahead, time manager.
3. **Experiments** (1½ pages): the version table from `results.csv`, ablations (each lever on and
   off), CPU vs win rate, pool robustness, leaderboard F/O vs local.
4. **What failed and insights** (¾ page): honest negatives (for example lookahead depth, features
   that did not help), lessons beyond Catan (evaluation quality vs search depth under a hard compute
   budget, exact-information sub-decisions).
5. **References**: catanatron, Buro 2002, Hansen and Ostermeier 2001, Long et al. 2010, plus any
   Catan paper we used.

### Q&A prep (the questions carry the most weight)
* Why not MCTS? (Clock numbers: 25 simulations already ran out in every game.)
* Why is one sampled world enough? (Only dev cards are hidden. Show WR at 1 vs 4 worlds if measured.)
* Is your gain significant? (CIs, paired seeds, holdout set.)
* Did you overfit to F? (Pool results, and leaderboard O vs F.)
* What would you do with 10x compute?
