"""Parallel version of catan_kit.evaluate: same seeds, same seat pairing, many worker processes.

    python tools/arena.py --a agent.py --b <kit>/examples/f_agent.py --games 400 --workers 20 --tag v1

Run with the kit on PYTHONPATH. Appends one row per run to tools/results.csv.
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def _chunk(args):
    path_a, path_b, seeds, idx, vps = args
    from catan_kit import match
    records, info = match.play_chunk(path_a, path_b, seeds, idx, vps=vps)
    return records


def main():
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()
    from catan_kit.evaluate import summarise

    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed0", type=int, default=990000)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--vps", type=int, default=10)
    p.add_argument("--tag", default="")
    p.add_argument("--note", default="")
    args = p.parse_args()

    path_a, path_b = os.path.abspath(args.a), os.path.abspath(args.b)
    games = args.games - args.games % 2
    pairs = list(range(games // 2))
    w = max(1, min(args.workers, len(pairs)))
    jobs = []
    for k in range(w):
        mine = pairs[k::w]
        idx = [2 * p + s for p in mine for s in (0, 1)]
        seeds = [args.seed0 + i // 2 for i in idx]
        jobs.append((path_a, path_b, seeds, idx, args.vps))

    t0 = time.time()
    records = []
    with ProcessPoolExecutor(max_workers=w) as ex:
        for recs in ex.map(_chunk, jobs):
            records += recs
    wall = time.time() - t0
    r = summarise(records)
    turns = sum(x["turns"] for x in records) / len(records)
    print(f"A = {args.a}\nB = {args.b}")
    print(f"win rate  {r['win_rate']:.4f} +-{r['ci95']:.4f}  ({r['wins']}/{r['games']})  "
          f"no_winner={r['no_winner']}")
    print(f"cpu/game  mean {r['mean_cpu_s']*1000:.1f} ms  worst {r['worst_cpu_s']*1000:.1f} ms  "
          f"over_budget={r['games_over_budget']}  errors={r['errors']}  "
          f"timeouts={r['timeouts']}  crashes={r['crashes']}")
    print(f"decisions/game {r['decisions']/r['games']:.1f}  turns/game {turns:.1f}  wall {wall:.0f}s")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "tag", "a", "b", "games", "seed0", "win_rate", "ci95",
                         "mean_cpu_ms", "worst_cpu_ms", "over_budget", "errors", "timeouts",
                         "crashes", "no_winner", "note"])
        wr.writerow([time.strftime("%H:%M:%S"), args.tag, args.a, args.b, r["games"], args.seed0,
                     round(r["win_rate"], 4), round(r["ci95"], 4),
                     round(r["mean_cpu_s"] * 1000, 1), round(r["worst_cpu_s"] * 1000, 1),
                     r["games_over_budget"], r["errors"], r["timeouts"], r["crashes"],
                     r["no_winner"], args.note])


if __name__ == "__main__":
    sys.exit(main())
