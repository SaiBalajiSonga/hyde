"""Cross-check tournament: each of our versions against every bot in the pool and each other.

Every match uses the same boards (seed0, both seats per board). Games are split into chunks of
--chunk games played in ONE agent process each, so v4's in-run bandit has room to learn.
Results go to tools/tournament.csv (one row per match) and a summary with worst-half scores.

    tools/tournament.sh --games 500 --chunk 50 --workers 16
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.environ.get("CATAN_KIT", r"C:\Users\SAI BALAJI\OneDrive\Desktop\catan-hackathon-agent")
POOL = os.path.join(TFI, "tools", "pool")

OURS = {"v2": "v2", "v3": "v3", "v4": "v4"}
OPPONENTS = {
    "F": os.path.join(KIT, "examples", "f_agent.py"),
    "random": os.path.join(KIT, "examples", "random_agent.py"),
    "F_build": "F_build", "F_dev": "F_dev", "F_road": "F_road", "F_city": "F_city",
    "F_contender": "F_contender", "F_look": "F_look", "learned_only": "lin_k4",
    "mlp_mixed": "mlp2_k4", "v1": "hyb_1e6_k4",
}


def path_of(x):
    return x if x.endswith(".py") else os.path.join(POOL, x, "agent.py")


def _chunk(job):
    a, b, pa, pb, seeds, idx = job
    os.environ.pop("TFI_CFG", None)
    from catan_kit import match
    recs, _ = match.play_chunk(pa, pb, seeds, idx)
    return a, b, recs


def main():
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=600000)
    ap.add_argument("--only", default="", help="comma list of our versions to run (default all)")
    a = ap.parse_args()

    ours = {k: v for k, v in OURS.items() if not a.only or k in a.only.split(",")}
    matches = []
    for o in ours:
        for opp in OPPONENTS:
            matches.append((o, opp, path_of(OURS[o]), path_of(OPPONENTS[opp])))
    names = list(ours)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            matches.append((names[i], names[j], path_of(OURS[names[i]]), path_of(OURS[names[j]])))

    jobs = []
    for m in matches:
        for s in range(0, a.games, a.chunk):
            idx = list(range(s, min(s + a.chunk, a.games)))
            jobs.append((m[0], m[1], m[2], m[3], [a.seed0 + i // 2 for i in idx], idx))
    print(f"{len(matches)} matches, {len(jobs)} jobs of {a.chunk} games, {a.workers} workers",
          flush=True)

    t0 = time.time()
    recs = defaultdict(list)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(_chunk, j) for j in jobs]
        for f in as_completed(futs):
            x, y, r = f.result()
            recs[(x, y)] += r
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} jobs  {time.time() - t0:.0f}s", flush=True)

    out = os.path.join(TFI, "tools", "tournament.csv")
    new = not os.path.exists(out)
    rows = []
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "a", "b", "games", "seed0", "a_win_rate", "ci95", "a_mean_cpu_ms",
                        "a_worst_cpu_ms", "a_over_budget", "a_errors", "b_over_budget", "no_winner"])
        for (x, y), r in sorted(recs.items()):
            n = len(r)
            wins = sum(g["winner"] == "a" for g in r)
            p = wins / n
            ci = 1.96 * (p * (1 - p) / n) ** 0.5
            cpu = [g["a"]["cpu_s"] for g in r]
            row = [time.strftime("%H:%M:%S"), x, y, n, a.seed0, round(p, 4), round(ci, 4),
                   round(1000 * sum(cpu) / n, 1), round(1000 * max(cpu), 1),
                   sum(g["a"]["over"] is not None for g in r), sum(g["a"]["errors"] for g in r),
                   sum(g["b"]["over"] is not None for g in r), sum(g["winner"] is None for g in r)]
            w.writerow(row)
            rows.append(row)

    print(f"\n{'A':4s} {'B':13s} {'win':>6s} {'ci':>6s} {'cpu':>6s} {'worst':>6s} over err")
    for row in rows:
        print(f"{row[1]:4s} {row[2]:13s} {row[5]:6.3f} {row[6]:6.3f} {row[7]:6.0f} {row[8]:6.0f} "
              f"{row[9]:4d} {row[10]:3d}")
    print("\nworst-half mean over the F-like field (F, F variants, F_look, v1, learned_only, mlp_mixed):")
    field = [k for k in OPPONENTS if k != "random"]
    for o in ours:
        rates = sorted(r[5] for r in rows if r[1] == o and r[2] in field)
        half = rates[: (len(rates) + 1) // 2]
        print(f"  {o}: worst-half {sum(half) / len(half):.3f}  overall {sum(rates) / len(rates):.3f}  "
              f"min {rates[0]:.3f}")
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
