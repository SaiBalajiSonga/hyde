"""Tune the value-function weights on win rate against an opponent (cross-entropy method).

Each weight is the default times exp(x). Every generation samples `pop` candidates around the mean,
plays each against the opponent on the same training boards (both seats), and moves the mean towards
the best `elite`. Training boards start at 100000 and change every generation; evaluation boards
(990000+) are never used here.

    tools/tune.sh --gens 15 --pop 10 --games 200 --out tools/tune_run1
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TFI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.environ.get("CATAN_KIT", r"C:\Users\SAI BALAJI\OneDrive\Desktop\catan-hackathon-agent")

# The weights we tune, and the base each multiplier applies to.
TUNED = {
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1.0,
    "reachable_production_0": 1e2,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10.0,
    "hand_synergy": 1e2,
    "hand_resources": 1.0,
    "discard_penalty": -5.0,
    "hand_devs": 10.0,
    "army_size": 10.1,
    # extras from value.py (start near zero: x0 = -6)
    "port_2to1": 1e8,
    "port_3to1": 1e7,
    "enemy_vps": -3e13,
    "city_ready": 1e8,
    "hidden_vp": 3e14,
}
X0 = {"port_2to1": -6, "port_3to1": -6, "enemy_vps": -6, "city_ready": -6, "hidden_vp": -6}
NAMES = list(TUNED)


def weights_of(x):
    return {k: TUNED[k] * math.exp(float(v)) for k, v in zip(NAMES, x)}


def _chunk(job):
    cfg_path, path_b, seeds, idx = job
    os.environ["TFI_CFG"] = cfg_path
    from catan_kit import match
    recs, _ = match.play_chunk(os.path.join(TFI, "agent.py"), path_b, seeds, idx)
    return cfg_path, sum(r["winner"] == "a" for r in recs), len(recs), \
        sum(r["a"]["cpu_s"] for r in recs), sum(r["a"]["over"] is not None for r in recs)


def main():
    from catan_kit._patch import ensure_hash_seed
    ensure_hash_seed()
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--pop", type=int, default=10)
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sigma", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--chunk", type=int, default=20, help="games per job")
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--b", default=os.path.join(KIT, "examples", "f_agent.py"))
    p.add_argument("--init", default=None, help="json with a starting mean x")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", default=None, help="json merged under every candidate's config")
    a = p.parse_args()
    base_cfg = {}
    if a.base:
        with open(a.base) as f:
            base_cfg = json.load(f)

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    d = len(NAMES)
    mean = np.array([X0.get(k, 0.0) for k in NAMES])
    if a.init:
        with open(a.init) as f:
            mean = np.array(json.load(f)["x"], dtype=float)
    sigma = np.full(d, a.sigma)
    log = open(os.path.join(a.out, "log.jsonl"), "a")

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for gen in range(a.gens):
            t0 = time.time()
            xs = [mean.copy()] + [mean + sigma * rng.standard_normal(d) for _ in range(a.pop - 1)]
            seed0 = 100000 + gen * 1000 + a.seed * 100000
            jobs, paths = [], []
            for c, x in enumerate(xs):
                path = os.path.abspath(os.path.join(a.out, f"g{gen}_c{c}.json"))
                cfg = dict(base_cfg)
                cfg.update({"top_k": a.top_k, "weights": weights_of(x)})
                with open(path, "w") as f:
                    json.dump(cfg, f)
                paths.append(path)
                for s in range(0, a.games, a.chunk):
                    idx = list(range(s, min(s + a.chunk, a.games)))
                    jobs.append((path, a.b, [seed0 + i // 2 for i in idx], idx))
            wins = {q: 0 for q in paths}
            n = {q: 0 for q in paths}
            cpu = {q: 0.0 for q in paths}
            over = {q: 0 for q in paths}
            for q, w, k, c, o in ex.map(_chunk, jobs):
                wins[q] += w
                n[q] += k
                cpu[q] += c
                over[q] += o
            rates = np.array([wins[q] / n[q] for q in paths])
            order = np.argsort(-rates)
            elite = np.array([xs[i] for i in order[:a.elite]])
            new_mean = elite.mean(axis=0)
            sigma = np.maximum(0.6 * sigma + 0.4 * elite.std(axis=0) + 0.05, 0.1)
            rec = dict(gen=gen, seed0=seed0, rates=rates.round(4).tolist(),
                       mean_rate=float(rates[0]), best=float(rates[order[0]]),
                       best_x=xs[order[0]].round(3).tolist(), mean_x=mean.round(3).tolist(),
                       new_mean_x=new_mean.round(3).tolist(), sigma=sigma.round(3).tolist(),
                       cpu_ms=[round(cpu[q] / n[q] * 1000) for q in paths],
                       over=[over[q] for q in paths], wall=round(time.time() - t0))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"gen {gen:2d}  mean-candidate {rates[0]:.3f}  best {rates[order[0]]:.3f}  "
                  f"pop-avg {rates.mean():.3f}  wall {rec['wall']}s", flush=True)
            mean = new_mean
            with open(os.path.join(a.out, "mean.json"), "w") as f:
                json.dump({"x": mean.tolist(), "names": NAMES,
                           "weights": weights_of(mean)}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
