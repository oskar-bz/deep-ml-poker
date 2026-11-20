"""
Coordinate-search tuner for ApexBot's P_* parameters.

Each candidate is scored on the same table seeds as the incumbent (paired
comparison), across a weighted mix of fields. A move is accepted only if the paired
improvement exceeds Z standard errors; accepted moves are re-checked on fresh seeds.

    python tools/tune.py --tables 1200 --passes 2
Results are appended to tools/tune_log.txt.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import run_points  # noqa: E402

WEIGHTS = {"realistic": 0.30, "llm": 0.25, "sdkmix": 0.20, "mirror": 0.15, "example": 0.10}

# name: (default, step, lo, hi)
SPACE = {
    "P_OPEN_BASE": (0.44, 0.06, 0.20, 0.70),
    "P_OPEN_PER_BEHIND": (0.065, 0.02, 0.0, 0.12),
    "P_OPEN_SIZE_BB": (2.62, 0.5, 2.0, 4.0),
    "P_PUSH_ZONE_BB": (12.0, 3.0, 6.0, 20.0),
    "P_PUSH_MULT": (1.0, 0.2, 0.5, 1.8),
    "P_PUSH_MIN_PCT": (0.46, 0.08, 0.2, 0.7),
    "P_CALL_SAFETY_HU": (0.035, 0.03, -0.05, 0.15),
    "P_CALL_SAFETY_MW": (0.065, 0.03, -0.05, 0.15),
    "P_CALL_MIN_ALLIN_PCT": (0.66, 0.08, 0.4, 0.9),
    "P_3BET_MIN": (0.86, 0.04, 0.75, 0.95),
    "P_CALL_MARGIN_ADD": (0.0, 0.03, -0.08, 0.10),
    "P_VALUE_MULT": (1.0, 0.25, 0.5, 1.8),
    "P_VALUE_TPTK_EQ": (0.62, 0.06, 0.45, 0.8),
    "P_VALUE_EQ": (0.68, 0.06, 0.5, 0.85),
    "P_CBET_FOLD_GATE": (0.58, 0.08, 0.3, 0.8),
}

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_log.txt")


def spec(params):
    diff = {k: v for k, v in params.items() if abs(v - SPACE[k][0]) > 1e-12}
    if not diff:
        return "apex_sdk"
    return "apex_sdk:" + ",".join(f"{k}={v:g}" for k, v in sorted(diff.items()))


def score(params, tables, seed, pool):
    """Returns {field: per-table points list}."""
    return {f: run_points(spec(params), f, tables, seed, pool=pool) for f in WEIGHTS}


def paired(a, b):
    """Weighted mean of (b - a) and its standard error."""
    mean = 0.0
    var = 0.0
    for f, w in WEIGHTS.items():
        d = [y - x for x, y in zip(a[f], b[f])]
        n = len(d)
        m = sum(d) / n
        v = sum((x - m) ** 2 for x in d) / max(1, n - 1) / n
        mean += w * m
        var += w * w * v
    return mean, math.sqrt(var)


def wmean(r):
    return sum(w * sum(r[f]) / len(r[f]) for f, w in WEIGHTS.items())


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=1200)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    params = {k: v[0] for k, v in SPACE.items()}
    seed = 11
    pool = ProcessPoolExecutor(a.workers)
    log(f"=== tune start {time.ctime()} tables={a.tables}/field")
    base = score(params, a.tables, seed, pool)
    log(f"baseline {wmean(base):.3f}")
    for p in range(a.passes):
        improved = False
        for k, (_d, step, lo, hi) in SPACE.items():
            for direction in (+1, -1):
                cand = dict(params)
                cand[k] = min(hi, max(lo, params[k] + direction * step))
                if cand[k] == params[k]:
                    continue
                t0 = time.time()
                res = score(cand, a.tables, seed, pool)
                d, se = paired(base, res)
                log(f"  {k}={cand[k]:g}: {wmean(res):.3f} diff {d:+.3f} ± {se:.3f} ({time.time() - t0:.0f}s)")
                if d > a.z * se:
                    # confirm on fresh seeds
                    seed2 = seed + 1000 + p * 50 + len(k)
                    b2 = score(params, a.tables, seed2, pool)
                    r2 = score(cand, a.tables, seed2, pool)
                    d2, se2 = paired(b2, r2)
                    log(f"    confirm: diff {d2:+.3f} ± {se2:.3f}")
                    if d2 > 0.5 * se2:
                        params = cand
                        seed += 1  # new seeds for the new incumbent
                        base = score(params, a.tables, seed, pool)
                        improved = True
                        log(f"  ACCEPT {k}={cand[k]:g} -> {spec(params)}  base {wmean(base):.3f}")
                        break
        log(f"pass {p + 1} done: {spec(params)}")
        if not improved:
            break
    log(f"=== final {spec(params)}")
    pool.shutdown()


if __name__ == "__main__":
    main()
