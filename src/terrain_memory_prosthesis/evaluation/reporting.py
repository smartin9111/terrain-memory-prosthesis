"""Statistics and reporting helpers shared by all pilot / evaluation scripts."""
from __future__ import annotations

import csv
import json
import math
import os


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion k/n (n small is the normal case here)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (min(p, max(0.0, (c - h) / d)), max(p, min(1.0, (c + h) / d)))  # clamp: rounding must not leave [0, 1] or exclude p


def summarize(episodes: list[dict]) -> dict:
    n = len(episodes)
    k = sum(1 for e in episodes if e["fell"])
    lo, hi = wilson_ci(k, n)
    mean = lambda key: (sum(e[key] for e in episodes) / n) if n else float("nan")
    mean_opt = lambda key: (sum(e[key] for e in episodes if key in e) / max(1, sum(1 for e in episodes if key in e))) if n else float("nan")
    return {
        "n": n,
        "falls": k,
        "fall_rate": k / n if n else float("nan"),
        "fall_ci_lo": lo,
        "fall_ci_hi": hi,
        "mean_steps": mean("steps"),
        "mean_distance_m": mean("distance"),
        "mean_speed_mps": mean("speed"),
        "speed_err_mps": mean("speed_err"),
        "mean_abs_prosthesis_cmd": mean("prosthesis_cmd"),
        "mean_return": mean_opt("reward"),            # the environment's own reward summed over the episode (absent in old rows)
        "mean_reward_per_step": mean_opt("reward_per_step"),
    }


def save_rows(out: str, name: str, rows: list[dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, f"{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)


def print_table(rows: list[dict], cols: list[str], fmt: dict | None = None):
    fmt = fmt or {}
    widths = {c: max([len(c)] + [len(_f(r.get(c), fmt.get(c))) for r in rows]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(_f(r.get(c), fmt.get(c)).ljust(widths[c]) for c in cols))


def _f(v, f):
    if v is None:
        return "-"
    if f and isinstance(v, (int, float)):
        return format(v, f)
    return str(v)


def verdict(out: str, name: str, status: str, reasons: list[str], extra: dict | None = None) -> int:
    """Print + save a verdict. Returns the process exit code: PASS=0, WARN=1, FAIL=2."""
    code = {"PASS": 0, "WARN": 1, "FAIL": 2}[status]
    print("\n" + "=" * 70)
    print(f"{name}: {status}")
    for r in reasons:
        print("  - " + r)
    print("=" * 70)
    with open(os.path.join(out, f"{name}_verdict.json"), "w") as f:
        json.dump({"status": status, "reasons": reasons, **(extra or {})}, f, indent=2)
    return code
