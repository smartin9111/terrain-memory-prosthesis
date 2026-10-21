#!/usr/bin/env python
"""Seed-level statistics for the noise grid (the TRAINING SEED is the unit; no pooling of episodes / checkpoints).

    python scripts/seed_stats.py 0/C1=a.csv 0/K16=b.csv P/C1=c.csv ... [clean/C1=old.csv ...] --metric steps --amp 0.06

Per run: ONLY its last checkpoint (no checkpoint picking), random terrain of the given amplitude, value = mean over the held-out terrain
seeds (episodes weighted). Then, per family of comparisons:
  A  P2 - P1 for every (level, controller)          B  K16 - C1 for every (level, condition)
  C  K16 - C1 averaged over the NOISY levels (one pre-specified test per condition; strata = levels)
  D  (P2 - P1) at each noisy level minus (P2 - P1) at the clean level  (does the preview benefit change with noise?)
Each line: n per group, difference, 95% bootstrap interval (seeds resampled within cells, 10000x) and a two-sided permutation p
(exact when <= 20000 labelings, else Monte Carlo). A and B are Holm-adjusted within their family. The p-values are descriptive:
with 3-6 seeds per cell only large effects can be detected; "not significant" is NOT "no effect" -- read the interval.
metric: steps (mean steps survived, of 1000; the fall rate saturates at strong noise) | falls (fall rate).
"""
import argparse
import itertools
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from terrain_memory_prosthesis.evaluation.analysis import read_rows  # noqa: E402

RNG = np.random.default_rng(0)


def perm_p(a, b, stat=lambda x, y: np.mean(y) - np.mean(x)):
    """two-sided permutation p for stat(a, b); exact for small groups"""
    pool = np.array(list(a) + list(b), dtype=float)
    n, na = len(pool), len(a)
    obs = abs(stat(np.array(a), np.array(b)))
    from math import comb

    if comb(n, na) <= 20000:
        cnt = tot = 0
        for idx in itertools.combinations(range(n), na):
            m = np.zeros(n, bool); m[list(idx)] = True
            cnt += abs(stat(pool[m], pool[~m])) >= obs - 1e-12
            tot += 1
        return cnt / tot
    cnt = 0
    for _ in range(20000):
        RNG.shuffle(pool)
        cnt += abs(stat(pool[:na], pool[na:])) >= obs - 1e-12
    return (cnt + 1) / 20001


def boot_ci(groups_a, groups_b, B=10000):
    """groups_*: list of arrays (strata); statistic = mean over strata of (mean b - mean a), seeds resampled within each stratum"""
    d = np.empty(B)
    for i in range(B):
        d[i] = np.mean([RNG.choice(gb, len(gb)).mean() - RNG.choice(ga, len(ga)).mean() for ga, gb in zip(groups_a, groups_b)])
    return np.percentile(d, [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+")
    ap.add_argument("--metric", default="steps", choices=["steps", "falls"])
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--noisy", default="W,M,S,P,Q", help="levels counted as noisy in C / D")
    ap.add_argument("--clean", default="0", help="level used as the clean reference in D")
    a = ap.parse_args()

    last = {}
    for it in a.items:
        key, path = it.split("=", 1)
        lv, ctrl = key.split("/")
        for r in read_rows(path):
            if r["terrain_kind"] != "random" or abs(r["amplitude"] - a.amp) > 1e-9:
                continue
            k = (lv, ctrl, r["run"])
            last[k] = max(last.get(k, 0), r["ckpt_steps"])
    acc = defaultdict(lambda: [0.0, 0])
    meta = {}
    for it in a.items:
        key, path = it.split("=", 1)
        lv, ctrl = key.split("/")
        for r in read_rows(path):
            k = (lv, ctrl, r["run"])
            if r["terrain_kind"] != "random" or abs(r["amplitude"] - a.amp) > 1e-9 or r["ckpt_steps"] != last[k]:
                continue
            v = float(r["mean_steps"]) * int(r["n"]) if a.metric == "steps" else float(r["falls"])
            acc[k][0] += v
            acc[k][1] += int(r["n"])
            meta[k] = (lv, ctrl, r["cond"], int(r["seed"]), int(r["ckpt_steps"]))
    cell = defaultdict(list)   # (lv, ctrl, cond) -> [values per run]
    for k, (s, n) in acc.items():
        lv, ctrl, cond, seed, ck = meta[k]
        cell[(lv, ctrl, cond)].append(s / n)
    levels = list(dict.fromkeys(i.split("=")[0].split("/")[0] for i in a.items))
    noisy = [l for l in a.noisy.split(",") if l in levels]
    ctrls = sorted({k[1] for k in cell})
    print(f"metric: {a.metric}, random terrain {a.amp * 100:g} cm, last checkpoint of every run, unit = training seed\n")
    print("cells (mean [min, max], n seeds):")
    for lv in levels:
        for c in ctrls:
            for p in ("P1", "P2"):
                v = cell.get((lv, c, p))
                if v:
                    print(f"  {lv:6}{c:5}{p:3} {np.mean(v):8.1f} [{min(v):7.1f},{max(v):7.1f}] n={len(v)}")

    def report(title, rows):
        print(f"\n{title}")
        ps = holm([r[4] for r in rows]) if len(rows) > 1 and title[0] in "AB" else [None] * len(rows)
        for (name, d, ci, n, p), ph in zip(rows, ps):
            extra = f"  Holm p={ph:.3f}" if ph is not None else ""
            print(f"  {name:26} n={n:7} diff={d:+8.1f}  95% CI [{ci[0]:+8.1f},{ci[1]:+8.1f}]  p={p:.3f}{extra}")

    A = []
    for lv in levels:
        for c in ctrls:
            x, y = cell.get((lv, c, "P1")), cell.get((lv, c, "P2"))
            if x and y:
                A.append((f"{lv} {c}: P2-P1", np.mean(y) - np.mean(x), boot_ci([x], [y]), f"{len(x)}v{len(y)}", perm_p(x, y)))
    report("A  preview effect (P2 - P1)", A)
    B = []
    for lv in levels:
        for p in ("P1", "P2"):
            x, y = cell.get((lv, "C1", p)), cell.get((lv, "K16", p))
            if x and y:
                B.append((f"{lv} {p}: K16-C1", np.mean(y) - np.mean(x), boot_ci([x], [y]), f"{len(x)}v{len(y)}", perm_p(x, y)))
    report("B  memory effect (K16 - C1)", B)
    Crow = []
    for p in ("P1", "P2"):
        xs = [cell.get((l, "C1", p)) for l in noisy]
        ys = [cell.get((l, "K16", p)) for l in noisy]
        ok = [(x, y) for x, y in zip(xs, ys) if x and y]
        if ok:
            gx, gy = [o[0] for o in ok], [o[1] for o in ok]
            obs = np.mean([np.mean(y) - np.mean(x) for x, y in ok])
            cnt = 0
            for _ in range(20000):   # permute the controller label within every level (stratified)
                dd = []
                for x, y in ok:
                    pool = np.array(list(x) + list(y)); RNG.shuffle(pool)
                    dd.append(pool[len(x):].mean() - pool[:len(x)].mean())
                cnt += abs(np.mean(dd)) >= abs(obs) - 1e-12
            Crow.append((f"{p}: K16-C1 over {','.join(noisy)}", obs, boot_ci(gx, gy), "strat.", (cnt + 1) / 20001))
    report("C  memory effect averaged over the noisy levels (pre-specified)", Crow)
    Drow = []
    for c in ctrls:
        ref1, ref2 = cell.get((a.clean, c, "P1")), cell.get((a.clean, c, "P2"))
        if not (ref1 and ref2):
            continue
        for lv in noisy:
            x, y = cell.get((lv, c, "P1")), cell.get((lv, c, "P2"))
            if not (x and y):
                continue
            stat = lambda: (np.mean(y) - np.mean(x)) - (np.mean(ref2) - np.mean(ref1))   # noqa: E731
            d = stat()
            boots = [(RNG.choice(y, len(y)).mean() - RNG.choice(x, len(x)).mean()) - (RNG.choice(ref2, len(ref2)).mean() - RNG.choice(ref1, len(ref1)).mean())
                     for _ in range(10000)]
            labs = [x, y, ref1, ref2]
            cnt = 0
            for _ in range(20000):   # permute P1/P2 labels within each of the two levels
                p1 = np.array(list(x) + list(y)); RNG.shuffle(p1)
                p2 = np.array(list(ref1) + list(ref2)); RNG.shuffle(p2)
                dd = (p1[len(x):].mean() - p1[:len(x)].mean()) - (p2[len(ref1):].mean() - p2[:len(ref1)].mean())
                cnt += abs(dd) >= abs(d) - 1e-12
            Drow.append((f"{c}: ({lv}) - ({a.clean}) of P2-P1", d, np.percentile(boots, [2.5, 97.5]), f"{len(x)}/{len(ref1)}", (cnt + 1) / 20001))
    report("D  does the preview benefit change with noise? (difference of differences)", Drow)


if __name__ == "__main__":
    main()
