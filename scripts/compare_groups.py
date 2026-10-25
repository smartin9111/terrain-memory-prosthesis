#!/usr/bin/env python
"""Bar chart with standard deviation, per-seed dots and seed-level tests for any set of groups of runs (the TRAINING SEED is the unit).

    python scripts/compare_groups.py "none C1=eval_noise/P_mlp/stage2_eval.csv@P1" "0.2 m C1=eval_exp/hz1_mlp/stage2_eval.csv" \
        "all C1=eval_noise/P_mlp/stage2_eval.csv@P2" --ref "none C1" --metric steps --out horizon_c1

Item = LABEL=csv[@COND]: a group of runs; @P1 / @P2 keeps only that condition of the csv. The same label twice merges the runs.
Per run only the LAST checkpoint, random terrain of --amp, value = mean over the held-out terrain seeds (steps: mean steps survived out of 1000;
falls: fall rate). Per group: mean, standard deviation over seeds (ddof=1), min, max, n. Against --ref (default: the first group): difference,
95% bootstrap interval and a two-sided permutation p (exact for small groups). Descriptive: with 4 seeds only large effects are detectable.
Writes <out>.png (bars = mean, error bars = +-1 SD, dots = seeds) and <out>.txt.
"""
import argparse
import itertools
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from terrain_memory_prosthesis.evaluation.analysis import read_rows  # noqa: E402

RNG = np.random.default_rng(0)


def per_run_values(path, cond, amp, metric):
    rows = [r for r in read_rows(path) if r["terrain_kind"] == "random" and abs(r["amplitude"] - amp) < 1e-9 and (cond is None or r["cond"] == cond)]
    last = {}
    for r in rows:
        last[r["run"]] = max(last.get(r["run"], 0), r["ckpt_steps"])
    acc = defaultdict(lambda: [0.0, 0])
    for r in rows:
        if r["ckpt_steps"] != last[r["run"]]:
            continue
        v = {"steps": lambda: float(r["mean_steps"]) * int(r["n"]), "falls": lambda: float(r["falls"]),
             "reward": lambda: float(r["mean_reward_per_step"]) * int(r["n"])}[metric]()
        acc[r["run"]][0] += v
        acc[r["run"]][1] += int(r["n"])
    return {k: s / n for k, (s, n) in acc.items()}


def perm_p(a, b):
    pool = np.array(list(a) + list(b), dtype=float)
    n, na = len(pool), len(a)
    obs = abs(np.mean(b) - np.mean(a))
    if comb(n, na) <= 20000:
        cnt = tot = 0
        for idx in itertools.combinations(range(n), na):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            cnt += abs(pool[~m].mean() - pool[m].mean()) >= obs - 1e-12
            tot += 1
        return cnt / tot
    cnt = 0
    for _ in range(20000):
        RNG.shuffle(pool)
        cnt += abs(pool[na:].mean() - pool[:na].mean()) >= obs - 1e-12
    return (cnt + 1) / 20001


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+")
    ap.add_argument("--ref", default=None, help="label of the reference group (default: the first)")
    ap.add_argument("--metric", default="steps", choices=["steps", "falls", "reward"],
                    help="steps = mean steps survived | falls = fall rate | reward = the environment reward per step (rows made after this was added)")
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", default="groups")
    a = ap.parse_args()

    groups = defaultdict(dict)
    for it in a.items:
        label, rest = it.split("=", 1)
        path, _, cond = rest.partition("@")
        groups[label].update(per_run_values(path, cond or None, a.amp, a.metric))
    labels = list(groups)
    ref = a.ref or labels[0]
    unit = {"steps": "mean steps survived (of 1000)", "falls": "fall rate", "reward": "mean reward per step"}[a.metric]
    lines = [f"{unit}, random terrain {a.amp * 100:g} cm, last checkpoint, unit = training seed, reference = '{ref}'\n",
             f"{'group':28} {'n':>2} {'mean':>8} {'SD':>7} {'min':>8} {'max':>8}  {'diff':>8}  {'95% CI':>18}  {'p':>6}"]
    stats = {}
    for lb in labels:
        v = np.array(list(groups[lb].values()))
        if len(v) == 0:
            lines.append(f"{lb:28} no runs found")
            continue
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        stats[lb] = (v.mean(), sd, v)
        row = f"{lb:28} {len(v):2d} {v.mean():8.1f} {sd:7.1f} {v.min():8.1f} {v.max():8.1f}"
        if lb != ref and ref in stats and len(v) > 1 and len(stats[ref][2]) > 1:
            x = stats[ref][2]
            boots = [RNG.choice(v, len(v)).mean() - RNG.choice(x, len(x)).mean() for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row += f"  {v.mean() - x.mean():+8.1f}  [{lo:+7.1f},{hi:+7.1f}]  {perm_p(x, v):6.3f}"
        lines.append(row)
    lines.append("\nSD = standard deviation over training seeds; CI = bootstrap over seeds (too narrow with few seeds); p = permutation test, "
                 "uncorrected for the number of groups compared (multiply by that number for a rough Bonferroni bound).")
    print("\n".join(lines))
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [lb for lb in labels if lb in stats]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.25 * len(names) + 1.5), 4.4))
    for i, lb in enumerate(names):
        m, sd, v = stats[lb]
        ax.bar(i, m, width=0.62, color="#9ecae1" if lb != ref else "#bdbdbd", edgecolor="none")
        ax.errorbar(i, m, yerr=0 if np.isnan(sd) else sd, color="black", capsize=4, lw=1.2)
        ax.scatter(np.full(len(v), i) + RNG.uniform(-0.12, 0.12, len(v)), v, s=14, color="black", alpha=0.6, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n", 1) if len(n) > 12 else n for n in names], fontsize=8.5)
    ax.set_ylabel(unit)
    ax.set_title(a.title or f"{unit}: mean, ±1 SD over seeds, dots = seeds", fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if a.metric == "steps":
        ax.set_ylim(0, 1000)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=160)
    print("written", a.out + ".png", a.out + ".txt")


if __name__ == "__main__":
    main()
