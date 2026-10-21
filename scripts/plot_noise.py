#!/usr/bin/env python
"""Fall rate against sensor-noise level: P1/P2 x C1/K16, one dot per training seed, line = mean over seeds.

    python scripts/plot_noise.py W/C1=eval_noise/W_mlp/stage2_eval.csv W/K16=eval_noise/W_gru/stage2_eval.csv \
        M/C1=... M/K16=... S/C1=... S/K16=... [clean/C1=old.csv clean/C1=old2.csv clean/K16=...] --amp 0.06 --min-steps 7.9e6 --out noise_levels

Item = LEVEL/CONTROLLER=csv (level: any name, order = first appearance; known names get a label; controller: C1 K16; the same item twice merges seeds). Only random terrain of the given
amplitude and checkpoints >= --min-steps are used; per seed fall rate = falls / episodes over those checkpoints and terrain seeds.
The statistics are over TRAINING SEEDS (the unit that varies between runs), not over episodes: mean and range, no confidence interval
from pooled episodes. 'clean' = the old observation layout (before the sensor model): same task, a reference, not a strict control.
Writes <out>.png and <out>.txt (table).
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from terrain_memory_prosthesis.evaluation.analysis import read_rows  # noqa: E402

NAMES = {"clean": "clean\n(old layout)", "0": "clean\n(new layout)", "A": "noise\nα0.05", "B": "noise\nα0.1", "N": "noise\nα0.2", "D": "delay\n2 steps", "R": "dropout\n10%",
         "P": "practical\n0.3°, 6°/s,\n33 ms, 1%", "Q": "pessimistic\n1.1°, 23°/s,\n67 ms, 5%", "W": "W (stress): α0.2,\n1 step, 2%", "M": "M: α0.5,\n2 steps, 5%", "S": "S: α1.0,\n3 steps, 10%"}
COLORS = {"P1": "#0072B2", "P2": "#D55E00"}   # Okabe-Ito, as in the other figures
STYLES = {"C1": ("-", "o"), "K16": ("--", "s")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+")
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--min-steps", type=float, default=7.9e6)
    ap.add_argument("--out", default="noise_levels")
    a = ap.parse_args()

    cell = defaultdict(lambda: [0, 0])   # (level, ctrl, cond, seed) -> falls, n
    for it in a.items:
        key, path = it.split("=", 1)
        lv, ctrl = key.split("/")
        for r in read_rows(path):
            if r["terrain_kind"] != "random" or abs(r["amplitude"] - a.amp) > 1e-9 or r["ckpt_steps"] < a.min_steps:
                continue
            c = cell[(lv, ctrl, r["cond"], int(r["seed"]))]
            c[0] += int(r["falls"]); c[1] += int(r["n"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = list(dict.fromkeys(i.split("=", 1)[0].split("/")[0] for i in a.items))   # order of the items on the command line
    xs = {lv: i for i, lv in enumerate(levels)}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    lines = ["falls on random terrain %g cm; per training seed (dots), mean over seeds (line)\n" % (a.amp * 100)]
    lines.append(f"{'level':7} {'ctrl':5} {'cond':4} {'mean':>6} {'min':>6} {'max':>6}  seeds")
    jit = {("P1", "C1"): -0.09, ("P1", "K16"): -0.03, ("P2", "C1"): 0.03, ("P2", "K16"): 0.09}
    for ctrl in ("C1", "K16"):
        for cond in ("P1", "P2"):
            px, py = [], []
            for lv in levels:
                v = [f / n for (l, c, p, s), (f, n) in sorted(cell.items()) if (l, c, p) == (lv, ctrl, cond) and n]
                if not v:
                    continue
                x = xs[lv] + jit[(cond, ctrl)]
                ax.scatter([x] * len(v), v, s=14, alpha=0.45, color=COLORS[cond], marker=STYLES[ctrl][1], linewidths=0)
                px.append(x); py.append(sum(v) / len(v))
                lines.append(f"{lv:7} {ctrl:5} {cond:4} {py[-1]:6.2f} {min(v):6.2f} {max(v):6.2f}  {len(v)}")
            ax.plot(px, py, STYLES[ctrl][0], color=COLORS[cond], marker=STYLES[ctrl][1], ms=6, lw=1.8, label=f"{cond} {ctrl}")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([NAMES.get(lv, lv) for lv in levels], fontsize=9)
    ax.set_ylabel("fall rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.4, len(levels) - 0.6)
    ax.set_title("Falls vs. sensor imperfection (prosthesis controller)", fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=160)
    open(a.out + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written", a.out + ".png", a.out + ".txt")


if __name__ == "__main__":
    main()
