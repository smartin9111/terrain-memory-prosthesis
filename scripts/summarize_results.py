#!/usr/bin/env python
"""What is in the results folder?  An inventory of all Stage-2 runs joined with their evaluation, and the GRU time-window figure.

    python scripts/summarize_results.py --results /work/ai3_test_train/myoassist/rl_train/results \
        --eval /work/ai3_test_train/eval_noise_r /work/ai3_test_train/eval_exp --out /work/ai3_test_train/summary_$(date +%Y%m%d-%H%M)

--eval takes folders that contain <something>/stage2_eval.csv (eval_noise.sh / eval_exp.sh output); a run is joined by its folder name.
Writes in --out:
  inventory.csv / inventory.md   one row per run: tag, condition, controller, seed, window K [steps and s], GRU hidden, preview distances, sensor noise
                                 (angle / velocity / force std, delay, dropout), steps trained, and (if evaluated) steps survived / fall rate / reward on
                                 random terrain of --amp at the LAST checkpoint
  groups.md                      the same, averaged over seeds per configuration (mean, SD, n)
  gru_window.png / .txt          C2 only: steps survived against the window length (seconds) and against the GRU hidden size, with the C1 reference band
Everything is read from stage2_info.json of each run (the settings the run was trained with), nothing is typed in.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from terrain_memory_prosthesis.evaluation.analysis import read_rows  # noqa: E402

CONTROL_HZ = 30.0


def sensor_summary(env):
    env = env or {}
    out = {"noise_ang": "", "noise_vel": "", "noise_force": "", "delay_steps": "", "dropout": ""}
    try:
        std = json.loads(env.get("MYO_SENSOR_NOISE_STD", "null"))
        if isinstance(std, list):
            out["noise_ang"] = max(std[0:8]) if len(std) >= 8 else ""
            out["noise_vel"] = max(std[8:17]) if len(std) >= 17 else ""
            out["noise_force"] = max(std[35:39]) if len(std) >= 39 else ""
        elif std is not None:
            out["noise_ang"] = out["noise_vel"] = out["noise_force"] = std
    except Exception:
        pass
    out["delay_steps"] = env.get("MYO_SENSOR_DELAY_STEPS", "")
    out["dropout"] = env.get("MYO_SENSOR_DROPOUT_PROBABILITY", "")
    return out


def load_runs(results):
    runs = {}
    for p in sorted(glob.glob(os.path.join(results, "stage2_*"))):
        f = os.path.join(p, "stage2_info.json")
        if not os.path.isfile(f):
            continue
        try:
            info = json.load(open(f))
        except Exception:
            continue
        a = info.get("args", {})
        cenv, penv = info.get("controller_env") or {}, info.get("perception_env") or {}
        name = os.path.basename(p)
        m = re.match(r"stage2_P[12]_(?:mlp|gru)_s\d+_(.+)_\d{8}-\d{6}$", name)
        ctrl = a.get("controller", "mlp")
        ck = [int(re.findall(r"model_(\d+)\.zip", c)[0]) for c in glob.glob(os.path.join(p, "trained_models", "model_*.zip"))]
        pd = penv.get("MYO_PREVIEW_DISTANCES", "")
        row = dict(run=name, tag=m.group(1) if m else a.get("tag", ""), cond=a.get("cond", ""), controller="C2 (gru)" if ctrl == "gru" else "C1 (mlp)",
                   seed=a.get("seed", ""), window_K=cenv.get("MYO_HISTORY", "") if ctrl == "gru" else "",
                   window_s=round(int(cenv["MYO_HISTORY"]) / CONTROL_HZ, 2) if ctrl == "gru" and cenv.get("MYO_HISTORY") else "",
                   gru_hidden=cenv.get("MYO_GRU_HIDDEN", "") if ctrl == "gru" else "",
                   preview=("none" if a.get("cond") == "P1" else (pd or "0.2,0.4,0.6,0.8,1.0 (default)")),
                   steps_trained=max(ck) if ck else 0, **sensor_summary(penv))
        runs[name] = row
    return runs


def join_eval(runs, eval_dirs, amp):
    for d in eval_dirs:
        for f in sorted(glob.glob(os.path.join(d, "*", "stage2_eval.csv"))):
            rows = [r for r in read_rows(f) if r["terrain_kind"] == "random" and abs(r["amplitude"] - amp) < 1e-9]
            last = {}
            for r in rows:
                last[r["run"]] = max(last.get(r["run"], 0), r["ckpt_steps"])
            acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])
            for r in rows:
                if r["ckpt_steps"] != last[r["run"]]:
                    continue
                n = int(r["n"])
                x = acc[r["run"]]
                x[0] += float(r["mean_steps"]) * n
                x[1] += float(r["falls"])
                if r.get("mean_reward_per_step") not in (None, ""):
                    x[2] += float(r["mean_reward_per_step"]) * n
                    x[4] += n
                x[3] += n
            for run, (s, fl, rw, n, nr) in acc.items():
                if run in runs and n:
                    runs[run].update(steps_survived=round(s / n, 1), fall_rate=round(fl / n, 3), reward_per_step=round(rw / nr, 4) if nr else "",
                                     evaluated_in=os.path.basename(d.rstrip("/")))
    return runs


def mean_sd(v):
    v = np.array(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the folder with the stage2_* run directories")
    ap.add_argument("--eval", nargs="*", default=[], help="evaluation folders (eval_noise_r, eval_exp, ...)")
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--out", default="summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = join_eval(load_runs(a.results), a.eval, a.amp)
    if not runs:
        raise SystemExit(f"no stage2_* runs with stage2_info.json in {a.results}")
    cols = ["run", "tag", "cond", "controller", "seed", "window_K", "window_s", "gru_hidden", "preview", "noise_ang", "noise_vel", "noise_force",
            "delay_steps", "dropout", "steps_trained", "steps_survived", "fall_rate", "reward_per_step", "evaluated_in"]
    rows = sorted(runs.values(), key=lambda r: (r["tag"], r["cond"], r["controller"], int(r["seed"]) if str(r["seed"]).isdigit() else 0))
    with open(os.path.join(a.out, "inventory.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "inventory.md"), "w") as f:
        f.write(f"# Runs in {a.results}\n\nsteps survived / fall rate / reward: last checkpoint, random terrain {a.amp * 100:g} cm (blank = not evaluated)\n\n")
        f.write("| " + " | ".join(cols[1:]) + " |\n|" + "---|" * (len(cols) - 1) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols[1:]) + " |\n")

    # one line per configuration: mean, SD, n over seeds
    key = lambda r: (r["tag"], r["cond"], r["controller"], r["window_K"], r["gru_hidden"], r["preview"], r["noise_ang"], r["delay_steps"], r["dropout"])  # noqa: E731
    groups = defaultdict(list)
    for r in rows:
        if "steps_survived" in r:
            groups[key(r)].append(r)
    with open(os.path.join(a.out, "groups.md"), "w") as f:
        f.write(f"# Configurations (mean over seeds)\n\n| tag | cond | controller | K | hidden | preview | n seeds | steps survived mean | SD | fall rate mean |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for k, rr in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            m, s = mean_sd([r["steps_survived"] for r in rr])
            f.write(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {k[5]} | {len(rr)} | {m:.1f} | {s:.1f} | {np.mean([r['fall_rate'] for r in rr]):.2f} |\n")
    print(f"{len(rows)} runs, {sum(1 for r in rows if 'steps_survived' in r)} evaluated -> {a.out}/inventory.csv, inventory.md, groups.md")

    # the GRU time-window figure: C2, P2, level P (tags nP and g*), reference = C1 P2 level P
    gru = [r for r in rows if r["controller"].startswith("C2") and r["cond"] == "P2" and "steps_survived" in r and (r["tag"] == "nP" or r["tag"].startswith("g"))]
    c1 = [r["steps_survived"] for r in rows if r["controller"].startswith("C1") and r["cond"] == "P2" and r["tag"] == "nP" and "steps_survived" in r]
    if not gru:
        print("no evaluated C2 runs (tags nP, g*): gru_window.png skipped")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    lines = ["C2, P2, sensor level P: steps survived (random terrain %g cm, last checkpoint), mean +- SD over seeds\n" % (a.amp * 100)]
    panels = [("window length K (control steps, 30 per second)", lambda r: int(r["window_K"]), lambda r: str(r["gru_hidden"]) == "32", "hidden 32"),
              ("GRU hidden size (K = 16)", lambda r: int(r["gru_hidden"]), lambda r: str(r["window_K"]) == "16", "K = 16")]
    for axis, (xl, xf, sel, note) in zip(ax, panels):
        byx = defaultdict(list)
        for r in gru:
            if sel(r):
                byx[xf(r)].append(r["steps_survived"])
        xs = sorted(byx)
        for i, x in enumerate(xs):
            m, s = mean_sd(byx[x])
            axis.scatter(np.full(len(byx[x]), i) + np.linspace(-0.08, 0.08, len(byx[x])), byx[x], s=14, color="black", alpha=0.55, zorder=3)
            axis.errorbar(i, m, yerr=0 if np.isnan(s) else s, fmt="o", color="#D55E00", capsize=4, ms=7, zorder=4)
            lines.append(f"{xl.split(' (')[0]:28} {x:4d}  mean {m:7.1f}  SD {s:6.1f}  n={len(byx[x])}  ({note})")
        axis.set_xticks(range(len(xs)))
        axis.set_xticklabels([f"{x}\n({x / CONTROL_HZ:.2f} s)" if "window" in xl else str(x) for x in xs])
        axis.set_xlabel(xl)
        if c1:
            m, s = mean_sd(c1)
            axis.axhspan(m - (0 if np.isnan(s) else s), m + (0 if np.isnan(s) else s), color="#0072B2", alpha=0.15)
            axis.axhline(m, color="#0072B2", lw=1.2, label="C1 (feed-forward), mean +- SD")
        axis.grid(axis="y", alpha=0.25)
        for sp in ("top", "right"):
            axis.spines[sp].set_visible(False)
    ax[0].set_ylabel("mean steps survived (of 1000)")
    ax[0].set_ylim(0, 1000)
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("GRU memory: how long a window, how large a state?  (dots = seeds, orange = mean +- SD)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "gru_window.png"), dpi=160)
    open(os.path.join(a.out, "gru_window.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written", os.path.join(a.out, "gru_window.png"))


if __name__ == "__main__":
    main()
