"""Turn Stage-2 evaluation rows (scripts/eval_stage2.py) and training logs into the comparison figures and tables.

Pure functions (testable without MuJoCo); `scripts/analyze_stage2.py` is the CLI. Cell of the 2x2 = (perception, controller):
P1/P2 x C1/C2; the Stage-1 policy without Stage 2 is shown as the reference ("Stage 1"). Runs of the same cell (different
seeds) are pooled by summing falls and episodes; per-seed values are drawn as dots, never hidden behind the mean.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict

from terrain_memory_prosthesis.evaluation.reporting import wilson_ci

CONTROLLER_NAMES = {"mlp": "C1", "gru": "C2"}
COLORS = {"P1": "#0072B2", "P2": "#D55E00", "Stage 1": "#555555"}  # colour-blind safe (Okabe-Ito)
LINESTYLES = {"C1": "-", "C2": "--", "": ":"}
WEIGHTED = ("steps_frac", "mean_distance_m", "speed_err_mps", "mean_abs_prosthesis_cmd", "mean_steps", "mean_speed_mps")
NUMERIC = ("ckpt_steps", "amplitude", "terrain_seed", "n", "falls", "fall_rate", "fall_ci_lo", "fall_ci_hi", "seed") + WEIGHTED


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            if k in r and r[k] not in ("", None):
                r[k] = float(r[k])
    return rows


def cell_label(r: dict) -> str:
    """'P1xC1' ... or 'Stage 1' for the reference policy."""
    if r.get("cond") == "stage1":
        return "Stage 1"
    return f"{r['cond']}x{CONTROLLER_NAMES.get(r.get('controller', 'mlp'), r.get('controller', ''))}"


def split_label(label: str) -> tuple[str, str]:
    if label == "Stage 1":
        return "Stage 1", ""
    p, c = label.split("x")
    return p, c


def style(label: str) -> dict:
    p, c = split_label(label)
    return dict(color=COLORS.get(p, "#333333"), linestyle=LINESTYLES.get(c, "-"))


def pool(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Group rows by `keys`; falls and n are summed, means are weighted by n, the Wilson interval is recomputed."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rr in groups.items():
        n = int(sum(r["n"] for r in rr))
        k = int(sum(r["falls"] for r in rr))
        lo, hi = wilson_ci(k, n)
        row = dict(zip(keys, key), n=n, falls=k, fall_rate=k / n if n else float("nan"), fall_ci_lo=lo, fall_ci_hi=hi)
        for m in WEIGHTED:
            if all(m in r for r in rr):
                row[m] = sum(r[m] * r["n"] for r in rr) / n if n else float("nan")
        out.append(row)
    return out


def with_labels(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["cell"] = cell_label(r)
        r["amp_cm"] = float(r["amplitude"]) * 100.0
    return rows


def latest_ckpt_rows(rows: list[dict]) -> list[dict]:
    """Only the last evaluated checkpoint of every run (the other ones are for the learning curve)."""
    last: dict = {}
    for r in rows:
        last[r["run"]] = max(last.get(r["run"], 0), r["ckpt_steps"])
    return [r for r in rows if r["ckpt_steps"] == last[r["run"]]]


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference p2 - p1 of two proportions with the Newcombe (hybrid score) 95 % interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def terrain_name(kind: str, amp_cm: float) -> str:
    return "flat" if kind == "flat" else f"{kind} {amp_cm:g} cm"


def summary_table(rows: list[dict]) -> list[dict]:
    """One row per cell x terrain (all terrain seeds and training seeds pooled)."""
    pooled = pool(with_labels(rows), ("cell", "terrain_kind", "amp_cm"))
    order = {"Stage 1": 9, "P1xC1": 0, "P2xC1": 1, "P1xC2": 2, "P2xC2": 3}
    kind_order = {"flat": 0, "random": 1, "sinusoidal": 2}
    pooled.sort(key=lambda r: (kind_order.get(r["terrain_kind"], 3), r["amp_cm"], order.get(r["cell"], 5)))
    for r in pooled:
        r["terrain"] = terrain_name(r["terrain_kind"], r["amp_cm"])
    return pooled


def difference_table(table: list[dict]) -> list[dict]:
    """P2 minus P1 fall rate (same controller) per terrain, with the Newcombe interval. Empty if a cell is missing."""
    by = {(r["cell"], r["terrain"]): r for r in table}
    out = []
    for c in ("C1", "C2"):
        for t in dict.fromkeys(r["terrain"] for r in table):
            a, b = by.get((f"P1x{c}", t)), by.get((f"P2x{c}", t))
            if a and b and a["n"] and b["n"]:
                d, lo, hi = newcombe_diff(a["falls"], a["n"], b["falls"], b["n"])
                out.append(dict(controller=c, terrain=t, p1=a["fall_rate"], p2=b["fall_rate"], diff_p2_minus_p1=d, ci_lo=lo, ci_hi=hi,
                                n_p1=a["n"], n_p2=b["n"]))
    return out


def markdown_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cell = lambda r, c: (format(r[c], fmt[c]) if c in fmt and isinstance(r.get(c), (int, float)) else str(r.get(c, "")))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(r, c) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- training logs
def load_training_log(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "train_log.json")) as f:
        return json.load(f)["log_datas"]


def run_cell(run_dir: str) -> tuple[str, int]:
    """(cell label, seed) of a Stage-2 run directory, from its stage2_info.json."""
    with open(os.path.join(run_dir, "stage2_info.json")) as f:
        a = json.load(f)["args"]
    return f"{a['cond']}x{CONTROLLER_NAMES.get(a.get('controller', 'mlp'), a.get('controller', ''))}", int(a.get("seed", 0))


def smooth(y: list[float], w: int = 5) -> list[float]:
    if w <= 1 or len(y) < 2:
        return list(y)
    out = []
    for i in range(len(y)):
        seg = y[max(0, i - w + 1): i + 1]
        out.append(sum(seg) / len(seg))
    return out


TRAIN_PANELS = [("average_reward_per_episode", "mean reward per training episode"),
                ("average_num_timestep", "mean training-episode length [control steps]"),
                ("std", "policy std (exploration)"),
                ("explained_variance", "critic explained variance")]


# ---------------------------------------------------------------- figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    return plt


def fig_fall_rate(rows: list[dict], path: str, per_seed: bool = True):
    """Left: fall rate vs random-terrain amplitude (flat = 0), 95 % Wilson intervals, dots = single training seeds.
    Right: sinusoidal terrain."""
    plt = _mpl()
    rows = with_labels(list(rows))
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [3, 1]})
    for i, cell in enumerate(cells):
        sel = [r for r in rows if r["cell"] == cell and r["terrain_kind"] in ("flat", "random")]
        pts = sorted(pool(sel, ("amp_cm",)), key=lambda r: r["amp_cm"])
        if not pts:
            continue
        x = [p["amp_cm"] + (i - (len(cells) - 1) / 2) * 0.12 for p in pts]
        y = [p["fall_rate"] for p in pts]
        err = [[p["fall_rate"] - p["fall_ci_lo"] for p in pts], [p["fall_ci_hi"] - p["fall_rate"] for p in pts]]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{cell} (n={pts[0]['n']}/point)", **style(cell))
        if per_seed:
            seeds = sorted({r.get("seed", 0) for r in sel})
            if len(seeds) > 1:
                for s in seeds:
                    sp = sorted(pool([r for r in sel if r.get("seed", 0) == s], ("amp_cm",)), key=lambda r: r["amp_cm"])
                    ax.scatter([p["amp_cm"] for p in sp], [p["fall_rate"] for p in sp], s=12, color=style(cell)["color"], alpha=0.45, zorder=3)
    ax.set_xlabel("random terrain amplitude [cm] (0 = flat)")
    ax.set_ylabel("fall rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Falls vs terrain roughness (held-out terrains)")
    ax.legend(fontsize=8, loc="upper left")
    sin = [r for r in rows if r["terrain_kind"] == "sinusoidal"]
    if sin:
        pts = {p["cell"]: p for p in pool(sin, ("cell",))}
        for i, cell in enumerate(cells):
            if cell in pts:
                p = pts[cell]
                bx.bar(i, p["fall_rate"], color=style(cell)["color"], alpha=0.8)
                bx.errorbar(i, p["fall_rate"], yerr=[[p["fall_rate"] - p["fall_ci_lo"]], [p["fall_ci_hi"] - p["fall_rate"]]], color="k", capsize=3)
        bx.set_xticks(range(len(cells)))
        bx.set_xticklabels(cells, rotation=30, ha="right", fontsize=8)
        bx.set_ylim(0, 1.03)
        bx.set_title("sinusoidal 4 cm")
    else:
        bx.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_metrics(rows: list[dict], path: str):
    plt = _mpl()
    rows = with_labels(list(rows))
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    panels = [("steps_frac", "fraction of the episode survived"), ("mean_distance_m", "distance walked [m]"),
              ("speed_err_mps", "speed error vs target [m/s]"), ("mean_abs_prosthesis_cmd", "mean |prosthesis command|")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (m, title) in zip(axs.ravel(), panels):
        for cell in cells:
            sel = [r for r in rows if r["cell"] == cell and r["terrain_kind"] in ("flat", "random") and m in r]
            pts = sorted(pool(sel, ("amp_cm",)), key=lambda r: r["amp_cm"])
            if pts and m in pts[0]:
                ax.plot([p["amp_cm"] for p in pts], [p[m] for p in pts], marker="o", label=cell, **style(cell))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("random terrain amplitude [cm]")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_training(runs: list[str], path: str):
    """Training curves of every run (colour = perception, dashes = controller, thin = single seeds)."""
    plt = _mpl()
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5))
    seen = set()
    for d in runs:
        try:
            cell, seed = run_cell(d)
            log = load_training_log(d)
        except (OSError, KeyError, ValueError):
            continue
        for ax, (k, title) in zip(axs.ravel(), TRAIN_PANELS):
            pts = [(e["num_timesteps"], e[k]) for e in log if k in e and e[k] not in (-1, None)]
            if not pts:
                continue
            x, y = zip(*pts)
            lab = f"{cell}" if cell not in seen else None
            ax.plot(x, smooth(list(y)), alpha=0.9, linewidth=1.3, label=lab, **style(cell))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Stage-2 timesteps")
        seen.add(cell)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Stage-2 training (caveat: training-episode length is not walking distance; trust the held-out evaluation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_checkpoints(rows: list[dict], path: str, terrains=(("random", 4.0), ("random", 6.0))) -> bool:
    """Fall rate of the held-out terrains vs training steps (needs several evaluated checkpoints per run). False if not possible."""
    plt = _mpl()
    rows = with_labels(list(rows))
    if len({r["ckpt_steps"] for r in rows if r["cond"] != "stage1"}) < 2:
        return False
    fig, axs = plt.subplots(1, len(terrains), figsize=(5 * len(terrains), 3.8), squeeze=False)
    for ax, (kind, amp) in zip(axs[0], terrains):
        for cell in dict.fromkeys(r["cell"] for r in rows if r["cond"] != "stage1"):
            sel = [r for r in rows if r["cell"] == cell and r["terrain_kind"] == kind and abs(r["amp_cm"] - amp) < 1e-6]
            pts = sorted(pool(sel, ("ckpt_steps",)), key=lambda r: r["ckpt_steps"])
            if pts:
                ax.errorbar([p["ckpt_steps"] for p in pts], [p["fall_rate"] for p in pts],
                            yerr=[[p["fall_rate"] - p["fall_ci_lo"] for p in pts], [p["fall_ci_hi"] - p["fall_rate"] for p in pts]],
                            marker="o", capsize=3, label=cell, **style(cell))
        ref = [r for r in rows if r["cond"] == "stage1" and r["terrain_kind"] == kind and abs(r["amp_cm"] - amp) < 1e-6]
        if ref:
            p = pool(ref, ("cell",))[0]
            ax.axhline(p["fall_rate"], color=COLORS["Stage 1"], linestyle=":", label="Stage 1 (start)")
        ax.set_title(f"{kind} {amp:g} cm")
        ax.set_xlabel("Stage-2 timesteps")
        ax.set_ylabel("fall rate")
        ax.set_ylim(-0.03, 1.03)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def expand_runs(patterns: list[str]) -> list[str]:
    out = []
    for p in patterns:
        out += sorted(d for d in glob.glob(os.path.expanduser(p)) if os.path.isdir(d))
    return out
