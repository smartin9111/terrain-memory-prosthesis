#!/usr/bin/env python
"""Plot the training curves of every MyoAssist session in one figure (PNG).

Left panel : mean training-episode length (control steps, 30 Hz) vs. timesteps
Right panel: mean reward per training episode vs. timesteps
Each session is labelled by its device and out_of_trajectory_threshold (read from session_config.json), so the
prosthesis run, the exoskeleton control run and the threshold variant are told apart automatically.

Caveat: training episodes stop early (imitation threshold), so absolute values are NOT walking distances; compare
curves of runs that use the SAME threshold. The walking check (00_stage1_walk_check.py) is the termination-independent test.

Usage (inside the UCloud job, after `source .ucloud_env.sh`):
    python plot_progress.py                        # all sessions -> <workspace>/progress.png
    python plot_progress.py --min-entries 5        # hide tiny runs (e.g. the `measure` run)
    python plot_progress.py --results DIR --out FILE.png
"""
import argparse
import glob
import json
import os
import sys


def find_default_results():
    here = os.path.dirname(os.path.abspath(__file__))
    ws = os.environ.get("WS") or os.path.abspath(os.path.join(here, "..", "..", ".."))
    return os.path.join(ws, "myoassist", "rl_train", "results"), ws


def load_session(path):
    try:
        entries = json.load(open(os.path.join(path, "train_log.json")))["log_datas"]
    except Exception:
        return None
    try:
        cfg = json.load(open(os.path.join(path, "session_config.json")))
        env = cfg.get("env_params", {})
        label = f"{env.get('device_key', '?')}, thr={env.get('out_of_trajectory_threshold', '?')}"
    except Exception:
        label = "?"
    steps = [e["num_timesteps"] for e in entries]
    ep_len = [e["average_num_timestep"] for e in entries]
    rew = [e["average_reward_per_episode"] for e in entries]
    return dict(name=os.path.basename(path), label=label, steps=steps, ep_len=ep_len, reward=rew)


def main():
    default_results, ws = find_default_results()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=default_results)
    ap.add_argument("--out", default=os.path.join(ws, "progress.png"))
    ap.add_argument("--min-entries", type=int, default=3, help="skip sessions with fewer log entries (default 3)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sessions = []
    for p in sorted(glob.glob(os.path.join(args.results, "train_session_*"))):
        s = load_session(p)
        if s and len(s["steps"]) >= args.min_entries:
            sessions.append(s)
    if not sessions:
        print("no sessions with enough log entries in", args.results)
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, s in enumerate(sessions):
        x = [v / 1e6 for v in s["steps"]]
        lab = f"{s['label']}  [{s['name'][-15:]}]"
        axes[0].plot(x, s["ep_len"], lw=2, label=lab, color=f"C{i}")
        axes[1].plot(x, s["reward"], lw=2, label=lab, color=f"C{i}")
        axes[0].annotate(f"{s['ep_len'][-1]:.0f}", (x[-1], s["ep_len"][-1]), textcoords="offset points", xytext=(4, 0), color=f"C{i}")
    axes[0].set_title("Mean training-episode length")
    axes[0].set_ylabel("control steps (30 Hz)   [1000 = full episode]")
    axes[1].set_title("Mean reward per training episode")
    for ax in axes:
        ax.set_xlabel("timesteps [millions]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Stage 1 training progress (higher = survives longer before the episode ends)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)
    for s in sessions:
        print(f"  {s['name']}  {s['label']:<32} latest {s['steps'][-1] / 1e6:5.2f} M steps  ep_len {s['ep_len'][-1]:5.1f}  reward {s['reward'][-1]:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
