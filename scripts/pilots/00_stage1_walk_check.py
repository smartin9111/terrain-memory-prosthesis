#!/usr/bin/env python
"""00 - Does the human learn to walk with the powered prosthesis? (Stage 1 sanity check)

Question   : Is Stage 1 (human + prosthesis, flat terrain, no preview) learning at all, or is it stuck?
             Everything else (P1 vs P2, C1 vs C2) is meaningless if the human cannot walk with the OSL.
Input      : a finished OR still-running MyoAssist train session (the folder with trained_models/).
Method     : deterministic evaluation of up to --max-ckpts checkpoints (evenly spread), N episodes each,
             MyoAssist's own evaluation setup (long episodes, imitation early-termination off).
             Optional --control: a second session (e.g. the same human WITHOUT prosthesis / a different
             prosthesis setting) evaluated the same way, to separate "OSL is the problem" from "training is short".
Verdict    : PASS - latest checkpoint walks (fall rate <= --pass-fall-rate and survives >= --pass-steps-frac of max-steps)
             WARN - not there yet but still improving (keep training)
             FAIL - stalled: no real gain over the last --stall-after timesteps and still failing
Exit code  : 0 PASS, 1 WARN, 2 FAIL.

Note: the `average_num_timestep` / reward in train_log.json are computed on training episodes WITH the early
termination and random phases, so they are plotted only as a rough guide. The verdict uses deterministic eval.

Example (smoke / plumbing):
    python 00_stage1_walk_check.py --session rl_train/results/train_session_XXXX --episodes 2 --max-steps 100
Example (real):
    python 00_stage1_walk_check.py --session rl_train/results/train_session_XXXX --control path/to/other_session --workers 8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="train_session_* folder of the stage-1 run (prosthesis)")
    ap.add_argument("--control", default=None, help="optional comparison session evaluated the same way")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000, help="control steps per episode (30 Hz -> 1000 = 33 s)")
    ap.add_argument("--max-ckpts", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--target-velocity", type=float, default=None)
    ap.add_argument("--pass-fall-rate", type=float, default=0.2)
    ap.add_argument("--pass-steps-frac", type=float, default=0.8)
    ap.add_argument("--stall-after", type=float, default=5e6, help="timesteps of training before 'stalled' can be declared")
    ap.add_argument("--stall-gain", type=float, default=0.1, help="min gain in survived-step fraction over the second half of checkpoints")
    ap.add_argument("--root", default=None, help="MyoAssist checkout (default: MYOASSIST_DIR or installed rl_train)")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def eval_session(session, args, tag):
    cks = common.pick_checkpoints(session, args.max_ckpts)
    if not cks:
        raise SystemExit(f"No checkpoints in {session}/trained_models")
    tasks = [
        dict(
            config=session,
            checkpoint=p,
            episodes=args.episodes,
            max_steps=args.max_steps,
            target_velocity=args.target_velocity,
            seed0=0,
            label=f"{tag}:{ts}",
            extra={"run": tag, "timesteps": ts},
        )
        for ts, p in cks
    ]
    print(f"[{tag}] evaluating {len(tasks)} checkpoints x {args.episodes} episodes ...", flush=True)
    rows = common.run_parallel(tasks, args.workers, root=args.root)
    for r in rows:
        r["steps_frac"] = r["mean_steps"] / args.max_steps
    return rows


def train_curve(session):
    p = os.path.join(common.resolve(session), "train_log.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def plot(out, runs, args, curves):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print("(no matplotlib, skipping plot)", e)
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for tag, rows in runs.items():
        x = [r["timesteps"] for r in rows]
        ax[0].plot(x, [r["steps_frac"] for r in rows], "o-", label=tag)
        ax[1].errorbar(
            x,
            [r["fall_rate"] for r in rows],
            yerr=[[max(0.0, r["fall_rate"] - r["fall_ci_lo"]) for r in rows], [max(0.0, r["fall_ci_hi"] - r["fall_rate"]) for r in rows]],
            fmt="o-",
            capsize=3,
            label=tag,
        )
    ax[0].axhline(args.pass_steps_frac, ls="--", c="gray")
    ax[0].set_title("deterministic eval: survived fraction of episode")
    ax[1].axhline(args.pass_fall_rate, ls="--", c="gray")
    ax[1].set_title("fall rate (95% Wilson CI)")
    for a in ax[:2]:
        a.set_xlabel("timesteps")
        a.legend()
    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(0, 1.05)
    for tag, c in curves.items():
        logs = (c or {}).get("log_datas") if isinstance(c, dict) else None
        if logs:
            ax[2].plot([l["num_timesteps"] for l in logs], [l.get("average_reward_per_episode", float("nan")) for l in logs], label=tag)
    ax[2].set_xlabel("timesteps")
    ax[2].set_title("train_log: avg reward/episode (rough guide only)")
    if ax[2].get_legend_handles_labels()[0]:
        ax[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "00_stage1_walk_check.png"), dpi=120)


def main():
    args = parse()
    root = common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/00")
    args.root = root
    runs = {"prosthesis": eval_session(args.session, args, "prosthesis")}
    if args.control:
        runs["control"] = eval_session(args.control, args, "control")

    allrows = [r for rows in runs.values() for r in rows]
    common.save_rows(out, "00_checkpoints", common.strip_episodes(allrows))
    cols = ["run", "timesteps", "n", "falls", "fall_rate", "fall_ci_lo", "fall_ci_hi", "steps_frac", "mean_speed_mps", "speed_err_mps"]
    print()
    common.print_table(common.strip_episodes(allrows), cols, {c: ".2f" for c in cols[4:]})
    plot(out, runs, args, {t: train_curve(s) for t, s in (("prosthesis", args.session), ("control", args.control)) if s})

    rows = runs["prosthesis"]
    last = rows[-1]
    reasons = [
        f"latest checkpoint {last['timesteps']}: fall rate {last['fall_rate']:.2f} "
        f"[{last['fall_ci_lo']:.2f},{last['fall_ci_hi']:.2f}], survives {last['steps_frac']:.0%} of {args.max_steps} steps, "
        f"speed {last['mean_speed_mps']:.2f} m/s"
    ]
    walks = last["fall_rate"] <= args.pass_fall_rate and last["steps_frac"] >= args.pass_steps_frac
    if "control" in runs:
        c = runs["control"][-1]
        reasons.append(f"control (ckpt {c['timesteps']}): fall rate {c['fall_rate']:.2f}, survives {c['steps_frac']:.0%}")
    if walks:
        status = "PASS"
        reasons.append("the human walks with the prosthesis on flat ground -> Stage 1 is usable, continue to 01/02")
    else:
        t_last = last["timesteps"]
        half = [r for r in rows if r["timesteps"] >= t_last / 2]
        gain = (half[-1]["steps_frac"] - half[0]["steps_frac"]) if len(half) >= 2 else float("nan")
        if t_last >= args.stall_after and len(half) >= 2 and gain < args.stall_gain:
            status = "FAIL"
            reasons.append(
                f"stalled: survived-fraction gain over the 2nd half of training = {gain:.2f} (< {args.stall_gain}) at {t_last} steps. "
                "Check: prosthesis dynamics/joint limits, reward weights, reference gait (no ankle in ref), more steps."
            )
        else:
            status = "WARN"
            reasons.append(
                "not walking yet; "
                + (f"trained only {t_last} steps (stall can be declared from {int(args.stall_after)})" if t_last < args.stall_after else f"gain {gain:.2f} still positive")
                + " -> keep training and re-run on the newer checkpoints"
            )
    return common.verdict(out, "00_stage1_walk_check", status, reasons, {"args": vars(args)})


if __name__ == "__main__":
    sys.exit(main())
