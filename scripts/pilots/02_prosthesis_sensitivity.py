#!/usr/bin/env python
"""02 - Does the prosthesis actually matter (and how fragile is the gait to it)?

Question : If the prosthesis command is degraded, does walking get worse? And is it a smooth degradation or a cliff?
           * insensitive to even zero torque -> the human compensates alone, the prosthesis controller has nothing to
             learn, and P1 vs P2 / C1 vs C2 cannot show an effect.
           * cliff right next to the nominal torque (e.g. -25% already falls) -> the task is too brittle for RL
             on top; there is no headroom to improve smoothly.
Method   : take the Stage-1 policy (human + trained prosthesis actor) and perturb ONLY the prosthesis command:
             scale (0.5 .. 1.5), additive noise, delay (control steps, 30 Hz), constant offset, zero torque.
           Fixed seeds (same start phases in every condition), deterministic policy, long episodes, N episodes each.
Verdict  : FAIL  baseline already falls too often (Stage 1 not usable) -> go back to 00
           WARN  zero torque is as good as baseline (prosthesis irrelevant) OR a cliff within +-25 % of nominal
           PASS  baseline walks, zero/low torque clearly worse, degradation smooth
Exit code: 0 PASS, 1 WARN, 2 FAIL.

Example:
    python 02_prosthesis_sensitivity.py --session rl_train/results/train_session_X --episodes 20 --workers 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="stage-1 train_session_* folder (config + trained_models)")
    ap.add_argument("--checkpoint", default=None, help="default: latest checkpoint of the session")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--scales", default="0.5,0.75,1.25,1.5")
    ap.add_argument("--noise", default="0.1,0.3", help="additive N(0,sigma) in normalised action units")
    ap.add_argument("--delays", default="1,2,4", help="control steps of prosthesis-command delay")
    ap.add_argument("--offsets", default="", help="constant offsets, e.g. -0.2,0.2")
    ap.add_argument("--target-velocity", type=float, default=None)
    ap.add_argument("--baseline-max-fall", type=float, default=0.2)
    ap.add_argument("--cliff-jump", type=float, default=0.3, help="fall-rate jump between neighbouring scales that counts as a cliff")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    args = parse()
    root = common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/02")
    ck = common.resolve(args.checkpoint) if args.checkpoint else common.list_checkpoints(args.session)[-1][1]
    print("checkpoint:", ck)

    conds = [("baseline", "none", 0.0), ("zero torque", "zero", 0.0)]
    conds += [(f"scale x{v:g}", "scale", v) for v in floats(args.scales)]
    conds += [(f"noise s={v:g}", "noise", v) for v in floats(args.noise)]
    conds += [(f"delay {int(v)} step", "delay", v) for v in floats(args.delays)]
    conds += [(f"offset {v:+g}", "offset", v) for v in floats(args.offsets)]

    tasks = [
        dict(
            config=args.session,
            checkpoint=ck,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed0=0,  # identical start phases in every condition
            target_velocity=args.target_velocity,
            perturb={"kind": k, "value": v},
            label=name,
            extra={"kind": k, "value": v},
        )
        for name, k, v in conds
    ]
    print(f"{len(tasks)} conditions x {args.episodes} episodes, workers={args.workers}", flush=True)
    rows = common.run_parallel(tasks, args.workers, root=root)
    for r in rows:
        r["steps_frac"] = r["mean_steps"] / args.max_steps
    flat = common.strip_episodes(rows)
    common.save_rows(out, "02_sensitivity", flat)
    cols = ["label", "n", "fall_rate", "fall_ci_lo", "fall_ci_hi", "steps_frac", "mean_speed_mps", "speed_err_mps", "mean_abs_prosthesis_cmd"]
    print()
    common.print_table(flat, cols, {c: ".2f" for c in cols[2:]})

    by = {r["label"]: r for r in rows}
    base, zero = by["baseline"], by["zero torque"]
    scale_pts = sorted([(r["value"], r) for r in rows if r["kind"] == "scale"] + [(1.0, base)], key=lambda t: t[0])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        xs = [v for v, _ in scale_pts]
        ax[0].errorbar(xs, [r["fall_rate"] for _, r in scale_pts], yerr=[[r["fall_rate"] - r["fall_ci_lo"] for _, r in scale_pts], [r["fall_ci_hi"] - r["fall_rate"] for _, r in scale_pts]], fmt="o-", capsize=3)
        ax[0].axhline(zero["fall_rate"], ls="--", c="r", label="zero torque")
        ax[0].set_xlabel("prosthesis torque scale")
        ax[0].set_ylabel("fall rate")
        ax[0].legend()
        ax[0].set_title("torque scaling")
        ax[1].barh([r["label"] for r in rows][::-1], [r["fall_rate"] for r in rows][::-1])
        ax[1].set_xlabel("fall rate")
        ax[1].set_title("all conditions")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "02_sensitivity.png"), dpi=120)
    except Exception as e:
        print("(plot skipped)", e)

    reasons, status = [], "PASS"
    reasons.append(f"baseline fall rate {base['fall_rate']:.2f}, zero torque {zero['fall_rate']:.2f}")
    if base["fall_rate"] > args.baseline_max_fall:
        status = "FAIL"
        reasons.append(f"baseline falls too often (> {args.baseline_max_fall}) -> Stage 1 not usable yet, back to 00")
    else:
        d_fall = zero["fall_rate"] - base["fall_rate"]
        d_dist = base["mean_steps"] - zero["mean_steps"]
        if zero["fall_ci_lo"] <= base["fall_ci_hi"] and zero["steps_frac"] >= base["steps_frac"] - 0.05:
            status = "WARN"
            reasons.append("zero torque is NOT clearly worse than baseline: the prosthesis is (nearly) irrelevant -> no room for the "
                           "controller to matter. Consider heavier load on the prosthesis side / different task.")
        else:
            reasons.append(f"prosthesis matters: zero torque raises fall rate by {d_fall:+.2f} (survived steps {d_dist:+.0f})")
        # cliff: big jump between neighbouring scale points within +-25% of nominal
        for (v0, r0), (v1, r1) in zip(scale_pts[:-1], scale_pts[1:]):
            near = min(abs(v0 - 1), abs(v1 - 1)) <= 0.25 + 1e-9
            if near and abs(r1["fall_rate"] - r0["fall_rate"]) >= args.cliff_jump:
                status = "WARN" if status == "PASS" else status
                reasons.append(f"cliff: fall rate {r0['fall_rate']:.2f} @x{v0:g} -> {r1['fall_rate']:.2f} @x{v1:g}; the gait is brittle around nominal torque")
        if status == "PASS":
            reasons.append("degradation is smooth around nominal torque -> there is headroom for a learned controller")
    reasons.append("caveat: sensitivity of THIS stage-1 policy; a better stage-2 controller can still be robust or not")
    return common.verdict(out, "02_prosthesis_sensitivity", status, reasons, {"checkpoint": ck})


if __name__ == "__main__":
    sys.exit(main())
