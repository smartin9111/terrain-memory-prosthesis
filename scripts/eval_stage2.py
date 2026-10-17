#!/usr/bin/env python
"""Evaluate Stage-2 runs (and the Stage-1 policy as reference) on a FIXED set of held-out terrains.

    python scripts/eval_stage2.py --runs rl_train/results/stage2_P1_mlp_s0_* rl_train/results/stage2_P2_mlp_s0_* \
        --stage1-session rl_train/results/train_session_<stamp> --episodes 10 --workers 16 --out eval_stage2

Per run the last checkpoint is evaluated (--ckpt-steps N picks the checkpoint closest to N steps instead). Deterministic
policy, same episode seeds for every run, terrain geometry identical to training (160 m, 640 cells = 25 cm; a different
cell size makes even the flat-trained gait fall, see docs/stage1_findings.md). "fell" = the env's own `terminated`;
the end of the reference data (`truncated`) is not a fall.

Terrain set (held out: the training terrain is re-drawn every episode, these seeds are fixed):
    flat; random amplitude 2/4/6 cm (inside the training range 2-6 cm) and 8 cm (outside = extrapolation), terrain seeds 1-3;
    sinusoidal 4 cm, period 1 m (a different terrain family).
Outputs: <out>/stage2_eval.csv|json (one row per run x terrain), a printed table per run x terrain family.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilots"))
import common  # noqa: E402

EXTENT, RES = 160.0, 640


def terrains(seeds):
    out = [("flat", 0.0, 0, None)]
    for amp in (0.02, 0.04, 0.06, 0.08):
        for s in seeds:
            out.append(("random", amp, s, common.terrain_spec("random", amp, s, extent=EXTENT, resolution=RES)))
    out.append(("sinusoidal", 0.04, 0, common.terrain_spec("sinusoidal", 0.04, 0, period=1.0, extent=EXTENT, resolution=RES)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[], help="Stage-2 session folders (globs allowed)")
    ap.add_argument("--stage1-session", default=None, help="Stage-1 session: evaluated as the no-Stage-2 reference")
    ap.add_argument("--ckpt-steps", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--terrain-seeds", default="1,2,3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="eval_stage2")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = common.init(a.root)
    out = common.out_dir(a.out, "eval_stage2")

    entries = []  # (label, session, condition, env vars of the run)
    for pat in a.runs:
        for p in sorted(glob.glob(common.resolve(pat))):
            with open(os.path.join(p, "stage2_info.json")) as f:
                info = json.load(f)
            # the run's own perception settings (MYO_PERCEPTION, preview distances): evaluation must see what training saw
            entries.append((os.path.basename(p), p, info["args"]["cond"], info.get("perception_env")))
    if a.stage1_session:
        entries.append(("stage1_reference", common.resolve(a.stage1_session), "stage1", None))
    if not entries:
        raise SystemExit("nothing to evaluate (--runs / --stage1-session)")

    seeds = [int(x) for x in a.terrain_seeds.split(",")]
    tasks = []
    for label, sess, cond, penv in entries:
        cks = common.list_checkpoints(sess)
        if not cks:
            print(f"skip {label}: no checkpoints")
            continue
        steps, ck = min(cks, key=lambda c: abs(c[0] - a.ckpt_steps)) if a.ckpt_steps else cks[-1]
        for kind, amp, ts, spec in terrains(seeds):
            tasks.append(dict(config=os.path.join(sess, "session_config.json"), checkpoint=ck, episodes=a.episodes, max_steps=a.max_steps,
                              seed0=0, terrain=spec if spec is not None else None, root=root, env=penv, label=f"{label}_{kind}{amp}_{ts}",
                              extra=dict(run=label, cond=cond, ckpt_steps=steps, terrain_kind=kind, amplitude=amp, terrain_seed=ts)))
    print(f"{len(tasks)} evaluation tasks, {a.workers} workers", flush=True)
    rows = common.strip_episodes(common.run_parallel(tasks, workers=a.workers, root=root))
    for r in rows:
        r["steps_frac"] = r["mean_steps"] / a.max_steps
    common.save_rows(out, "stage2_eval", rows)

    import statistics as st

    table = []
    for run in dict.fromkeys(r["run"] for r in rows):
        for kind, amp in dict.fromkeys((r["terrain_kind"], r["amplitude"]) for r in rows if r["run"] == run):
            rr = [r for r in rows if r["run"] == run and r["terrain_kind"] == kind and r["amplitude"] == amp]
            table.append(dict(run=run, cond=rr[0]["cond"], terrain=f"{kind} {amp * 100:g} cm", fall_rate=st.mean(r["fall_rate"] for r in rr),
                              steps_frac=st.mean(r["steps_frac"] for r in rr), distance_m=st.mean(r["mean_distance_m"] for r in rr),
                              speed_err=st.mean(r["speed_err_mps"] for r in rr), n=sum(r["n"] for r in rr)))
    common.save_rows(out, "stage2_summary", table)
    cols = ["run", "cond", "terrain", "fall_rate", "steps_frac", "distance_m", "speed_err", "n"]
    common.print_table(table, cols, {c: ".3f" for c in ("fall_rate", "steps_frac", "distance_m", "speed_err")})


if __name__ == "__main__":
    main()
