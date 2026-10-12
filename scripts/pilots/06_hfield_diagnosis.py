#!/usr/bin/env python
"""06 - Is the heightfield ground itself the problem? (03a: every terrain, even 0.5 mm, made the walker fall.)

Question : A 0.5 mm 'terrain' cannot be hard. So either the heightfield contact path is an artifact for a gait that was trained
           on the infinite PLANE, or something in the terrain setup is broken. This pilot separates the cases.
Method   : evaluate the SAME checkpoint (deterministic, same seeds) on
             plane (no terrain)                         = reference
             hfield random, amplitude 1e-5 m            at several grid resolutions / extents
             hfield sinusoidal, amplitude 1e-5 m
           and record, per variant, the fall rate, distance, when it falls, and the contact statistics while walking
           (mean number of contacts, deepest penetration).
Verdict  : PASS  all variants walk like the plane      -> the heightfield path is fine; terrain roughness is the real difficulty
           WARN  some resolution walks, others fall    -> grid resolution / extent matters (see table); use the good one
           FAIL  plane walks, EVERY heightfield falls  -> heightfield contact is an artifact for this gait (or brittleness to the
                 contact model). Fix: train Stage 1 on a heightfield too (amplitude ~1e-5 .. small curriculum), then re-run.
Exit code: 0 PASS, 1 WARN, 2 FAIL.

    python scripts/pilots/06_hfield_diagnosis.py --session <session dir> --episodes 10 --max-steps 600 --workers 8
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--root", default=None)
_known, _ = _pre.parse_known_args()
common.init(_known.root)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: latest")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--amplitude", type=float, default=1e-5, help="'flat' heightfield relief [m]")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def diag_task(task):
    """Worker: one variant, N deterministic episodes; also logs contact statistics of the walking steps."""
    import numpy as np

    common.init(task.get("root"))
    cfg = common.load_config(task["config"])
    cfg.env_params.terrain = task["terrain"]
    env = common.make_eval_env(cfg, task["max_steps"])
    model = common.load_model(task["checkpoint"])
    m, d = env.sim.model, env.sim.data
    m, d = getattr(m, "ptr", m), getattr(d, "ptr", d)
    eps = []
    ncon, depth = [], []
    t0 = time.time()
    for i in range(task["episodes"]):
        with common.seeded_rng(i):
            obs, _ = env.reset()
        x0 = float(env.sim.data.joint("pelvis_tx").qpos[0])
        fell, steps = False, 0
        for t in range(task["max_steps"]):
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _ = env.step(a)
            steps += 1
            ncon.append(int(d.ncon))
            if d.ncon:
                depth.append(float(min(c.dist for c in d.contact[: d.ncon])))
            if term:
                fell = True
                break
            if trunc:
                break
        x1 = float(env.sim.data.joint("pelvis_tx").qpos[0])
        eps.append({"seed": i, "steps": steps, "fell": fell, "distance": x1 - x0, "speed": (x1 - x0) / (steps * env.dt) if steps else 0.0,
                    "speed_err": 0.0, "prosthesis_cmd": 0.0})
    env.close()
    res = {"label": task["label"], **common.summarize(eps), "wall_s": time.time() - t0}
    res["mean_contacts"] = float(np.mean(ncon)) if ncon else float("nan")
    res["deepest_penetration_mm"] = 1000.0 * float(-min(depth)) if depth else float("nan")
    res["episodes"] = eps
    return res


def main():
    args = parse()
    root = common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/06")
    ck = common.resolve(args.checkpoint) if args.checkpoint else common.list_checkpoints(args.session)[-1][1]
    print("checkpoint:", ck)
    a = args.amplitude
    T = common.terrain_spec
    variants = [
        ("plane (no terrain)", None),
        ("hfield random 60m/768 (cell 7.8cm)", T("random", a, 0, 1.0, 60.0, 768)),
        ("hfield random 60m/240 (cell 25cm)", T("random", a, 0, 1.0, 60.0, 240)),
        ("hfield random 60m/120 (cell 50cm)", T("random", a, 0, 1.0, 60.0, 120)),
        ("hfield random 20m/256 (upstream)", T("random", a, 0, 1.0, 20.0, 256)),
        ("hfield sinus 60m/768", T("sinusoidal", a, 0, 1.0, 60.0, 768)),
    ]
    tasks = [dict(config=args.session, checkpoint=ck, episodes=args.episodes, max_steps=args.max_steps, terrain=t, label=lab) for lab, t in variants]
    print(f"{len(tasks)} variants x {args.episodes} episodes, workers={args.workers}", flush=True)
    rows = common.run_parallel(tasks, args.workers, fn=diag_task, root=root)
    for r in rows:
        r["steps_frac"] = r["mean_steps"] / args.max_steps
    flat = common.strip_episodes(rows)
    common.save_rows(out, "06_variants", flat)
    cols = ["label", "n", "fall_rate", "steps_frac", "mean_distance_m", "mean_contacts", "deepest_penetration_mm"]
    print()
    common.print_table(flat, cols, {c: ".2f" for c in cols[2:]})

    ref, hf = rows[0], rows[1:]
    reasons = [f"plane fall rate {ref['fall_rate']:.2f}, walks {ref['mean_distance_m']:.1f} m"]
    if ref["fall_rate"] > 0.3:
        status = "FAIL"
        reasons.append("the plane reference itself falls: wrong checkpoint/session? (this pilot needs a walking policy)")
    else:
        good = [r for r in hf if r["fall_rate"] <= ref["fall_rate"] + 0.2]
        bad = [r for r in hf if r["fall_rate"] > ref["fall_rate"] + 0.2]
        if not bad:
            status = "PASS"
            reasons.append("every heightfield variant walks like the plane -> the heightfield path is fine; the earlier 03a failures need another explanation")
        elif good:
            status = "WARN"
            reasons.append("walks on: " + ", ".join(r["label"] for r in good))
            reasons.append("falls on: " + ", ".join(r["label"] for r in bad) + " -> grid resolution/extent matters; use a walking variant for terrain")
        else:
            status = "FAIL"
            reasons.append(f"plane walks but EVERY heightfield variant falls even at amplitude {a:g} m (<=0.01 mm): the heightfield contact is an artifact for this gait")
            reasons.append("fix: train Stage 1 (and 2) on a heightfield too (amplitude ~1e-5 up to a small curriculum) and re-run this pilot")
    return common.verdict(out, "06_hfield_diagnosis", status, reasons, {"args": vars(args)})


if __name__ == "__main__":
    sys.exit(main())
