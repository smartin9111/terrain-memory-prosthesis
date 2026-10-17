#!/usr/bin/env python
"""03b - Does terrain preview (P2) help the prosthesis controller at all?  (short go/no-go, NOT the experiment)

Question : With the Stage-1 human frozen, does a prosthesis actor that sees the terrain ahead (P2) end up measurably
           better on rough terrain than one that sees proprioception only (P1) - after the same short PPO budget?
Setup    : * env  myoAssistLegImitationExoPreview-v0 (src/terrain_memory_prosthesis/environment/preview_env.py): obs[40:45] = ground height at
             0.2,0.4,0.6,0.8,1.0 m ahead relative to the pelvis, in units of 10 cm.
           * Stage 1 human (--stage1 checkpoint) is copied into both conditions and frozen (controllers/freeze.py);
             the human actor never sees the preview. Only the prosthesis actor is trained.
           * P1: prosthesis actor input = proprioception.  P2: the same + obs[40:45].  The critic sees obs[0:45] in BOTH
             (privileged critic, identical across conditions) so the value baseline cannot explain a difference.
           * Same PPO hyper-parameters, `--seeds` training seeds per condition, trained on ONE random terrain
             (--train-terrain-seed); evaluation = deterministic, fixed episode seeds, on HELD-OUT terrains
             (--eval-terrain-seeds) + the training terrain.
Verdict  : PASS  P2 better than P1 by >= --min-effect (survived-step fraction) and by more than the between-seed spread
           WARN  inconclusive (no clear difference / too few seeds / floor-or-ceiling effect) - this is the usual outcome
                 of a short run; it is NOT evidence against P2
           FAIL  stage 1 missing, or all runs identical/degenerate (pipeline broken)
Honest limits: tiny budget, few seeds, one training terrain; an effect here is a signal that the real experiment is
           worth its compute, no effect means "not visible at this budget".
Exit code: 0 PASS, 1 WARN, 2 FAIL.

Example (plumbing smoke, minutes):
    python 03b_p1_vs_p2_short_train.py --config p1_c1_osl.json --stage1 <ckpt.zip> --smoke
Example (real pilot, ~hours; see README for compute):
    python 03b_p1_vs_p2_short_train.py --config p1_c1_osl.json --stage1 <ckpt.zip> --timesteps 3000000 --seeds 3 \
        --num-envs 16 --parallel 2 --amplitude 0.03
"""
import argparse
import json
import os
import subprocess
import sys
import time

# --- registration must also happen when SB3 forkserver/spawn workers import this file as __main__ (see environment/preview_env.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--root", default=None)
_known, _ = _pre.parse_known_args()
common.init(_known.root)
from terrain_memory_prosthesis.environment import preview_env as terrain_preview_env  # noqa: E402,F401  (registers the env id at import)

N_MUSCLE = 18


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="stage-1 style config json (p1_c1_osl.json) or a session folder")
    ap.add_argument("--stage1", help="stage-1 checkpoint .zip (human to freeze). Required unless --no-freeze")
    ap.add_argument("--no-freeze", action="store_true", help="train human too from scratch (only meaningful for plumbing smoke)")
    ap.add_argument("--conditions", default="P1,P2")
    ap.add_argument("--seeds", type=int, default=3, help="training seeds per condition")
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=1, help="training runs at the same time (cores needed = parallel x num-envs)")
    ap.add_argument("--terrain", default="random", choices=["random", "sinusoidal"])
    ap.add_argument("--amplitude", type=float, default=0.03, help="terrain amplitude [m] (take from 03a)")
    ap.add_argument("--train-terrain-seed", type=int, default=100)
    ap.add_argument("--eval-terrain-seeds", default="1,2,3")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--min-effect", type=float, default=0.10, help="min P2-P1 gain in survived-step fraction to call it a signal")
    ap.add_argument("--smoke", action="store_true", help="tiny settings: plumbing only, meaningless numbers")
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.smoke:
        args.timesteps, args.num_envs, args.n_steps, args.batch_size, args.n_epochs = 512, 2, 128, 128, 2
        args.seeds, args.eval_episodes, args.max_steps, args.eval_terrain_seeds = 1, 2, 60, "1"
    return args


# --------------------------------------------------------------------------- one training run (worker)
def make_config_dict(base: dict, cond: str, args_d: dict) -> dict:
    terrain = common.terrain_spec(args_d["terrain"], args_d["amplitude"], args_d["train_terrain_seed"])
    return common.make_session_config(base, cond, terrain)


def run_worker(job: dict) -> int:
    common.init(job.get("root"))
    import numpy as np
    import torch
    from terrain_memory_prosthesis.controllers.freeze import load_and_freeze_human

    run_dir = job["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    base = common.read_config_dict(job["config"])
    d = make_config_dict(base, job["cond"], job["args"])
    cfg = common.config_from_dict(d, run_dir, "session_config")
    cfg.env_params.seed = int(job["seed"])
    a = job["args"]
    t0 = time.time()
    model, env = common.build_model(cfg, num_envs=a["num_envs"], n_steps=a["n_steps"], batch_size=a["batch_size"], n_epochs=a["n_epochs"])
    model.set_random_seed(int(job["seed"]))
    if job.get("stage1"):
        stage1 = common.load_model(job["stage1"])
        load_and_freeze_human(model, stage1, n_muscle=N_MUSCLE)
        frozen = True
    else:
        frozen = False
    model.learn(total_timesteps=a["timesteps"])
    ck = os.path.join(run_dir, "model_final.zip")
    model.save(ck)
    train_s = time.time() - t0
    env.close()

    # ---- evaluation: held-out terrains + train terrain, deterministic, same episode seeds in every run
    ev_terrains = [("heldout", int(s)) for s in a["eval_terrain_seeds"].split(",")] + [("train", int(a["train_terrain_seed"]))]
    results = []
    for kind, ts in ev_terrains:
        r = common.eval_task(
            dict(
                config=os.path.join(run_dir, "session_config.json"),
                checkpoint=ck,
                episodes=a["eval_episodes"],
                max_steps=a["max_steps"],
                seed0=0,
                terrain=common.terrain_spec(a["terrain"], a["amplitude"], ts),
                root=job.get("root"),
                label=f"{job['cond']}_s{job['seed']}_{kind}{ts}",
                extra={"cond": job["cond"], "seed": job["seed"], "terrain_kind": kind, "terrain_seed": ts},
            )
        )
        r["steps_frac"] = r["mean_steps"] / a["max_steps"]
        results.append(r)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump({"cond": job["cond"], "seed": job["seed"], "frozen": frozen, "train_s": train_s, "timesteps": a["timesteps"],
                   "fps": a["timesteps"] / max(train_s, 1e-9), "evals": common.strip_episodes(results)}, f, indent=2)
    print(f"[worker {job['cond']} s{job['seed']}] done: train {train_s/60:.1f} min")
    return 0


# --------------------------------------------------------------------------- orchestration
def main():
    args = parse()
    if args._worker:
        return run_worker(json.load(open(args._worker)))
    root = common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/03b")
    if not args.config:
        raise SystemExit("--config is required")
    if not args.stage1 and not args.no_freeze:
        raise SystemExit("--stage1 <checkpoint.zip> is required (the frozen human). Use --no-freeze only for a plumbing smoke test.")
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    args_d = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    jobs = []
    for c in conds:
        for s in range(args.seeds):
            rd = os.path.join(out, "runs", f"{c}_s{s}")
            job = dict(cond=c, seed=s, run_dir=rd, config=common.resolve(args.config), stage1=common.resolve(args.stage1) if args.stage1 else None, args=args_d, root=root)
            os.makedirs(rd, exist_ok=True)
            jp = os.path.join(rd, "job.json")
            json.dump(job, open(jp, "w"))
            jobs.append((c, s, jp, rd))
    print(f"{len(jobs)} training runs ({conds} x {args.seeds} seeds), {args.parallel} at a time, "
          f"{args.num_envs} envs each, {args.timesteps} steps each", flush=True)

    running = []
    pending = [j for j in jobs if not os.path.exists(os.path.join(j[3], "result.json"))]  # resumable
    done_before = len(jobs) - len(pending)
    if done_before:
        print(f"(resuming: {done_before} runs already have result.json)")
    failed = []
    while pending or running:
        while pending and len(running) < args.parallel:
            c, s, jp, rd = pending.pop(0)
            log = open(os.path.join(rd, "log.txt"), "w")
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_worker", jp, "--root", root], stdout=log, stderr=subprocess.STDOUT)
            running.append((p, c, s, rd, log, time.time()))
            print(f"  started {c} seed {s}", flush=True)
        time.sleep(2)
        for item in list(running):
            p, c, s, rd, log, t0 = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                ok = p.returncode == 0 and os.path.exists(os.path.join(rd, "result.json"))
                print(f"  finished {c} seed {s} in {(time.time()-t0)/60:.1f} min -> {'ok' if ok else 'FAILED (see ' + rd + '/log.txt)'}", flush=True)
                if not ok:
                    failed.append((c, s))

    # ---- stage-1 reference (same terrains), if possible
    rows = []
    for c, s, jp, rd in jobs:
        rp = os.path.join(rd, "result.json")
        if os.path.exists(rp):
            res = json.load(open(rp))
            for e in res["evals"]:
                e.update(cond=res["cond"], seed=res["seed"])
                rows.append(e)
    if not rows:
        return common.verdict(out, "03b_p1_vs_p2", "FAIL", ["no run produced results; see runs/*/log.txt"])
    common.save_rows(out, "03b_runs", rows)

    import statistics as st

    def per_run(cond, kind):
        return [r for r in rows if r["cond"] == cond and r["terrain_kind"] == kind]

    table = []
    for c in conds:
        for kind in ("heldout", "train"):
            rr = per_run(c, kind)
            if not rr:
                continue
            # average over terrains first, then look at spread across training seeds
            by_seed = {}
            for r in rr:
                by_seed.setdefault(r["seed"], []).append(r)
            sf = [sum(x["steps_frac"] for x in v) / len(v) for v in by_seed.values()]
            fr = [sum(x["fall_rate"] for x in v) / len(v) for v in by_seed.values()]
            ds = [sum(x["mean_distance_m"] for x in v) / len(v) for v in by_seed.values()]
            table.append(dict(cond=c, terrain=kind, seeds=len(sf), steps_frac_mean=st.mean(sf), steps_frac_sd=(st.stdev(sf) if len(sf) > 1 else float("nan")),
                              fall_rate_mean=st.mean(fr), distance_mean=st.mean(ds), per_seed_steps_frac=",".join(f"{x:.2f}" for x in sf)))
    common.save_rows(out, "03b_summary", table)
    cols = ["cond", "terrain", "seeds", "steps_frac_mean", "steps_frac_sd", "fall_rate_mean", "distance_mean", "per_seed_steps_frac"]
    print()
    common.print_table(table, cols, {c: ".3f" for c in cols[3:7]})

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, c in enumerate(conds):
            vals = {}
            for r in per_run(c, "heldout"):
                vals.setdefault(r["seed"], []).append(r["steps_frac"])
            ys = [sum(v) / len(v) for v in vals.values()]
            ax.scatter([i] * len(ys), ys, alpha=0.7)
            if ys:
                ax.hlines(sum(ys) / len(ys), i - 0.2, i + 0.2, colors="k")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_ylabel("survived fraction (held-out terrains)")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "03b_p1_vs_p2.png"), dpi=120)
    except Exception as e:
        print("(plot skipped)", e)

    # ---- verdict
    reasons = []
    t = {(r["cond"], r["terrain"]): r for r in table}
    if failed:
        reasons.append(f"{len(failed)} run(s) failed: {failed}")
    if args.smoke:
        reasons.append("SMOKE mode: plumbing only, numbers are meaningless")
        return common.verdict(out, "03b_p1_vs_p2", "WARN", reasons)
    if ("P1", "heldout") not in t or ("P2", "heldout") not in t:
        return common.verdict(out, "03b_p1_vs_p2", "FAIL", reasons + ["need both P1 and P2 results"])
    p1, p2 = t[("P1", "heldout")], t[("P2", "heldout")]
    eff = p2["steps_frac_mean"] - p1["steps_frac_mean"]
    spread = max(x for x in (p1["steps_frac_sd"], p2["steps_frac_sd"], 0.0) if x == x)
    reasons.append(f"held-out terrains, survived fraction: P1 {p1['steps_frac_mean']:.2f} (sd {p1['steps_frac_sd']:.2f}), "
                   f"P2 {p2['steps_frac_mean']:.2f} (sd {p2['steps_frac_sd']:.2f}); effect P2-P1 = {eff:+.2f}; "
                   f"fall rate P1 {p1['fall_rate_mean']:.2f} vs P2 {p2['fall_rate_mean']:.2f}")
    if max(p1["steps_frac_mean"], p2["steps_frac_mean"]) < 0.15:
        status = "FAIL"
        reasons.append("both conditions barely walk: the frozen human cannot cope with this terrain/amplitude (lower the amplitude, check 00/03a) or training diverged")
    elif min(p1["steps_frac_mean"], p2["steps_frac_mean"]) > 0.95 and p1["fall_rate_mean"] < 0.05 and p2["fall_rate_mean"] < 0.05:
        status = "WARN"
        reasons.append("ceiling: both walk the whole episode -> terrain too easy, no headroom (raise amplitude)")
    elif eff >= args.min_effect and eff > spread and args.seeds >= 2:
        status = "PASS"
        reasons.append("P2 clearly ahead of P1 and larger than the seed spread -> real experiment looks worth its compute")
    else:
        status = "WARN"
        reasons.append("no clear difference (or too few seeds/too short): inconclusive, NOT evidence against P2. "
                       "Options: more timesteps, more seeds, larger amplitude where preview is useful, check the preview signal")
    reasons.append(f"mean training speed ~{st.mean([json.load(open(os.path.join(j[3],'result.json')))['fps'] for j in jobs if os.path.exists(os.path.join(j[3],'result.json'))]):.0f} steps/s per run (use it to plan compute)")
    return common.verdict(out, "03b_p1_vs_p2", status, reasons, {"effect": eff, "spread": spread})


if __name__ == "__main__":
    sys.exit(main())
