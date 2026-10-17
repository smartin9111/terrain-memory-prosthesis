#!/usr/bin/env python
"""One command: train P1xC1 and P2xC1 (Stage 2), evaluate both on held-out terrains, make figures + tables + videos.

    python scripts/compare_p1_p2.py --stage1-session ../stage1_session --timesteps 2000000 --out compare_out

Steps (each can be skipped / reused, see the options):
  1. train   scripts/train_stage2.py --cond P1 and --cond P2 (same seed, same recipe, one after the other; all cores each)
  2. eval    scripts/eval_stage2.py: both runs + the Stage-1 policy as reference on flat / random 2,4,6,8 cm / sinusoidal terrains
  3. report  scripts/analyze_stage2.py: fig_fall_rate, fig_metrics, fig_training (+ fig_checkpoints with --ckpts > 1), summary.md, differences.md
  4. videos  scripts/pilots/05_render_video.py: P1 and P2 on the SAME terrain and seeds, side by side if ffmpeg is available
Everything lands in --out:  logs/  eval/  report/  videos/  compare_info.json.
Time: dominated by the training (steps/s is printed); 'python scripts/compare_p1_p2.py ... --smoke' checks the whole chain in a few minutes.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KEEP = ("Session directory", "equivalent to Stage 1", "total_timesteps", "fps", "stage 2 done", "Error", "Traceback")


def pick_num_envs(cpus: int) -> int:
    """Largest power of two <= cpus (must divide the 16384-step rollout)."""
    n = 1
    while n * 2 <= max(cpus, 1):
        n *= 2
    return n


def run(cmd, log_path=None, show=KEEP, env=None):
    """Run a command; stream the interesting lines, keep everything in the log file. Raises on a non-zero exit."""
    print("$", " ".join(cmd), flush=True)
    log = open(log_path, "w") if log_path else None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    session = None
    for line in p.stdout:
        if log:
            log.write(line)
        if "Session directory:" in line:
            session = line.split("Session directory:")[1].strip()
        if show is None or any(k in line for k in show):
            print("   ", line.rstrip(), flush=True)
    code = p.wait()
    if log:
        log.close()
    if code:
        raise SystemExit(f"command failed ({code}); log: {log_path}")
    return session


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1-session", required=True, help="folder with session_config.json + trained_models/ of the final Stage 1")
    ap.add_argument("--out", default="compare_out")
    ap.add_argument("--controller", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=2_000_000, help="Stage-2 steps PER RUN")
    ap.add_argument("--num-envs", type=int, default=None, help="default: largest power of two <= CPU count (must divide 16384)")
    ap.add_argument("--parallel-train", action="store_true",
                    help="train P1 and P2 AT THE SAME TIME (for big machines): --num-envs is then per run, default = largest power of two <= CPUs/2")
    ap.add_argument("--amp-lo", type=float, default=0.02)
    ap.add_argument("--amp-hi", type=float, default=0.06)
    ap.add_argument("--episodes", type=int, default=10, help="evaluation episodes per terrain instance")
    ap.add_argument("--workers", type=int, default=None, help="evaluation processes (default: CPU count - 1)")
    ap.add_argument("--ckpts", type=int, default=1, help=">1: evaluate that many checkpoints per run too (learning curve of the falls)")
    ap.add_argument("--video-amp", type=float, default=0.04, help="random terrain amplitude of the comparison videos [m]")
    ap.add_argument("--video-seeds", default="0,1")
    ap.add_argument("--video-steps", type=int, default=600)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="reuse --p1-run / --p2-run")
    ap.add_argument("--p1-run", default=None, help="existing P1 Stage-2 session folder (with --skip-train)")
    ap.add_argument("--p2-run", default=None)
    ap.add_argument("--steps", default="train,eval,report,video",
                    help="comma list of the stages to run (default all). E.g. several seeds in parallel: --steps train for each seed, "
                         "then eval_stage2.py + analyze_stage2.py once over all runs (docs/stage2.md)")
    ap.add_argument("--smoke", action="store_true", help="plumbing check: tiny training, 2 episodes, short videos")
    ap.add_argument("--root", default=None, help="MyoAssist checkout (default: auto)")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from terrain_memory_prosthesis.environment import myoassist as ma

    cpus = os.cpu_count() or 4
    n_envs = a.num_envs or pick_num_envs(cpus // 2 if a.parallel_train else cpus)
    workers = a.workers or max(1, cpus - 1)
    out = os.path.abspath(a.out)
    s1 = os.path.abspath(a.stage1_session)
    root = ma.init(a.root or os.environ.get("MYOASSIST_DIR"), threads=int(os.environ.get("MYOASSIST_NUM_THREADS", "8")))
    for d in ("logs", "eval", "report", "videos"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    py = sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(HERE, "..", "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MYOASSIST_DIR"] = root
    t0 = time.time()

    steps = set(a.steps.split(","))
    runs = {}
    if a.skip_train or "train" not in steps:
        runs = {"P1": os.path.abspath(a.p1_run), "P2": os.path.abspath(a.p2_run)}
    elif a.parallel_train:
        print(f"\n=== 1/4 training P1 and P2 in parallel ({a.timesteps:,} steps each, {n_envs} envs each) ===", flush=True)
        procs = {}
        for cond in ("P1", "P2"):
            cmd = [py, os.path.join(HERE, "train_stage2.py"), "--stage1-session", s1, "--cond", cond, "--controller", a.controller,
                   "--seed", str(a.seed), "--timesteps", str(a.timesteps), "--num-envs", str(n_envs), "--amp-lo", str(a.amp_lo),
                   "--amp-hi", str(a.amp_hi), "--tag", "cmp", "--root", root] + (["--smoke"] if a.smoke else [])
            logp = os.path.join(out, "logs", f"train_{cond}.log")
            print("$", " ".join(cmd), flush=True)
            procs[cond] = (subprocess.Popen(cmd, stdout=open(logp, "w"), stderr=subprocess.STDOUT, text=True, env=env), logp)
            time.sleep(2)  # distinct session-directory time stamps
        while any(p.poll() is None for p, _ in procs.values()):
            time.sleep(60)
            for cond, (p, logp) in procs.items():
                steps, fps = "?", "?"
                for line in open(logp, errors="ignore"):
                    if "total_timesteps" in line:
                        steps = line.split("|")[2].strip()
                    elif "fps" in line:
                        fps = line.split("|")[2].strip()
                print(f"    {cond}: {steps} / {a.timesteps:,} steps, {fps} steps/s ({'running' if p.poll() is None else 'finished'})", flush=True)
        for cond, (p, logp) in procs.items():
            if p.returncode:
                raise SystemExit(f"training {cond} failed ({p.returncode}); log: {logp}")
            sess = [l.split("Session directory:")[1].strip() for l in open(logp, errors="ignore") if "Session directory:" in l][0]
            runs[cond] = sess if os.path.isabs(sess) else os.path.join(root, sess)
            print(f"    {cond} -> {runs[cond]}", flush=True)
    else:
        for cond in ("P1", "P2"):
            print(f"\n=== 1/4 training {cond}x{a.controller} ({a.timesteps:,} steps, {n_envs} envs) ===", flush=True)
            cmd = [py, os.path.join(HERE, "train_stage2.py"), "--stage1-session", s1, "--cond", cond, "--controller", a.controller,
                   "--seed", str(a.seed), "--timesteps", str(a.timesteps), "--num-envs", str(n_envs), "--amp-lo", str(a.amp_lo),
                   "--amp-hi", str(a.amp_hi), "--tag", "cmp", "--root", root]
            if a.smoke:
                cmd.append("--smoke")
            sess = run(cmd, os.path.join(out, "logs", f"train_{cond}.log"), env=env)
            runs[cond] = os.path.join(root, sess) if not os.path.isabs(sess) else sess
            print(f"    -> {runs[cond]}", flush=True)

    if "eval" in steps:
        print("\n=== 2/4 evaluation (held-out terrains) ===", flush=True)
        cmd = [py, os.path.join(HERE, "eval_stage2.py"), "--runs", runs["P1"], runs["P2"], "--stage1-session", s1,
               "--episodes", str(2 if a.smoke else a.episodes), "--max-steps", str(80 if a.smoke else 1000),
               "--workers", str(workers), "--ckpts", str(a.ckpts), "--out", os.path.join(out, "eval"), "--root", root]
        run(cmd, os.path.join(out, "logs", "eval.log"), show=("evaluation tasks", "fall_rate", "stage1", "P1", "P2", "Error", "Traceback"), env=env)

    if "report" in steps:
        print("\n=== 3/4 report (figures + tables) ===", flush=True)
        cmd = [py, os.path.join(HERE, "analyze_stage2.py"), "--eval", os.path.join(out, "eval"), "--runs", runs["P1"], runs["P2"],
               "--out", os.path.join(out, "report")]
        run(cmd, os.path.join(out, "logs", "analyze.log"), show=None, env=env)

    if "video" in steps and not a.no_video:
        print("\n=== 4/4 videos (same terrain, same seeds for P1 and P2) ===", flush=True)
        ff = ffmpeg_exe()
        seeds = a.video_seeds
        for tname, targs in (("flat", ["--terrain", "flat"]),
                             (f"random{a.video_amp * 100:g}cm", ["--terrain", "random", "--amplitude", str(a.video_amp), "--terrain-seed", "1"])):
            dirs = {}
            for cond in ("P1", "P2"):
                d = os.path.join(out, "videos", f"{cond}_{tname}")
                dirs[cond] = d
                run([py, os.path.join(HERE, "pilots", "05_render_video.py"), "--session", runs[cond], "--seeds", seeds,
                     "--max-steps", str(60 if a.smoke else a.video_steps), "--out", d, "--root", root] + targs,
                    os.path.join(out, "logs", f"video_{cond}_{tname}.log"), show=("saved", "fell=", "Error", "Traceback"), env=env)
            for sd in seeds.split(","):
                f1, f2 = (os.path.join(dirs[c], f"walk_seed{sd}.mp4") for c in ("P1", "P2"))
                if ff and os.path.exists(f1) and os.path.exists(f2):
                    dst = os.path.join(out, "videos", f"compare_{tname}_seed{sd}.mp4")
                    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", f1, "-i", f2, "-filter_complex",
                                        "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-pix_fmt", "yuv420p", dst])
                    print("    side by side:" if r.returncode == 0 else "    side-by-side failed:", dst)
                elif not ff:
                    print("    (no ffmpeg: the P1 and P2 videos are in separate folders, left = P1, right = P2 not combined)")
                    break

    info = dict(args=vars(a), runs=runs, num_envs=n_envs, eval_workers=workers, minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(out, "compare_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDONE in {info['minutes']} min. Open: {os.path.join(out, 'report')} (fig_fall_rate.png, summary.md) and {os.path.join(out, 'videos')}")


if __name__ == "__main__":
    main()
