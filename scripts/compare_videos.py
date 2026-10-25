#!/usr/bin/env python
"""Side-by-side walking videos of two trained runs on the SAME terrain and episode seeds (left | right).

    # C1 (left) vs C2 (right), the MEDIAN run of each group (picked from the evaluation csv: no cherry-picking), random terrain 6 cm:
    python scripts/compare_videos.py --left-csv eval_noise/P_mlp/stage2_eval.csv --right-csv eval_noise/P_gru/stage2_eval.csv \
        --cond P2 --results rl_train/results --out videos_c1_c2_P --amp 0.06 --seeds 0,1,2

    # or name the two run folders yourself:
    python scripts/compare_videos.py --left <run dir> --right <run dir> --out videos_x

The run's own noise / preview / window settings are applied automatically (stage2_info.json). Needs ffmpeg or imageio-ffmpeg for the side by side
(otherwise the two single videos are kept in <out>/left and <out>/right). Rendering is slow-ish (about a minute per 600 steps) but needs no training.
"""
import argparse
import os
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from terrain_memory_prosthesis.evaluation.analysis import read_rows  # noqa: E402


def median_run(csv_path, cond, amp):
    rows = [r for r in read_rows(csv_path) if r["terrain_kind"] == "random" and abs(r["amplitude"] - amp) < 1e-9 and (not cond or r["cond"] == cond)]
    last = {}
    for r in rows:
        last[r["run"]] = max(last.get(r["run"], 0), r["ckpt_steps"])
    acc = {}
    for r in rows:
        if r["ckpt_steps"] == last[r["run"]]:
            s, n = acc.get(r["run"], (0.0, 0))
            acc[r["run"]] = (s + float(r["mean_steps"]) * int(r["n"]), n + int(r["n"]))
    vals = sorted((s / n, run) for run, (s, n) in acc.items())
    if not vals:
        raise SystemExit(f"no runs in {csv_path} for cond={cond}")
    pick = vals[len(vals) // 2]
    print(f"{csv_path}: {len(vals)} runs, median = {pick[1]} ({pick[0]:.0f} steps); all: " + ", ".join(f"{v:.0f}" for v, _ in vals))
    return pick[1]


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
    ap.add_argument("--left", default=None)
    ap.add_argument("--right", default=None)
    ap.add_argument("--left-csv", default=None)
    ap.add_argument("--right-csv", default=None)
    ap.add_argument("--cond", default="P2")
    ap.add_argument("--results", default=None, help="folder with the run directories (needed with --left-csv / --right-csv)")
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--terrain", default="random", choices=["flat", "random", "sinusoidal"])
    ap.add_argument("--terrain-seed", type=int, default=1)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--out", default="videos_compare")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    if a.left_csv:
        a.left = os.path.join(a.results, median_run(a.left_csv, a.cond, a.amp))
    if a.right_csv:
        a.right = os.path.join(a.results, median_run(a.right_csv, a.cond, a.amp))
    if not (a.left and a.right):
        raise SystemExit("give --left and --right, or --left-csv and --right-csv with --results")
    out = os.path.abspath(a.out)
    py = sys.executable
    for side, run in (("left", a.left), ("right", a.right)):
        cmd = [py, os.path.join(HERE, "pilots", "05_render_video.py"), "--session", run, "--seeds", a.seeds, "--max-steps", str(a.max_steps),
               "--terrain", a.terrain, "--amplitude", str(a.amp), "--terrain-seed", str(a.terrain_seed), "--out", os.path.join(out, side)]
        if a.root:
            cmd += ["--root", a.root]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    ff = ffmpeg_exe()
    for sd in a.seeds.split(","):
        f1, f2 = (os.path.join(out, s, f"walk_seed{sd}.mp4") for s in ("left", "right"))
        if ff and os.path.exists(f1) and os.path.exists(f2):
            dst = os.path.join(out, f"compare_seed{sd}.mp4")
            r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", f1, "-i", f2, "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                                "-map", "[v]", "-pix_fmt", "yuv420p", dst])
            print("side by side:" if r.returncode == 0 else "side by side FAILED:", dst)
        else:
            print("no ffmpeg or a missing video: the single videos stay in", out)
    open(os.path.join(out, "README.txt"), "w").write(f"left : {a.left}\nright: {a.right}\nterrain {a.terrain} {a.amp} m, terrain seed {a.terrain_seed}, episode seeds {a.seeds}\n")


if __name__ == "__main__":
    main()
