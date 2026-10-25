#!/usr/bin/env python
"""Does the C2 controller USE its memory?  Evaluate trained C2 (GRU) runs with the history window damaged, no training involved.

    python scripts/gru_ablation.py --runs rl_train/results/stage2_P2_gru_s*_nP_* --out gru_ablation_P --workers 288 --episodes 20

Modes (controllers/gru_module.py:ablate_window):  none = the real window | repeat = all K frames replaced by the newest one (no temporal
information, realistic input) | shuffle = the same K frames in a random fixed order (time order destroyed) | zero = window missing.
Each run keeps its own noise and preview settings. If the memory is used, the steps survived DROP when the window is damaged; if the
policy ignores it, they do not change (the GRU branch learned nothing useful, whatever the training curve says).
Output: <out>/<mode>/stage2_eval.csv (same format as eval_stage2.py, last checkpoint, random terrain 6 cm, terrain seeds 1-3) and a table
with the seed-level difference to 'none' (scripts/compare_groups.py).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilots"))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXTENT, RES = 160.0, 640


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="C2 (gru) Stage-2 session folders (globs allowed)")
    ap.add_argument("--modes", default="none,repeat,shuffle,zero")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--amp", type=float, default=0.06)
    ap.add_argument("--terrain-seeds", default="1,2,3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="gru_ablation")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = common.init(a.root)
    out = common.out_dir(a.out, "gru_ablation")
    runs = []
    for pat in a.runs:
        for p in sorted(glob.glob(common.resolve(pat))):
            info = json.load(open(os.path.join(p, "stage2_info.json")))
            if info["args"].get("controller") != "gru":
                print("skip (not a C2 run):", p)
                continue
            runs.append((os.path.basename(p), p, info))
    if not runs:
        raise SystemExit("no C2 runs found")
    seeds = [int(x) for x in a.terrain_seeds.split(",")]
    items = []
    for mode in a.modes.split(","):
        tasks = []
        for label, p, info in runs:
            cks = common.list_checkpoints(p)
            if not cks:
                continue
            steps, ck = cks[-1]
            for ts in seeds:
                spec = common.terrain_spec("random", a.amp, ts, extent=EXTENT, resolution=RES)
                env = dict(info.get("perception_env") or {})
                env["MYO_GRU_ABLATE"] = "" if mode == "none" else mode
                tasks.append(dict(config=os.path.join(p, "session_config.json"), checkpoint=ck, episodes=a.episodes, max_steps=a.max_steps, seed0=0,
                                  terrain=spec, root=root, env=env, label=f"{label}_{mode}_{ts}",
                                  extra=dict(run=label, cond=info["args"]["cond"], controller="gru", seed=info["args"].get("seed", 0), ckpt_steps=steps,
                                             terrain_kind="random", amplitude=a.amp, terrain_seed=ts, ablation=mode)))
        print(f"mode {mode}: {len(tasks)} tasks", flush=True)
        rows = common.strip_episodes(common.run_parallel(tasks, workers=a.workers, root=root))
        d = os.path.join(out, mode)
        os.makedirs(d, exist_ok=True)
        common.save_rows(d, "stage2_eval", rows)
        items.append(f"{mode}={os.path.join(d, 'stage2_eval.csv')}")
    cmd = [sys.executable, os.path.join(HERE, "compare_groups.py"), *items, "--ref", "none", "--metric", "steps", "--amp", str(a.amp),
           "--title", "C2 with a damaged history window: steps survived", "--out", os.path.join(out, "ablation_steps")]
    subprocess.run(cmd, check=False)
    cmd[cmd.index("steps")] = "falls"
    cmd[-1] = os.path.join(out, "ablation_falls")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
