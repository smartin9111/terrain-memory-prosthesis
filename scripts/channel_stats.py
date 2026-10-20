#!/usr/bin/env python
"""Per-channel statistics of the 40 clean proprioceptive channels under the Stage-1 policy -> scale for the sensor noise.

    python scripts/channel_stats.py --stage1-session rl_train/results/train_session_<stamp> --episodes 6 --out channel_stats

Runs the (frozen, deterministic) Stage-1 policy on flat terrain and collects obs[0:40] (the clean block) at every control step.
Writes <out>/channel_std.json = {"std": [40 numbers], "mean": [...], "n": steps}. A relative sensor noise alpha is then
    MYO_SENSOR_NOISE_STD='[alpha * std_0, ..., alpha * std_39]'   (see scripts/noise_env.py)
so every channel gets noise in proportion to how much it varies in normal walking (joint angle in rad, velocity in rad/s, ... are not
comparable with one scalar). Channels that hardly vary get a floor (--floor, default 1e-3) so they are not noise-free by accident.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilots"))
import common  # noqa: E402

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1-session", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--checkpoint-steps", type=int, default=None)
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--out", default="channel_stats")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    common.init(a.root)
    sess = common.resolve(a.stage1_session)
    cks = common.list_checkpoints(sess)
    if not cks:
        raise SystemExit(f"no checkpoints in {sess}")
    ck = min(cks, key=lambda c: abs(c[0] - a.checkpoint_steps))[1] if a.checkpoint_steps else cks[-1][1]
    cfg = common.load_config(os.path.join(sess, "session_config.json"))
    env = common.make_eval_env(cfg, a.max_steps, None)
    model = common.load_model(ck)

    rows = []
    for ep in range(a.episodes):
        with common.seeded_rng(ep):
            obs, _ = env.reset()
        for _ in range(a.max_steps):
            rows.append(np.asarray(obs[:40], dtype=float))
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _i = env.step(act)
            if terminated or truncated:
                break
    env.close()
    x = np.stack(rows)
    std = np.maximum(x.std(axis=0), a.floor)
    out = common.out_dir(a.out, "channel_stats")
    with open(os.path.join(out, "channel_std.json"), "w") as f:
        json.dump({"std": std.tolist(), "mean": x.mean(axis=0).tolist(), "n": int(len(x)), "checkpoint": ck}, f)
    print(f"{len(x)} steps from {a.episodes} episodes, checkpoint {ck}")
    print("std per channel:", np.array2string(std, precision=3, max_line_width=120))
    print("written", os.path.join(out, "channel_std.json"))


if __name__ == "__main__":
    main()
