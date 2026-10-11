#!/usr/bin/env python
"""05 - Render a trained Stage-1/2 policy walking: MP4 (or GIF) + a short trace plot.

Runs ONE deterministic episode per requested seed (no imitation early-termination, same as 00) and renders a
side view that follows the pelvis. Headless on Linux/UCloud via EGL (MUJOCO_GL=egl is set by common.init);
if EGL is missing try `MUJOCO_GL=osmesa`.

Outputs in --out (default pilot_out/05):  walk_seed<N>.mp4 (or .gif if no ffmpeg), trace_seed<N>.png

Example (UCloud):
    python scripts/pilots/05_render_video.py --session /work/ai3_test_train/myoassist/rl_train/results/train_session_X \
        --seeds 0,1,2 --max-steps 600
    # a specific checkpoint and rough terrain (5 mm random):
    python scripts/pilots/05_render_video.py --session ... --checkpoint <path/model_29999104.zip> --terrain random --amplitude 0.005
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--checkpoint", default=None, help="default: latest checkpoint of the session")
    ap.add_argument("--seeds", default="0", help="comma separated; one video per seed")
    ap.add_argument("--max-steps", type=int, default=600, help="control steps (30 Hz) -> 600 = 20 s")
    ap.add_argument("--terrain", default="flat", choices=["flat", "random", "sinusoidal"])
    ap.add_argument("--amplitude", type=float, default=0.005)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--distance", type=float, default=3.2, help="camera distance [m]")
    ap.add_argument("--target-velocity", type=float, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def sim_ptrs(env):
    """Raw mujoco MjModel/MjData from a MyoSuite env (wrappers expose them as .ptr in some versions)."""
    m, d = env.sim.model, env.sim.data
    return getattr(m, "ptr", m), getattr(d, "ptr", d)


def main():
    args = parse()
    root = common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/05")
    import numpy as np

    cfg = common.load_config(args.session)
    if args.terrain != "flat":
        cfg.env_params.terrain = common.terrain_spec(args.terrain, args.amplitude, args.terrain_seed)
    ck = common.resolve(args.checkpoint) if args.checkpoint else common.list_checkpoints(args.session)[-1][1]
    print("checkpoint:", ck)
    env = common.make_eval_env(cfg, args.max_steps, args.target_velocity)
    model = common.load_model(ck)

    import mujoco

    mj_model, mj_data = sim_ptrs(env)
    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, args.distance  # side view, looking along +y

    fps = int(round(1.0 / env.dt))
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        with common.seeded_rng(seed):
            obs, _ = env.reset()
        frames, xs, hs, cmd = [], [], [], []
        jn = {"prosthesis ankle (r)": "OSL_A_L1_osl_ankle_angle_r", "human ankle (l)": "ankle_angle_l", "knee r": "knee_angle_r", "knee l": "knee_angle_l"}
        jn = {k: v for k, v in jn.items() if _has(env, v)}
        ang = {k: [] for k in jn}
        dev = common.device_actuator_ids(env)
        fell = False
        for t in range(args.max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _ = env.step(a)
            x = float(env.sim.data.joint("pelvis_tx").qpos[0])
            h = float(env.sim.data.joint("pelvis_ty").qpos[0]) if _has(env, "pelvis_ty") else float("nan")
            xs.append(x)
            hs.append(h)
            for k, v in jn.items():
                ang[k].append(float(np.degrees(env.sim.data.joint(v).qpos[0])))
            cmd.append(float(np.mean(np.asarray(a)[dev])) if dev else 0.0)
            cam.lookat[:] = [x, 0.0, 0.9]
            renderer.update_scene(mj_data, camera=cam)
            frames.append(renderer.render().copy())
            if terminated:
                fell = True
                break
            if truncated:
                break
        print(f"seed {seed}: {len(frames)} steps ({len(frames) / fps:.1f} s), fell={fell}, distance={xs[-1] - xs[0]:.2f} m")
        pa = ang.get("prosthesis ankle (r)")
        if pa:
            j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "OSL_A_L1_osl_ankle_angle_r")
            lim = float(np.degrees(mj_model.jnt_range[j][1]))
            over = float(np.mean(np.abs(pa) > lim + 2.0))
            print(
                f"  prosthesis ankle: min {min(pa):.0f} / max {max(pa):.0f} deg (joint limit +-{lim:.0f}); "
                f"{over:.0%} of steps beyond the limit by >2 deg  -> {'LIMIT NOT HOLDING' if over > 0.05 else 'ok'}"
            )
        path = save_video(out, f"walk_seed{seed}", frames, fps)
        print("saved", path)
        save_trace(out, seed, xs, hs, cmd, env.dt, fell, ang)
    renderer.close()
    env.close()
    return 0


def _has(env, name):
    try:
        env.sim.data.joint(name)
        return True
    except Exception:
        return False


def save_video(out, stem, frames, fps):
    import imageio

    try:
        p = os.path.join(out, stem + ".mp4")
        imageio.mimsave(p, frames, fps=fps, macro_block_size=1)
        return p
    except Exception as e:  # no ffmpeg -> GIF via pillow
        print("(mp4 failed:", e, "-> writing GIF)")
        p = os.path.join(out, stem + ".gif")
        imageio.mimsave(p, frames[::2], duration=2.0 / fps, loop=0)
        return p


def save_trace(out, seed, xs, hs, cmd, dt, fell, ang=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = [i * dt for i in range(len(xs))]
        fig, ax = plt.subplots(1, 4, figsize=(17, 3.2))
        ax[0].plot(t, [x - xs[0] for x in xs])
        ax[0].set_title("distance walked [m]")
        ax[1].plot(t, hs)
        ax[1].set_title("pelvis height [m]" + ("  (FELL)" if fell else ""))
        ax[2].plot(t, cmd)
        ax[2].set_title("mean prosthesis command")
        for k, v in (ang or {}).items():
            ax[3].plot(t, v, label=k)
        ax[3].set_title("joint angles [deg]")
        ax[3].legend(fontsize=7)
        for a in ax:
            a.set_xlabel("time [s]")
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"trace_seed{seed}.png"), dpi=110)
        plt.close(fig)
    except Exception as e:
        print("(trace plot skipped)", e)


if __name__ == "__main__":
    sys.exit(main())
