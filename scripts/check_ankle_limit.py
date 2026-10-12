#!/usr/bin/env python
"""Physics check of the prosthesis ankle limit: stock env (soft limit) vs stiff env (hard limit). No training needed for test A.

TEST A - gravity OFF, isolated joint-limit stress test (pure physics, ~1 min)
    Full prosthesis torque (+1 / -1 pulses, 168 Nm) for --seconds (default 8 s). PASS if the ankle never goes beyond limit + 3 deg.
    (Gravity is off so the muscle-less human does not collapse; only the foot/limit matters.)  -> also writes
    pilot_out/ankle_check/ankle_<stock|stiff>.mp4 (camera on the prosthesis foot).

TEST B - gravity ON, normal dynamics (needs --session: a trained policy walks; its checkpoint is evaluated in BOTH env variants)
    PASS if, over the episodes, there is
      * no solver instability: no NaN/inf, no new MuJoCo warnings (BADQACC, BADQVEL, BADQPOS, BADCTRL, INERTIA, ...),
      * no unrealistic jerking: peak ankle speed <= --max-speed rad/s (default 40) at the control-step samples,
      * no hard-stop hammering: ankle at the limit (within 1 deg) in <= 25 % of the steps, no stint at the limit
        longer than 0.5 s, and <= 4 separate limit hits per second.
    NOTE: a policy trained in the stock env may legitimately use the old soft limit; evaluate the stiff-trained policy
    for the real verdict. Falling episodes are cut at the fall (only walking steps are scored).

    python scripts/check_ankle_limit.py                              # test A only
    python scripts/check_ankle_limit.py --session <session dir>      # tests A + B  [--episodes 5 --max-steps 300]
Exit code 0 if the PHYS env (the current one) passes everything that was run, 1 otherwise. Variants: stock (soft limit), stiff (hard limit),
phys (hard limit + OSL motor model).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pilots"))
import common  # noqa: E402

common.init(None)
import numpy as np  # noqa: E402

CFGS = {
    "stock": os.path.join(HERE, "..", "configs", "experiments", "stage1_flat_osl.json"),
    "stiff": os.path.join(HERE, "..", "configs", "experiments", "stage1_flat_osl_stiff.json"),
    "phys": os.path.join(HERE, "..", "configs", "experiments", "stage1_flat_osl_phys.json"),
}
ENV_IDS = {"stock": "myoAssistLegImitationExo-v0", "stiff": "myoAssistLegImitationExoStiff-v0", "phys": "myoAssistLegImitationExoOSLPhys-v0"}
OUT = os.path.join(common.ORIG_CWD, "pilot_out", "ankle_check")
J = "OSL_A_L1_osl_ankle_angle_r"
WARN_NAMES = ["INERTIA", "CONTACTFULL", "CNSTRFULL", "VGEOMFULL", "BADQPOS", "BADQVEL", "BADQACC", "BADCTRL"]


def raw(env):
    import mujoco  # noqa: F401

    m, d = env.sim.model, env.sim.data
    return getattr(m, "ptr", m), getattr(d, "ptr", d)


def warn_counts(d):
    import mujoco

    return {n: int(d.warning[getattr(mujoco.mjtWarning, "mjWARN_" + n)].number) for n in WARN_NAMES}


def limit_deg(env):
    import mujoco

    m, _ = raw(env)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, J)
    return float(np.degrees(m.jnt_range[j][1]))


def ankle(env):
    return float(np.degrees(env.sim.data.joint(J).qpos[0])), float(env.sim.data.joint(J).qvel[0])


def test_a(name, args):
    import imageio
    import mujoco

    cfg = common.load_config(os.path.abspath(CFGS[name]))
    env = common.make_eval_env(cfg, 400)
    m, d = raw(env)
    lim = limit_deg(env)
    dev = common.device_actuator_ids(env)
    env.reset()
    m.opt.gravity[:] = 0.0
    body = next(i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower().endswith("osl_foot_assembly"))
    frames, th, om = [], [], []
    r = None
    try:
        r = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = 90.0, -5.0, 1.3
    except Exception as e:
        print(f"  ({name}: no renderer, video skipped: {e})")
    for t in range(int(round(args.seconds / env.dt))):
        a = np.zeros(env.action_space.shape)
        a[dev] = 1.0 if (t // 20) % 2 == 0 else -1.0
        env.step(a)
        _a, _w = ankle(env)
        th.append(_a)
        om.append(abs(_w))
        if r is not None:
            cam.lookat[:] = d.xipos[body]
            r.update_scene(d, camera=cam)
            frames.append(r.render().copy())
    if r is not None:
        r.close()
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f"ankle_{name}.mp4")
        imageio.mimsave(p, frames, fps=int(round(1.0 / env.dt)), macro_block_size=1)
        print("  saved", p)
    env.close()
    worst = max(abs(min(th)), abs(max(th)))
    ok = worst <= lim + 3.0
    print(f"  A {name:5s}: ankle range {min(th):8.1f} .. {max(th):7.1f} deg (limit +-{lim:.0f}), peak speed {max(om):6.1f} rad/s (OSL spec 6.13)  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_b(name, args):
    from terrain_memory_prosthesis.utils.myoassist import ensure_env_registered

    cfg = common.load_config(args.session)
    cfg.env_params.env_id = ENV_IDS[name]
    ensure_env_registered(cfg.env_params.env_id)
    ck = common.resolve(args.checkpoint) if args.checkpoint else common.list_checkpoints(args.session)[-1][1]
    env = common.make_eval_env(cfg, args.max_steps)
    model = common.load_model(ck)
    m, d = raw(env)
    lim = limit_deg(env)
    dt = env.dt
    w0 = warn_counts(d)
    unstable, speeds, at_lim, hits, longest, nsteps = False, [], 0, 0, 0, 0
    for ep in range(args.episodes):
        with common.seeded_rng(ep):
            obs, _ = env.reset()
        run = 0
        prev_at = False
        for t in range(args.max_steps):
            a, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _ = env.step(a)
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qacc).all()):
                unstable = True
                break
            th, w = ankle(env)
            if term:
                break
            nsteps += 1
            speeds.append(abs(w))
            now = abs(th) >= lim - 1.0
            at_lim += int(now)
            hits += int(now and not prev_at)
            run = run + 1 if now else 0
            longest = max(longest, run)
            prev_at = now
            if trunc:
                break
    w1 = warn_counts(d)
    env.close()
    new_warn = {k: w1[k] - w0[k] for k in w1 if w1[k] - w0[k] > 0}
    secs = max(nsteps * dt, 1e-9)
    frac, hz, peak, stint = at_lim / max(nsteps, 1), hits / secs, (max(speeds) if speeds else 0.0), longest * dt
    checks = {
        "no solver instability": (not unstable) and not new_warn,
        f"peak ankle speed {peak:.0f} rad/s <= {args.max_speed:g}": peak <= args.max_speed,
        f"at limit {frac:.0%} of steps <= 25%": frac <= 0.25,
        f"longest stint at limit {stint:.2f} s <= 0.5 s": stint <= 0.5,
        f"limit hits {hz:.1f}/s <= 4/s": hz <= 4.0,
    }
    ok = all(checks.values())
    print(f"  B {name:5s}: {nsteps} walking steps ({secs:.1f} s) -> {'PASS' if ok else 'FAIL'}")
    for k, v in checks.items():
        print(f"      [{'ok' if v else 'XX'}] {k}")
    if new_warn:
        print("      MuJoCo warnings:", new_warn)
    if nsteps < 60:
        print("      (very few walking steps: the policy falls at once, so this verdict is weak)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default=None, help="enables test B (trained policy, gravity on)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=8.0, help="length of the test-A video / stress test [s]")
    ap.add_argument("--max-speed", type=float, default=40.0)
    args = ap.parse_args()

    res = {}
    print("TEST A - gravity OFF, full-torque pulses (isolated joint-limit stress test)")
    for name in ("stock", "stiff", "phys"):
        res[("A", name)] = test_a(name, args)
    if args.session:
        print("\nTEST B - gravity ON, policy walking (normal dynamics)")
        for name in ("stock", "stiff", "phys"):
            res[("B", name)] = test_b(name, args)
    print("\nsummary:", ", ".join(f"{t}-{n}: {'PASS' if v else 'FAIL'}" for (t, n), v in res.items()))
    return 0 if all(v for (t, n), v in res.items() if n == "phys") else 1


if __name__ == "__main__":
    sys.exit(main())
