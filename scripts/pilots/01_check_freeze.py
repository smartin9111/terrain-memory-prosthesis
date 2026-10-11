#!/usr/bin/env python
"""01 - Does the freezing of the human policy really work in Stage 2?

Question : When the prosthesis controller (P2 = bigger input) is trained, does the human (muscle) policy stay
           bit-for-bit what Stage 1 produced? If not, P1/P2 and MLP/GRU differences are confounded by a drifting human.
Method   : build model A (stage-1 stand-in: --stage1 checkpoint, or random init) and model B whose prosthesis actor
           has a LARGER input (extra observation range, as in P2). `load_and_freeze_human(B, A)`, then train B a few
           steps with PPO and check:
             1. human weights in B == human weights in A right after loading           (exact)
             2. muscle log_std in B == in A                                            (exact)
             3. human weights unchanged after training                                  (exact)
             4. muscle log_std unchanged after training                                 (exact)
             5. human (muscle) action on fixed observations unchanged after training    (allclose, atol 1e-7)
             6. prosthesis actor weights DID change (training actually ran)
             7. negative control: the same run WITHOUT freezing changes the human weights (the test has teeth)
           Info: the frozen state (requires_grad / grad hook) is NOT stored in the checkpoint -> after a PPO.load
           (resume) call load_and_freeze_human-style freezing again. The script reports whether weights survive a save/load.
Exit code: 0 PASS, 2 FAIL.

Example:
    python 01_check_freeze.py --config path/to/p1_c1_osl.json
    python 01_check_freeze.py --config path/to/p1_c1_osl.json --stage1 rl_train/results/train_session_X/trained_models/model_N.zip
"""
import argparse
import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

N_MUSCLE = 18


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="stage-1 style config (json template or session_config.json)")
    ap.add_argument("--stage1", default=None, help="optional stage-1 checkpoint; default = random-init stand-in")
    ap.add_argument("--extra-obs", default="17:22", help="obs range the bigger prosthesis actor additionally sees (P2 stand-in)")
    ap.add_argument("--steps", type=int, default=512, help="PPO steps of the short stage-2 training")
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse()
    common.init(args.root)
    out = common.out_dir(args.out, "pilot_out/01")
    import numpy as np
    import torch
    from terrain_memory_prosthesis.utils.freeze import load_and_freeze_human

    lo, hi = map(int, args.extra_obs.split(":"))
    tmp = tempfile.mkdtemp(prefix="freeze_")
    base = common.read_config_dict(args.config)
    ppo = dict(n_steps=args.n_steps, batch_size=args.n_steps, n_epochs=2)

    def build(d, name):
        return common.build_model(common.config_from_dict(d, tmp, name), num_envs=1, **ppo)[0]

    # ---- model A: stage-1 stand-in
    if args.stage1:
        A = common.load_model(args.stage1)
        print(f"stage-1 model loaded from {args.stage1}")
    else:
        A = build(base, "A")
        with torch.no_grad():
            A.policy.log_std[:N_MUSCLE] = -1.0  # pretend stage 1 learned a smaller muscle std
        print("stage-1 stand-in: random init, muscle log_std set to -1.0")

    d_big = common.add_exo_obs_range(base, lo, hi)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'OK ' if ok else 'BAD'}] {name} {detail}")

    def human_sd(m):
        return {k: v.detach().clone() for k, v in m.policy.policy_network.human_policy_net.state_dict().items()}

    def max_diff(a, b):
        return max(float((a[k] - b[k]).abs().max()) for k in a)

    hA = human_sd(A)
    n_in_A = A.policy.policy_network.exo_policy_net[0].weight.shape[1]

    def run(freeze: bool, name: str):
        B = build(d_big, name)
        if freeze:
            load_and_freeze_human(B, A, n_muscle=N_MUSCLE)
        else:  # plain copy, no freezing
            B.policy.policy_network.human_policy_net.load_state_dict(A.policy.policy_network.human_policy_net.state_dict())
            with torch.no_grad():
                B.policy.log_std[:N_MUSCLE] = A.policy.log_std[:N_MUSCLE].to(B.policy.log_std.device)
        return B

    B = run(True, "B_frozen")
    n_in_B = B.policy.policy_network.exo_policy_net[0].weight.shape[1]
    check("prosthesis actor input is larger in B (P2-like)", n_in_B > n_in_A, f"({n_in_A} -> {n_in_B})")

    # 1-2: after loading
    check("1. human weights identical after load", max_diff(hA, human_sd(B)) == 0.0)
    ls_A = A.policy.log_std[:N_MUSCLE].detach().cpu()
    check("2. muscle log_std identical after load", torch.equal(ls_A, B.policy.log_std[:N_MUSCLE].detach().cpu()))

    # fixed observation batch for 5
    rng = np.random.default_rng(0)
    obs_dim = B.observation_space.shape[0]
    obs = rng.normal(0, 1, size=(32, obs_dim)).astype(np.float32)

    def human_actions(m):
        a, _ = m.policy.predict(obs, deterministic=True)
        return np.asarray(a)[:, :N_MUSCLE].copy()

    a_before = human_actions(B)
    sd0 = {k: v.detach().clone() for k, v in B.policy.state_dict().items()}
    B.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    sd1 = {k: v.detach().clone() for k, v in B.policy.state_dict().items()}

    dh = max(float((sd1[k] - sd0[k]).abs().max()) for k in sd0 if "human_policy_net" in k)
    de = max(float((sd1[k] - sd0[k]).abs().max()) for k in sd0 if "exo_policy_net" in k)
    dls = float((sd1["log_std"] - sd0["log_std"]).abs()[:N_MUSCLE].max())
    check("3. human weights unchanged after training", dh == 0.0, f"(max |dW| = {dh:.3g})")
    check("4. muscle log_std unchanged after training", dls == 0.0, f"(max |d| = {dls:.3g})")
    a_after = human_actions(B)
    err = float(np.abs(a_after - a_before).max())
    check("5. human action on fixed observations unchanged", np.allclose(a_after, a_before, atol=1e-7), f"(max err {err:.3g})")
    check("6. prosthesis actor weights changed (training ran)", de > 0.0, f"(max |dW| = {de:.3g})")

    # 7: negative control
    C = run(False, "C_unfrozen")
    c0 = human_sd(C)
    C.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    dc = max_diff(c0, human_sd(C))
    check("7. negative control: WITHOUT freezing the human does change", dc > 0.0, f"(max |dW| = {dc:.3g})")

    # info: save/load round trip
    ck = os.path.join(tmp, "B_frozen.zip")
    B.save(ck)
    B2 = common.load_model(ck)
    rt = max_diff(human_sd(B), human_sd(B2))
    flags = {n: p.requires_grad for n, p in B2.policy.policy_network.human_policy_net.named_parameters()}
    print(f"[info] save/load: human weights preserved (max diff {rt:.3g}); requires_grad after PPO.load all True = {all(flags.values())}"
          " -> when resuming a stage-2 run, freeze again.")

    ok = all(c["ok"] for c in checks)
    common.save_rows(out, "01_checks", checks)
    reasons = [f"{c['check']} {c['detail']}" for c in checks if not c["ok"]] or ["all checks passed: stage-2 training leaves the human policy untouched"]
    return common.verdict(out, "01_check_freeze", "PASS" if ok else "FAIL", reasons)


if __name__ == "__main__":
    sys.exit(main())
