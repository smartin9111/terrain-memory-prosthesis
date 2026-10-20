"""Stage 2 assembly: Stage-1 policy -> frozen human + prosthesis controller (C1/C2) + perception (P1/P2) -> PPO on episodic rough terrain.

This is the ONLY place where perception and controllers meet; both are chosen by NAME (perception/registry.py,
controllers/registry.py), so neither side edits the other's code to add a variant. The launcher scripts/train_stage2.py is a thin
CLI around `run(args)`. Call it only after `environment.myoassist.init()` (thread caps, chdir to the MyoAssist root).
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from terrain_memory_prosthesis.common.obs_layout import PERCEPTION_ENV_VAR
from terrain_memory_prosthesis.environment import myoassist as ma

N_MUSCLE = 18
ROLLOUT = 16384
DEFAULT_PERCEPTION = {"P1": "none", "P2": "clean_preview"}


def claim_session_dir(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in range(100):  # collision-free, same idea as MyoAssist's run_train
        d = os.path.join("rl_train", "results", f"{name}_{stamp}" + (f"_{suffix}" if suffix else ""))
        try:
            os.makedirs(d, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise RuntimeError("could not claim a session directory")


def perception_env(cond: str, perception: str | None, preview_distances: str | None) -> dict:
    """Environment variables that define the perception of a run (stored in stage2_info.json, re-applied by the evaluation)."""
    env = {PERCEPTION_ENV_VAR: perception or DEFAULT_PERCEPTION[cond]}
    if preview_distances:
        env["MYO_PREVIEW_DISTANCES"] = preview_distances
    # Record and propagate the exact sensor impairment settings to subprocess workers and evaluation.
    env["MYO_SENSOR_NOISE_STD"] = os.environ.get("MYO_SENSOR_NOISE_STD", "0")
    env["MYO_SENSOR_DELAY_STEPS"] = os.environ.get("MYO_SENSOR_DELAY_STEPS", "0")
    env["MYO_SENSOR_DROPOUT_PROBABILITY"] = os.environ.get("MYO_SENSOR_DROPOUT_PROBABILITY", "0")
    return env


def run(a) -> str:
    """`a`: argparse namespace of scripts/train_stage2.py. Returns the session directory."""
    if ROLLOUT % a.num_envs:
        raise SystemExit(f"--num-envs must divide {ROLLOUT}")
    import numpy as np
    import torch
    from rl_train.envs.environment_handler import EnvironmentHandler as H
    from rl_train.utils import train_log_handler
    from rl_train.utils.data_types import DictionableDataclass

    from terrain_memory_prosthesis.common.obs_layout import N_PROPRIO
    from terrain_memory_prosthesis.common.session_config import make_session_config
    from terrain_memory_prosthesis.controllers import registry as controllers
    from terrain_memory_prosthesis.controllers.warm_start import check_equivalent
    from terrain_memory_prosthesis.environment.terrain import episodic_terrain
    from terrain_memory_prosthesis.perception import registry as perception

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    session = ma.resolve(a.stage1_session)
    ckpt = ma.resolve(a.stage1_ckpt) if a.stage1_ckpt else ma.list_checkpoints(session)[-1][1]
    base = ma.read_config_dict(session)
    base["env_params"]["prev_trained_policy_path"] = None

    # perception: chosen by name; the env variables are inherited by the SB3 workers created below
    penv = perception_env(a.cond, getattr(a, "perception", None), getattr(a, "preview_distances", None))
    os.environ.update(penv)
    n_percept = perception.build(penv[PERCEPTION_ENV_VAR]).size
    if (a.cond == "P1") != (n_percept == 0):
        raise SystemExit(f"condition {a.cond} does not fit perception {penv[PERCEPTION_ENV_VAR]!r} (size {n_percept})")

    terrain = episodic_terrain(a.amp_lo, a.amp_hi)
    terrain["safe_zone_radius"] = a.safe_radius
    d = make_session_config(base, a.cond, terrain, n_percept=n_percept)
    d["env_params"]["num_envs"] = a.num_envs
    d["ppo_params"]["n_steps"] = ROLLOUT // a.num_envs
    if a.smoke:
        d["ppo_params"].update(n_steps=64, batch_size=64, n_epochs=2)
    d["total_timesteps"] = a.timesteps
    d["logger_params"]["logging_frequency"] = a.logfreq
    name = f"stage2_{a.cond}_{a.controller}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    log_dir = claim_session_dir(name)
    print(f"Session directory: {log_dir}", flush=True)
    cfg = ma.config_from_dict(d, log_dir, "start_config")
    cfg.env_params.seed = a.seed
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    env = H.create_environment(cfg, False)
    model = H.get_stable_baselines3_model(cfg, env)
    model.set_random_seed(a.seed)
    stage1 = ma.load_model(ckpt)
    info = controllers.apply(a.controller, model, stage1, n_muscle=N_MUSCLE)
    diff = check_equivalent(model, stage1, n_obs_old=N_PROPRIO)
    n_train = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"warm start from {ckpt}\n  {info}\n  equivalent to Stage 1 on random observations (max diff {diff:.1e}); "
          f"trainable parameters: {n_train}", flush=True)

    H.updateconfig_from_model_policy(cfg, model)
    sc = DictionableDataclass.to_dict(cfg)
    sc["env_params"].pop("reference_data", None)
    with open(os.path.join(log_dir, "session_config.json"), "w", encoding="utf-8") as f:
        json.dump(sc, f, ensure_ascii=False, indent=2)
    with open(os.path.join(log_dir, "stage2_info.json"), "w") as f:
        json.dump(dict(args=vars(a), stage1_session=session, stage1_ckpt=ckpt, perception_env=penv, n_percept=n_percept,
                       warm_start=info, equiv_max_diff=diff, trainable_parameters=n_train, terrain=terrain,
                       git=git_info(), started=datetime.now().isoformat()), f, indent=2)

    handler = train_log_handler.TrainLogHandler(log_dir)
    callback = H.get_callback(cfg, handler)
    model.learn(reset_num_timesteps=False, total_timesteps=a.timesteps, log_interval=1, callback=callback, progress_bar=False)
    os.makedirs(os.path.join(log_dir, "trained_models"), exist_ok=True)
    model.save(os.path.join(log_dir, "trained_models", f"model_{model.num_timesteps}"))
    env.close()
    print("stage 2 done:", log_dir, flush=True)
    return log_dir


def git_info() -> dict:
    """Commit of THIS repository (provenance of the result); never fails."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        run_ = lambda *c: subprocess.check_output(["git", "-C", here, *c], stderr=subprocess.DEVNULL).decode().strip()
        return {"commit": run_("rev-parse", "--short", "HEAD"), "branch": run_("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run_("status", "--porcelain"))}
    except Exception:
        return {"commit": "unknown"}
