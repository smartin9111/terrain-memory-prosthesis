"""MyoAssist glue: locating the checkout, loading configs/checkpoints, building evaluation and training envs.

Rules
  * Call `init()` BEFORE importing torch / rl_train. It caps thread pools, sets a headless MuJoCo backend and chdir()s into
    the MyoAssist checkout (its configs use paths relative to it, e.g. rl_train/reference_data/...).
  * User-given paths are resolved against the directory the script was started from (`resolve()`).
  * Evaluation mirrors MyoAssist's own evaluator (rl_train/analyzer/gait_evaluate.py): evaluate mode, imitation
    early-termination OFF, long episodes. "fell" therefore means exactly one thing: the pelvis dropped below
    `safe_height` (the env's own fall condition), not "deviated from the reference gait".
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

ORIG_CWD = os.getcwd()
# <workspace>/terrain-memory-prosthesis/src/terrain_memory_prosthesis/environment/myoassist.py -> <workspace>/myoassist
SIBLING_MYOASSIST = Path(__file__).resolve().parents[4] / "myoassist"
PREVIEW_ENV_IDS = ("myoAssistLegImitationExoPreview-v0",)
STIFF_ENV_ID = "myoAssistLegImitationExoStiff-v0"
OSL_PHYS_ENV_ID = "myoAssistLegImitationExoOSLPhys-v0"


def resolve(p):
    """Resolve a user-given path against the directory the script was started from."""
    if p is None:
        return None
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(ORIG_CWD, p))


def init(root: str | None = None, threads: int = 1) -> str:
    """Prepare the process. Call before importing torch / rl_train. Returns the MyoAssist root."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(threads))
    if sys.platform.startswith("linux"):  # headless GL on Linux/UCloud
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif os.environ.get("MUJOCO_GL", "").lower() in ("egl", "osmesa"):  # macOS has no EGL/OSMesa: MuJoCo would refuse to import
        del os.environ["MUJOCO_GL"]
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    root = root or os.environ.get("MYOASSIST_DIR")
    if not root:
        import importlib.util

        spec = importlib.util.find_spec("rl_train")
        if spec is not None and spec.origin:
            root = str(Path(spec.origin).resolve().parent.parent)
        elif SIBLING_MYOASSIST.is_dir():  # repo layout of this project: <workspace>/myoassist next to the repo
            root = str(SIBLING_MYOASSIST)
        else:
            raise RuntimeError(
                "Cannot find the MyoAssist checkout. Install it (uv pip install -e ../myoassist), "
                "or set MYOASSIST_DIR=/path/to/myoassist"
            )
    root = str(Path(root).resolve())
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def out_dir(path: str | None, default: str) -> str:
    d = resolve(path) if path else os.path.join(ORIG_CWD, default)
    os.makedirs(d, exist_ok=True)
    return d


def session_config_path(path: str) -> str:
    p = resolve(path)
    return os.path.join(p, "session_config.json") if os.path.isdir(p) else p


def list_checkpoints(session_dir: str) -> list[tuple[int, str]]:
    """[(timesteps, path)] sorted, from <session>/trained_models/model_<timesteps>.zip"""
    out = []
    for f in glob.glob(os.path.join(resolve(session_dir), "trained_models", "model_*")):
        m = re.search(r"model_(\d+)(?:\.zip)?$", f)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def pick_checkpoints(session_dir: str, max_ckpts: int) -> list[tuple[int, str]]:
    cks = list_checkpoints(session_dir)
    if len(cks) <= max_ckpts:
        return cks
    idx = sorted({round(i * (len(cks) - 1) / (max_ckpts - 1)) for i in range(max_ckpts)})
    return [cks[i] for i in idx]


def ensure_env_registered(env_id: str):
    if env_id in PREVIEW_ENV_IDS:
        from terrain_memory_prosthesis.environment import preview_env

        preview_env.register()
    elif env_id == OSL_PHYS_ENV_ID:
        from terrain_memory_prosthesis.environment import osl_physics

        osl_physics.register()
    elif env_id == STIFF_ENV_ID:
        from terrain_memory_prosthesis.environment import stiff_limit

        stiff_limit.register()


def apply_run_env(session_or_config: str) -> dict:
    """If the session is a Stage-2 run, set the environment variables it was trained with - the perception settings (MYO_PERCEPTION,
    MYO_PREVIEW_*, MYO_SENSOR_*) and the controller settings (C2: MYO_HISTORY, MYO_GRU_HIDDEN) - so that every tool (video, pilots, evaluation)
    builds the env and the policy the checkpoint expects. Returns what was set ({} for Stage-1 sessions / templates)."""
    p = resolve(session_or_config)
    d = p if os.path.isdir(p) else os.path.dirname(p)
    info = os.path.join(d, "stage2_info.json")
    if not os.path.isfile(info):
        return {}
    with open(info) as f:
        meta = json.load(f)
    env = {**(meta.get("perception_env") or {}), **(meta.get("controller_env") or {})}
    # MYO_SENSOR_*: a run recorded without them (older runs) must not inherit noise / delay / dropout from the caller's shell
    for k in [k for k in os.environ if k in ("MYO_PERCEPTION", "MYO_HISTORY", "MYO_GRU_HIDDEN") or k.startswith(("MYO_PREVIEW_", "MYO_SENSOR_"))]:
        del os.environ[k]
    os.environ.update({k: str(v) for k, v in env.items()})
    return env


def load_config(path: str):
    """Typed MyoAssist session config from a session dir, a session_config.json or a config template."""
    apply_run_env(path)
    p = session_config_path(path)
    with open(p) as f:
        env_id = json.load(f)["env_params"]["env_id"]
    ensure_env_registered(env_id)
    import rl_train.train.train_configs.config as C
    from rl_train.envs.environment_handler import EnvironmentHandler as H

    d = H.get_session_config_from_path(p, C.TrainSessionConfigBase)
    t = H.get_config_type_from_session_id(d.env_params.env_id)
    return H.get_session_config_from_path(p, t)


def load_model(ckpt: str, env=None):
    """Load a stage-1/2 checkpoint (human + prosthesis actor policy) for inference.

    A checkpoint of a controller with its own policy class (C2 = "gru") stores that class BY REFERENCE (module + name), so SB3 imports
    it itself; we only have to set the run's environment variables first (window length, GRU size: stage2_info.json).
    `env` (optional): attach a training env, needed to CONTINUE training (the rollout buffer is sized from it)."""
    from stable_baselines3 import PPO
    from rl_train.train.policies.rl_agent_exo import HumanExoActorCriticPolicy

    ckpt = resolve(ckpt)
    run_dir = os.path.dirname(os.path.dirname(ckpt))  # <run>/trained_models/model_N.zip
    info = os.path.join(run_dir, "stage2_info.json")
    if os.path.isfile(info):
        with open(info) as f:
            controller = (json.load(f).get("args") or {}).get("controller", "mlp")
        if controller != "mlp":
            apply_run_env(run_dir)
            return PPO.load(ckpt, env=env, device="cpu")
    return PPO.load(ckpt, env=env, device="cpu", custom_objects={"policy_class": HumanExoActorCriticPolicy})


def make_eval_env(cfg, max_steps: int, target_velocity: float | None = None):
    """Single raw env in evaluate mode, imitation early-termination off (same as MyoAssist's evaluator)."""
    from rl_train.envs.environment_handler import EnvironmentHandler as H
    from rl_train.envs.myoassist_leg_base import MyoAssistLegBase

    cfg.env_params.num_envs = 1
    cfg.env_params.custom_max_episode_steps = int(max_steps)
    cfg.env_params.out_of_trajectory_threshold = 1e6
    env = H.create_environment(cfg, is_rendering_on=False, is_evaluate_mode=True)
    v = float(target_velocity if target_velocity is not None else cfg.env_params.min_target_velocity)
    env.set_target_velocity_mode_manually(MyoAssistLegBase.VelocityMode.UNIFORM, 0, v, v, v, target_velocity_period=None)
    return env


def device_actuator_ids(env) -> list[int]:
    import mujoco

    m = env.sim.model
    return [i for i in range(m.nu) if m.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_MUSCLE]


def read_config_dict(path: str) -> dict:
    with open(session_config_path(path)) as f:
        return json.load(f)


def config_from_dict(d: dict, tmp_dir: str, name: str = "cfg"):
    """Typed config from an edited config dict (written to tmp_dir/name.json, as MyoAssist loads from files)."""
    os.makedirs(tmp_dir, exist_ok=True)
    p = os.path.abspath(os.path.join(tmp_dir, f"{name}.json"))  # tmp_dir is relative to the CURRENT directory (the MyoAssist root)
    with open(p, "w") as f:
        json.dump(d, f)
    return load_config(p)


def build_model(cfg, num_envs: int = 1, n_steps: int | None = None, batch_size: int | None = None, n_epochs: int | None = None):
    """Training env + PPO model (HumanExoActorCriticPolicy) as MyoAssist's run_train builds them."""
    from rl_train.envs.environment_handler import EnvironmentHandler as H

    cfg.env_params.num_envs = num_envs
    for k, v in (("n_steps", n_steps), ("batch_size", batch_size), ("n_epochs", n_epochs)):
        if v is not None:
            setattr(cfg.ppo_params, k, v)
    env = H.create_environment(cfg, False)
    return H.get_stable_baselines3_model(cfg, env), env
