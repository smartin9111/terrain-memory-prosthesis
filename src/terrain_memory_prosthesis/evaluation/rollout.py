"""Deterministic, seeded rollouts of a (human + prosthesis) policy, optional perturbation of the prosthesis command."""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import multiprocessing as mp
import os
import random
import time

from terrain_memory_prosthesis.evaluation.reporting import summarize
from terrain_memory_prosthesis.environment.myoassist import device_actuator_ids, init, load_config, load_model, make_eval_env


@contextlib.contextmanager
def seeded_rng(seed: int):
    """The imitation env draws its random start phase from a fresh np.random.default_rng() in reset().
    Inside this context that call returns one seeded generator, so evaluation episodes are reproducible."""
    import numpy as np

    orig = np.random.default_rng
    gen = orig(seed)
    np.random.default_rng = lambda *a, **k: gen if (not a and not k) else orig(*a, **k)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    try:
        yield
    finally:
        np.random.default_rng = orig


class Perturbation:
    """Wraps the policy action. `kind` in: none | scale | noise | delay | zero | offset.
    Acts on the device (non-muscle) action entries only, in the normalised [-1, 1] action units."""

    def __init__(self, kind: str = "none", value: float = 0.0, device_ids=None, seed: int = 0):
        import numpy as np

        self.kind, self.value, self.ids = kind, value, list(device_ids or [])
        self.rng = np.random.default_rng(seed)
        self.buf: list = []

    def reset(self):
        self.buf = []

    def __call__(self, a):
        import numpy as np

        a = np.array(a, dtype=float, copy=True)
        if not self.ids or self.kind == "none":
            return a
        d = a[self.ids]
        if self.kind == "scale":
            d = d * self.value
        elif self.kind == "noise":
            d = d + self.rng.normal(0.0, self.value, size=d.shape)
        elif self.kind == "offset":
            d = d + self.value
        elif self.kind == "zero":
            d = np.zeros_like(d)
        elif self.kind == "delay":
            k = int(self.value)
            self.buf.append(d.copy())
            d = self.buf[-1 - k] if len(self.buf) > k else np.zeros_like(d)
        else:
            raise ValueError(f"unknown perturbation {self.kind}")
        a[self.ids] = np.clip(d, -1.0, 1.0)
        return a


def run_episodes(model, env, n_episodes: int, max_steps: int, seeds: list[int] | None = None, perturb: Perturbation | None = None):
    import numpy as np

    seeds = seeds or list(range(n_episodes))
    dev = device_actuator_ids(env)
    target = float(env._target_velocity)
    out = []
    for i in range(n_episodes):
        seed = seeds[i % len(seeds)]
        if perturb is not None:
            perturb.reset()
            perturb.rng = np.random.default_rng(seed + 1000)
        with seeded_rng(seed):
            obs, _ = env.reset()
        x0 = float(env.sim.data.joint("pelvis_tx").qpos[0])
        verr, cmd, steps, fell, ret = [], [], 0, False, 0.0
        for t in range(max_steps):
            a, _ = model.predict(obs, deterministic=True)
            if perturb is not None:
                a = perturb(a)
            obs, _r, terminated, truncated, _info = env.step(a)
            steps += 1
            ret += float(_r)
            verr.append(abs(float(env.sim.data.joint("pelvis_tx").qvel[0]) - target))
            if dev:
                cmd.append(float(np.mean(np.abs(np.asarray(a)[dev]))))
            if terminated:
                fell = True
                break
            if truncated:
                break
        x1 = float(env.sim.data.joint("pelvis_tx").qpos[0])
        dist = x1 - x0
        out.append(
            {
                "seed": seed,
                "steps": steps,
                "fell": fell,
                "distance": dist,
                "speed": dist / (steps * env.dt) if steps else 0.0,
                "speed_err": float(np.mean(verr)) if verr else float("nan"),
                "prosthesis_cmd": float(np.mean(cmd)) if cmd else 0.0,
                "reward": ret,
                "reward_per_step": ret / steps if steps else 0.0,
            }
        )
    return out


def eval_task(task: dict) -> dict:
    """Worker entry point (picklable, top-level). One env build, `episodes` deterministic episodes.

    task keys: config, checkpoint, episodes, max_steps, seed0, terrain (optional; 'keep' = leave config's),
               perturb ({'kind','value'}), target_velocity, label, extra (copied to the result),
               env (environment variables to set before the env is built, e.g. {"MYO_PERCEPTION": "none"})
    """
    init(task.get("root"))
    if task.get("env") is not None:  # the run's perception settings (must match training); replaces any left over from a previous task
        for k in [k for k in os.environ if k == "MYO_PERCEPTION" or k.startswith(("MYO_PREVIEW_", "MYO_SENSOR_", "MYO_GRU_ABLATE"))]:
            del os.environ[k]
        os.environ.update({k: str(v) for k, v in task["env"].items()})
    cfg = load_config(task["config"])
    if task.get("terrain", "keep") != "keep":
        cfg.env_params.terrain = task["terrain"]
    env = make_eval_env(cfg, task["max_steps"], task.get("target_velocity"))
    model = load_model(task["checkpoint"])
    p = task.get("perturb") or {"kind": "none", "value": 0.0}
    pert = Perturbation(p["kind"], p["value"], device_actuator_ids(env))
    seed0 = task.get("seed0", 0)
    t0 = time.time()
    eps = run_episodes(model, env, task["episodes"], task["max_steps"], seeds=[seed0 + i for i in range(task["episodes"])], perturb=pert)
    env.close()
    res = {"label": task.get("label"), **(task.get("extra") or {}), **summarize(eps), "wall_s": time.time() - t0}
    res["episodes"] = eps
    return res


def run_parallel(tasks: list[dict], workers: int = 1, fn=eval_task, root: str | None = None) -> list:
    """Run tasks in separate processes (spawn: a clean MuJoCo/torch state per worker). Order is preserved."""
    for t in tasks:
        t.setdefault("root", root or os.getcwd())
    if workers <= 1:
        return [fn(t) for t in tasks]
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        return list(ex.map(fn, tasks))


def strip_episodes(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "episodes"} for r in rows]
