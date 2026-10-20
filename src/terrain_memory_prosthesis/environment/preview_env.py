"""Stage-2 environment: MyoAssistLegImitationExo with actor-only sensor imperfections and terrain perception.

    obs = [ clean proprioception 40 | corrupted sensor values 40 | validity 40 | perception | privileged preview 5 ]

(see common/obs_layout.py). The human actor and critic read the clean observations; only the prosthesis actor reads selected channels
from the corrupted sensor block and matching validity bits. MYO_SENSOR_NOISE_STD, MYO_SENSOR_DELAY_STEPS, and
MYO_SENSOR_DROPOUT_PROBABILITY configure the sensor model. The perception model is chosen by MYO_PERCEPTION and its own
settings (MYO_PREVIEW_DISTANCES, MYO_PREVIEW_SCALE, MYO_PREVIEW_ZERO, ...).

Registration happens at import time because SB3's SubprocVecEnv workers are fresh processes that only know the ids registered
there: the script that creates the env must import this module at top level.
"""
from __future__ import annotations

import json
import os

import numpy as np

from terrain_memory_prosthesis.common.obs_layout import N_PRIVILEGED, N_PROPRIO, PERCEPTION_ENV_VAR, PREVIEW_ENV_ID
from terrain_memory_prosthesis.common.sensor_effects import ProprioceptiveSensorModel
from terrain_memory_prosthesis.environment.osl_physics import NoFlightPenaltyMixin, apply_osl_physics
from terrain_memory_prosthesis.environment.terrain import parse_episodic

_ENTRY_POINT = "terrain_memory_prosthesis.environment.preview_env:MyoAssistLegImitationExoPreview"


def register():
    """Idempotent: register the env id and teach MyoAssist's EnvironmentHandler about it."""
    from terrain_memory_prosthesis.environment._registry import register_exo_env

    register_exo_env(PREVIEW_ENV_ID, _ENTRY_POINT)


def build_perception():
    """(actor perception model, privileged clean-preview model) for this process."""
    from terrain_memory_prosthesis.perception.clean_preview import DEFAULT_DISTANCES, DEFAULT_SCALE, CleanPreview
    from terrain_memory_prosthesis.perception.registry import build

    percept = build(os.environ.get(PERCEPTION_ENV_VAR, "clean_preview"))
    priv = CleanPreview(distances=DEFAULT_DISTANCES, scale=DEFAULT_SCALE, zero=False)  # independent of the MYO_PREVIEW_* variables
    assert priv.size == N_PRIVILEGED
    return percept, priv


def _make_env_class():
    from rl_train.envs.myoassist_leg_imitation_exo import MyoAssistLegImitationExo

    class MyoAssistLegImitationExoPreview(NoFlightPenaltyMixin, MyoAssistLegImitationExo):
        _BASE_OBS_KEYS = list(MyoAssistLegImitationExo.DEFAULT_OBS_KEYS)
        DEFAULT_OBS_KEYS = _BASE_OBS_KEYS + ["proprio_sensor_values", "proprio_sensor_validity", "terrain_percept", "terrain_privileged"]

        def _setup(self, **kwargs):
            self._percept, self._priv = build_perception()
            def env_parameter(name, default):
                raw = os.environ.get(name)
                if raw is None or raw.strip() == "":
                    return default
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    # Convenient scalar syntax (e.g. "0.01") as well as JSON lists.
                    try:
                        return float(raw)
                    except ValueError as exc:
                        raise ValueError(f"{name} must be a scalar or JSON array") from exc
            self._sensor = ProprioceptiveSensorModel(
                size=N_PROPRIO,
                noise_std=env_parameter("MYO_SENSOR_NOISE_STD", 0.0),
                delay_steps=int(os.environ.get("MYO_SENSOR_DELAY_STEPS", "0")),
                dropout_probability=env_parameter("MYO_SENSOR_DROPOUT_PROBABILITY", 0.0),
            )
            super()._setup(**kwargs)
            m = self.sim.model
            apply_osl_physics(getattr(m, "ptr", m))  # same prosthesis physics as the Stage-1 env (hard limit + motor model, see environment/osl_physics.py)
            # Stage 2 training terrain: new random field at every reset if the terrain name says "episodic_<lo>_<hi>"
            self._episode_terrain = None
            rng_range = parse_episodic(getattr(kwargs.get("env_params"), "terrain", None))
            if rng_range is not None:
                from terrain_memory_prosthesis.environment.episode_terrain import EpisodeTerrain

                terr = kwargs["env_params"].terrain
                self._episode_terrain = EpisodeTerrain(m, *rng_range, safe_radius=float(terr.get("safe_zone_radius", 3.0)))

        def reset(self, **kwargs):
            if self._episode_terrain is not None:
                self._episode_terrain.resample(self.sim.model)
            self._percept.reset(kwargs.get("seed"))
            self._sensor.reset(kwargs.get("seed"))
            return super().reset(**kwargs)

        def get_obs_dict(self, sim):
            obs = super().get_obs_dict(sim)
            clean = np.concatenate([np.asarray(obs[k], dtype=np.float64).reshape(-1) for k in self._BASE_OBS_KEYS])
            if clean.shape != (N_PROPRIO,):
                raise ValueError(f"expected {N_PROPRIO} base proprioceptive values, got {clean.shape}")
            sample = self._sensor.observe(clean)
            obs["proprio_sensor_values"] = sample.values
            obs["proprio_sensor_validity"] = sample.validity
            obs["terrain_percept"] = np.asarray(self._percept.observe(sim.model, sim.data), dtype=float)
            obs["terrain_privileged"] = np.asarray(self._priv.observe(sim.model, sim.data), dtype=float)
            return obs

    return MyoAssistLegImitationExoPreview


def __getattr__(name):  # lazy: gym imports "<module>:MyoAssistLegImitationExoPreview"
    if name == "MyoAssistLegImitationExoPreview":
        cls = _make_env_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


try:  # register at import (needed in subprocess workers); harmless if MyoAssist is not importable yet
    register()
except Exception:  # pragma: no cover
    pass
