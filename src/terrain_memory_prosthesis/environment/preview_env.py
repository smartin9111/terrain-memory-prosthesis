"""Stage-2 environment: MyoAssistLegImitationExo (+ OSL physics, no-flight penalty, per-episode terrain) with two extra observation blocks

    obs = [ proprioceptive 40 | perception block (the prosthesis actor's, size = PerceptionModel.size) | privileged block (clean preview, critic) ]

(see common/obs_layout.py). The perception model is chosen by the environment variable MYO_PERCEPTION (default "clean_preview";
P1 = "none"), so it is inherited by SubprocVecEnv workers; model-specific settings are read by the models themselves
(MYO_PREVIEW_DISTANCES, MYO_PREVIEW_SCALE, MYO_PREVIEW_ZERO, ...). This module only wires: it contains no perception logic.

Registration happens at import time because SB3's SubprocVecEnv workers are fresh processes that only know the ids registered
there: the script that creates the env must import this module at top level.
"""
from __future__ import annotations

import os

import numpy as np

from terrain_memory_prosthesis.common.obs_layout import N_PRIVILEGED, PERCEPTION_ENV_VAR, PREVIEW_ENV_ID
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
        DEFAULT_OBS_KEYS = MyoAssistLegImitationExo.DEFAULT_OBS_KEYS + ["terrain_percept", "terrain_privileged"]

        def _setup(self, **kwargs):
            self._percept, self._priv = build_perception()
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
            return super().reset(**kwargs)

        def get_obs_dict(self, sim):
            obs = super().get_obs_dict(sim)
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
