"""P2 environment: MyoAssistLegImitationExo + terrain preview block `terrain_preview` = obs[40 : 40+K].

Usage
    from terrain_memory_prosthesis.envs import terrain_preview      # importing registers the gym id (idempotent)
    config env_params.env_id = "myoAssistLegImitationExoPreview-v0"

Registration happens at import time because SB3's SubprocVecEnv workers are fresh processes that only know the ids
registered there: the script that creates the env must import this module at top level (the pilots do).
"""
from __future__ import annotations

import os

import numpy as np

from terrain_memory_prosthesis.envs.osl_physics import NoFlightPenaltyMixin, apply_osl_physics
from terrain_memory_prosthesis.perception.terrain_preview import preview_distances, terrain_preview

PREVIEW_ENV_ID = "myoAssistLegImitationExoPreview-v0"
_ENTRY_POINT = "terrain_memory_prosthesis.envs.terrain_preview:MyoAssistLegImitationExoPreview"


def register():
    """Idempotent: register the env id and teach MyoAssist's EnvironmentHandler about it."""
    from terrain_memory_prosthesis.envs._registry import register_exo_env

    register_exo_env(PREVIEW_ENV_ID, _ENTRY_POINT)


def _make_env_class():
    from rl_train.envs.myoassist_leg_imitation_exo import MyoAssistLegImitationExo

    class MyoAssistLegImitationExoPreview(NoFlightPenaltyMixin, MyoAssistLegImitationExo):
        DEFAULT_OBS_KEYS = MyoAssistLegImitationExo.DEFAULT_OBS_KEYS + ["terrain_preview"]

        def _setup(self, **kwargs):
            self._pv_dist = preview_distances()
            self._pv_zero = os.environ.get("MYO_PREVIEW_ZERO", "0") == "1"
            super()._setup(**kwargs)
            m = self.sim.model
            apply_osl_physics(getattr(m, "ptr", m))  # same prosthesis physics as the Stage-1 env (hard limit + motor model, see envs/osl_physics.py)

        def get_obs_dict(self, sim):
            obs = super().get_obs_dict(sim)
            obs["terrain_preview"] = np.zeros(len(self._pv_dist)) if self._pv_zero else terrain_preview(sim.model, sim.data, self._pv_dist)
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
