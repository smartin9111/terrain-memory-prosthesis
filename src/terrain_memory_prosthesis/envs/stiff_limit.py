"""Stage-1 env with a HARD prosthesis joint limit (fixes the soft OSL ankle limit).

Problem (measured): in the stock model the OSL ankle has range +-30 deg, but the limit constraint is soft (solref
timeconst 0.02 s) while the foot assembly is very light (I ~ 0.0014 kg m^2). The 2.88 Nm motor then pushes the joint
to +110 / -300 deg (and a trained policy exploits it: the foot swings almost all the way round). Setting the limit's
solref timeconst to ~2 physics steps (0.002 s) makes the limit hold (overshoot ~1 deg at full torque).

Usage: config env_params.env_id = "myoAssistLegImitationExoStiff-v0" (same obs/actions/rewards as the stock exo env).
The preview env (envs/terrain_preview.py) applies the same fix, so Stage 1 and Stage 2 share one physics.
"""
from __future__ import annotations

STIFF_ENV_ID = "myoAssistLegImitationExoStiff-v0"
_ENTRY_POINT = "terrain_memory_prosthesis.envs.stiff_limit:MyoAssistLegImitationExoStiff"
LIMIT_TIMECONST = 0.002  # s; must be >= 2 * physics timestep (1/1200 s -> 0.0017)
DEVICE_JOINT_PREFIX = "OSL"


def harden_device_limits(mj_model, timeconst: float = LIMIT_TIMECONST, prefix: str = DEVICE_JOINT_PREFIX) -> list[str]:
    """Stiffen the limit constraint of every limited prosthesis joint (name starts with `prefix`). Returns their names."""
    import mujoco

    done = []
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.upper().startswith(prefix) and mj_model.jnt_limited[j]:
            mj_model.jnt_solref[j] = [max(timeconst, 2.0 * mj_model.opt.timestep), 1.0]
            done.append(name)
    return done


def register():
    from terrain_memory_prosthesis.envs._registry import register_exo_env

    register_exo_env(STIFF_ENV_ID, _ENTRY_POINT)


def _make_env_class():
    from rl_train.envs.myoassist_leg_imitation_exo import MyoAssistLegImitationExo

    class MyoAssistLegImitationExoStiff(MyoAssistLegImitationExo):
        def _setup(self, **kwargs):
            super()._setup(**kwargs)
            m = self.sim.model
            self._hardened = harden_device_limits(getattr(m, "ptr", m))

    return MyoAssistLegImitationExoStiff


def __getattr__(name):  # lazy: gym imports "<module>:MyoAssistLegImitationExoStiff"
    if name == "MyoAssistLegImitationExoStiff":
        cls = _make_env_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


try:  # register at import (needed in subprocess workers)
    register()
except Exception:  # pragma: no cover
    pass
