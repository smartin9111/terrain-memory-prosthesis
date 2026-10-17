"""Stage-1/2 env with a realistic OSL ankle (hard limit + motor torque/speed model).

Why (measured): in the stock model the OSL ankle joint (a) has a soft +-30 deg limit that a 168 Nm motor pushes through
(+110 / -300 deg), and (b) has NO speed limit and NO reflected motor inertia, so a 0.0014 kg m^2 foot is snapped at 55 rad/s;
the trained gait becomes bang-bang (80 % of the time at the stops, see scripts/check_ankle_limit.py test B).

Real Open Source Leg (https://opensourceleg.org/hardware, ankle = knee drivetrain): continuous ~29 Nm, peak ~145.25 Nm,
41.5:1 reduction (9:1 planetary x 4.61:1 belt), peak speed 6.13 rad/s, ankle range -30..30 deg.

Model (applied after the model is built, the MyoAssist files are untouched):
  * joint limit: solref time constant 0.002 s (hard stop, see stiff_limit.py)
  * motor torque/speed line:  tau = K*u - b*omega   with K*u_max = PEAK_TORQUE and free speed = PEAK_SPEED at full command
        gain K = PEAK_TORQUE / ctrl_max, velocity damping b = PEAK_TORQUE / PEAK_SPEED  (back-EMF / viscous line)
  * reflected rotor inertia (armature) ARMATURE: REQUIRED for stability (explicit Euler integrator: b*dt/J >> 2 with the bare foot
    inertia) and physically there (J_rotor * 41.5^2). Its value is NOT in the OSL documentation: ASSUMPTION 0.1 kg m^2
    (rotor ~6e-5 kg m^2 x 41.5^2); results should be checked for sensitivity (0.05 .. 0.2).

Walking constraint (reward, measured need): the Stage-1 policy learned a hopping/running gait (23 % of the steps with no foot on the
ground, 19 cm pelvis bounce, 1.5 m/s vs the 1.25 m/s target) that survived a 10x higher pelvis_ty imitation weight, because that term is
exp(-8 d^2) with d in METRES (a 10 cm error costs only 8 %). Walking by definition has no flight phase, so the env adds
`-dt` for every control step in which both feet (r_foot+r_toes, l_foot+l_toes touch sensors) carry < FLIGHT_FORCE_N. It is folded
into the existing `foot_force_penalty` term, so its weight is `reward_keys_and_weights.foot_force_penalty` in the config
(that weight also scales the impact term max(0, F/W - 1.2)).

Usage: config env_params.env_id = "myoAssistLegImitationExoOSLPhys-v0".
"""
from __future__ import annotations

OSL_PHYS_ENV_ID = "myoAssistLegImitationExoOSLPhys-v0"
_ENTRY_POINT = "terrain_memory_prosthesis.environment.osl_physics:MyoAssistLegImitationExoOSLPhys"
PEAK_TORQUE = 145.25  # Nm (OSL spec page, instantaneous)
PEAK_SPEED = 6.13  # rad/s (OSL spec page)
FLIGHT_FORCE_N = 1.0  # both feet below this total touch force = flight phase (same threshold as scripts/pilots/05_render_video.py)
ARMATURE = 0.1  # kg m^2 reflected rotor inertia (assumption, see module doc)
ANKLE_JOINT_SUFFIX = "osl_ankle_angle_r"
ANKLE_ACTUATOR_SUFFIX = "osl_ankle_torque_actuator"


def apply_osl_motor_model(mj_model, peak_torque: float = PEAK_TORQUE, peak_speed: float = PEAK_SPEED, armature: float = ARMATURE) -> dict:
    """Set gain / velocity-damping / armature of the OSL ankle actuator + joint in place. Returns what was set."""
    import mujoco

    act = next(i for i in range(mj_model.nu) if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").lower().endswith(ANKLE_ACTUATOR_SUFFIX))
    jnt = next(i for i in range(mj_model.njnt) if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").lower().endswith(ANKLE_JOINT_SUFFIX))
    ctrl_max = float(max(abs(mj_model.actuator_ctrlrange[act][0]), abs(mj_model.actuator_ctrlrange[act][1])))
    gain = peak_torque / ctrl_max
    b = peak_torque / peak_speed
    mj_model.actuator_gainprm[act][0] = gain
    mj_model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
    mj_model.actuator_biasprm[act][:] = 0.0
    mj_model.actuator_biasprm[act][2] = -b
    mj_model.dof_armature[mj_model.jnt_dofadr[jnt]] = armature
    mujoco.mj_setConst(mj_model, mujoco.MjData(mj_model))  # REQUIRED: MuJoCo caches the armature-dependent constants; without this the armature is ignored
    return {"gain": gain, "damping": b, "armature": armature, "peak_torque": gain * ctrl_max}


def apply_osl_physics(mj_model) -> dict:
    from terrain_memory_prosthesis.environment.stiff_limit import harden_device_limits

    limits = harden_device_limits(mj_model)
    motor = apply_osl_motor_model(mj_model)
    return {"limits": limits, **motor}


def register():
    from terrain_memory_prosthesis.environment._registry import register_exo_env

    register_exo_env(OSL_PHYS_ENV_ID, _ENTRY_POINT)


class NoFlightPenaltyMixin:
    """Adds -dt per control step without any foot contact to the `foot_force_penalty` reward term (weight: see module doc).

    Hooks `_calculate_imitation_rewards` (not `_calculate_base_reward`): the imitation env calls `super()._calculate_base_reward`
    directly, so an override of the latter would never run for the imitation envs used here.
    """

    def _calculate_imitation_rewards(self, obs_dict):
        rewards, info = super()._calculate_imitation_rewards(obs_dict)
        total = abs(float(self._get_foot_force("r"))) + abs(float(self._get_foot_force("l")))
        in_flight = total < FLIGHT_FORCE_N
        if in_flight:
            rewards["foot_force_penalty"] = rewards["foot_force_penalty"] - self.dt
        info = dict(info)
        info["flight"] = bool(in_flight)
        return rewards, info


def _make_env_class():
    from rl_train.envs.myoassist_leg_imitation_exo import MyoAssistLegImitationExo

    class MyoAssistLegImitationExoOSLPhys(NoFlightPenaltyMixin, MyoAssistLegImitationExo):
        def _setup(self, **kwargs):
            super()._setup(**kwargs)
            m = self.sim.model
            self._osl_physics = apply_osl_physics(getattr(m, "ptr", m))

    return MyoAssistLegImitationExoOSLPhys


def __getattr__(name):  # lazy: gym imports "<module>:MyoAssistLegImitationExoOSLPhys"
    if name == "MyoAssistLegImitationExoOSLPhys":
        cls = _make_env_class()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


try:  # register at import (needed in subprocess workers)
    register()
except Exception:  # pragma: no cover
    pass
