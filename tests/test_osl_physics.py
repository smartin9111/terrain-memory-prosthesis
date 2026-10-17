import pytest

mujoco = pytest.importorskip("mujoco")

XML = """
<mujoco><option timestep="0.000833" integrator="Euler"/>
  <worldbody><body><geom size="0.1"/>
    <body pos="0 0 1"><joint name="OSL_A_L1_osl_ankle_angle_r" axis="0 1 0" range="-30 30" limited="true"/><geom size="0.05" mass="0.3"/></body>
  </body></worldbody>
  <actuator><general name="OSL_A_L1_osl_ankle_torque_actuator" joint="OSL_A_L1_osl_ankle_angle_r" gainprm="58.4" ctrlrange="-2.88 2.88" ctrllimited="true"/></actuator>
</mujoco>
"""


def test_motor_model_matches_spec_and_is_stable():
    import numpy as np

    from terrain_memory_prosthesis.environment.osl_physics import PEAK_SPEED, PEAK_TORQUE, apply_osl_physics

    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    info = apply_osl_physics(m)
    assert info["peak_torque"] == pytest.approx(PEAK_TORQUE)
    assert info["limits"] == ["OSL_A_L1_osl_ankle_angle_r"]
    m.opt.gravity[:] = 0
    d.ctrl[0] = 2.88
    w = 0.0
    for _ in range(2000):
        mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all()
        w = max(w, abs(d.qvel[0]))
    assert w <= PEAK_SPEED * 1.1  # free speed at full command = spec speed (never exceeded)
    assert abs(np.degrees(d.qpos[0])) <= 33.0  # hard stop holds


def test_no_flight_penalty_only_without_foot_contact():
    """The walking term: -dt in `foot_force_penalty` exactly when both feet carry < FLIGHT_FORCE_N (no env needed)."""
    from terrain_memory_prosthesis.environment.osl_physics import FLIGHT_FORCE_N, NoFlightPenaltyMixin

    class Base:
        dt = 1.0 / 30.0

        def _calculate_imitation_rewards(self, obs_dict):
            return {"foot_force_penalty": -0.01}, {}

    class Env(NoFlightPenaltyMixin, Base):
        def __init__(self, r, l):
            self.f = {"r": r, "l": l}

        def _get_foot_force(self, side):
            return self.f[side]

    stance, info = Env(300.0, 0.0)._calculate_imitation_rewards(None)
    assert stance["foot_force_penalty"] == pytest.approx(-0.01) and info["flight"] is False
    flight, info = Env(0.0, 0.0)._calculate_imitation_rewards(None)
    assert flight["foot_force_penalty"] == pytest.approx(-0.01 - Base.dt) and info["flight"] is True
    near, _ = Env(0.4 * FLIGHT_FORCE_N, 0.4 * FLIGHT_FORCE_N)._calculate_imitation_rewards(None)  # 0.8 N total: still flight
    assert near["foot_force_penalty"] == pytest.approx(-0.01 - Base.dt)
    touch, _ = Env(0.6 * FLIGHT_FORCE_N, 0.6 * FLIGHT_FORCE_N)._calculate_imitation_rewards(None)  # 1.2 N total: contact
    assert touch["foot_force_penalty"] == pytest.approx(-0.01)
