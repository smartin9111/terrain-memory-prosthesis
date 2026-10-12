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

    from terrain_memory_prosthesis.envs.osl_physics import PEAK_SPEED, PEAK_TORQUE, apply_osl_physics

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
