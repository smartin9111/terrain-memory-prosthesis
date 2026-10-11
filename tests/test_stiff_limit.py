import pytest

mujoco = pytest.importorskip("mujoco")

XML = """
<mujoco><option timestep="0.000833"/>
  <worldbody><body><geom size="0.1"/>
    <body pos="0 0 1"><joint name="OSL_A_L1_osl_ankle_angle_r" axis="0 1 0" range="-30 30" limited="true"/><geom size="0.05"/></body>
    <body pos="0 0 2"><joint name="knee_angle_r" axis="0 1 0" range="-30 30" limited="true"/><geom size="0.05"/></body>
    <body pos="0 0 3"><joint name="OSL_free" axis="0 1 0"/><geom size="0.05"/></body>
  </body></worldbody>
</mujoco>
"""


def test_only_limited_prosthesis_joints_are_hardened():
    from terrain_memory_prosthesis.envs.stiff_limit import LIMIT_TIMECONST, harden_device_limits

    m = mujoco.MjModel.from_xml_string(XML)
    before = m.jnt_solref.copy()
    names = harden_device_limits(m)
    assert names == ["OSL_A_L1_osl_ankle_angle_r"]
    assert m.jnt_solref[0][0] == pytest.approx(max(LIMIT_TIMECONST, 2 * m.opt.timestep))
    assert (m.jnt_solref[1] == before[1]).all() and (m.jnt_solref[2] == before[2]).all()
