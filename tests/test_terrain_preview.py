import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from terrain_memory_prosthesis.perception.clean_preview import ground_height, terrain_preview

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <geom name="step" type="box" pos="1.0 0 0.05" size="0.3 1 0.05"/>   <!-- world geom: a 10 cm step at x=0.7..1.3 -->
    <body name="pelvis" pos="0 0 1.0">
      <freejoint/>
      <geom name="torso" type="box" size="0.1 0.1 0.1"/>                 <!-- walker geom: must be skipped -->
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture()
def sim():
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_ground_height_skips_walker_geoms(sim):
    m, d = sim
    assert ground_height(m, d, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)  # below the pelvis box: floor, not the torso
    assert ground_height(m, d, 1.0, 0.0) == pytest.approx(0.1, abs=1e-6)  # on the step


def test_preview_is_relative_and_scaled(sim):
    m, d = sim
    v = terrain_preview(m, d, distances=(0.2, 1.0), scale=0.1)
    assert np.allclose(v, [0.0, 1.0], atol=1e-5)  # 10 cm step / 0.1 scale = 1.0


def test_preview_beyond_field_is_zero(sim):
    m, d = sim
    v = terrain_preview(m, d, distances=(100.0,), scale=0.1)
    assert v[0] == 0.0


def test_many_stacked_walker_geoms_do_not_hide_the_ground():
    """Regression: at reset >12 walker geoms sit above the pelvis x,y; the old hop-based ray returned NaN -> preview 0."""
    geoms = "".join(f'<geom type="sphere" size="0.05" pos="0 0 {0.1 * i}"/>' for i in range(30))
    xml = XML.replace("</worldbody>", f'<body name="stack" pos="0 0 0.5"><freejoint/>{geoms}</body></worldbody>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert ground_height(m, d, 0.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(terrain_preview(m, d, distances=(0.2, 1.0), scale=0.1), [0.0, 1.0], atol=1e-5)


def test_ground_height_on_heightfield():
    xml = """<mujoco><asset><hfield name="h" nrow="3" ncol="3" size="2 2 0.5 0.1"
        elevation="0 0 0  0 1 0  0 0 0"/></asset>
      <worldbody><geom type="hfield" hfield="h"/></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert ground_height(m, d, 0.0, 0.0) == pytest.approx(0.5, abs=1e-3)  # centre cell is the peak (z size 0.5)
    assert ground_height(m, d, 1.9, 0.0) == pytest.approx(0.0, abs=0.05)  # edge node is flat (cell interpolation)
