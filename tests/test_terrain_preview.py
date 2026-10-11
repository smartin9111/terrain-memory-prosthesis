import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from terrain_memory_prosthesis.perception.terrain_preview import ground_height, terrain_preview

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
