import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from terrain_memory_prosthesis.environment.episode_terrain import EpisodeTerrain
from terrain_memory_prosthesis.environment.terrain import episodic_terrain, parse_episodic
from terrain_memory_prosthesis.perception.clean_preview import ground_height

XML = """<mujoco><asset><hfield name="h" nrow="40" ncol="40" size="8 8 0.06 0.1"/></asset>
<worldbody><geom type="hfield" hfield="h"/></worldbody></mujoco>"""


def _model():
    m = mujoco.MjModel.from_xml_string(XML)
    return m, mujoco.MjData(m)


def test_resample_amplitude_range_new_field_and_safe_zone():
    m, d = _model()
    et = EpisodeTerrain(m, 0.02, 0.06, safe_radius=2.0, rng=np.random.default_rng(0))
    fields, amps = [], []
    for _ in range(5):
        amps.append(et.resample(m))
        n = et.nrow * et.ncol
        h = m.hfield_data[et.adr:et.adr + n] * et.size_z  # physical heights [m]
        fields.append(h.copy())
        assert h.max() <= amps[-1] + 1e-9 and h.min() >= 0
        assert h.reshape(et.nrow, et.ncol)[et.nrow // 2, et.ncol // 2] < 0.1 * amps[-1]  # (nearly) flat start zone
    assert all(0.02 <= a <= 0.06 for a in amps)
    assert len({round(a, 6) for a in amps}) == 5 and not np.allclose(fields[0], fields[1])


def test_written_heights_are_what_the_preview_ray_sees():
    m, d = _model()
    et = EpisodeTerrain(m, 0.05, 0.05, safe_radius=0.0, rng=np.random.default_rng(1))
    et.resample(m)
    mujoco.mj_forward(m, d)
    hs = [ground_height(m, d, x, 0.0) for x in np.linspace(-4, 4, 17)]
    assert 0.0 <= min(hs) and max(hs) <= 0.05 + 1e-6 and max(hs) > 0.01  # rough, bounded by the amplitude


def test_amp_hi_above_compiled_relief_is_rejected():
    m, _ = _model()
    with pytest.raises(ValueError):
        EpisodeTerrain(m, 0.02, 0.08)


def test_episodic_terrain_dict_roundtrip():
    t = episodic_terrain(0.02, 0.06)
    assert parse_episodic(t) == (0.02, 0.06) and t["amplitude"] == 0.06 and t["extent"] == 160.0 and t["resolution"] == 640
    assert parse_episodic({"terrain_name": "rnd_0.03_1"}) is None and parse_episodic(None) is None
    with pytest.raises(ValueError):
        episodic_terrain(0.06, 0.02)
