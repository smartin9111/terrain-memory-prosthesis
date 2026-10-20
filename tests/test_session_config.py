import pytest

from terrain_memory_prosthesis.environment.terrain import terrain_spec
from terrain_memory_prosthesis.common.obs_layout import N_PRIVILEGED, N_PROPRIO, ObsLayout
from terrain_memory_prosthesis.common.session_config import PREVIEW_ENV_ID, add_exo_obs_range, make_session_config


def _ranges(d, net):
    return [tuple(o["range"]) for o in d["policy_params"]["custom_policy_params"]["net_indexing_info"][net]["observation"]]


def test_p1_vs_p2(base_config):
    p1 = make_session_config(base_config, "P1", terrain_spec("random", 0.03, 1))
    p2 = make_session_config(base_config, "P2", terrain_spec("random", 0.03, 1))
    assert p1["env_params"]["env_id"] == p2["env_params"]["env_id"] == PREVIEW_ENV_ID
    # Layout: clean 0:40 | sensor values 40:80 | validity 80:120 | perception | privileged.
    assert (120, 125) not in _ranges(p1, "exo_actor") and (120, 125) in _ranges(p2, "exo_actor")
    assert (120, 125) in _ranges(p1, "common_critic") and (125, 130) in _ranges(p2, "common_critic")
    assert all(hi <= 40 for lo, hi in _ranges(p1, "human_actor") + _ranges(p2, "human_actor"))  # human stays clean
    assert _ranges(p1, "human_actor") == _ranges(p2, "human_actor")
    assert all(lo >= 40 and hi <= 80 for lo, hi in _ranges(p1, "exo_actor")[:2])
    assert all(lo >= 80 and hi <= 120 for lo, hi in _ranges(p1, "exo_actor")[2:])


def test_percept_size_follows_perception_model(base_config):
    p2 = make_session_config(base_config, "P2", n_percept=3)  # e.g. 3 preview distances
    assert (120, 123) in _ranges(p2, "exo_actor") and (123, 128) in _ranges(p2, "common_critic")


def test_inconsistent_perception_size(base_config):
    with pytest.raises(ValueError):
        make_session_config(base_config, "P1", n_percept=5)
    with pytest.raises(ValueError):
        make_session_config(base_config, "P2", n_percept=0)


def test_obs_layout():
    lay = ObsLayout(n_percept=3)
    assert lay.sensor_values_range == (40, 80)
    assert lay.sensor_validity_range == (80, 120)
    assert lay.percept_range == (120, 123)
    assert lay.priv_range == (123, 123 + N_PRIVILEGED)
    assert lay.total == 40 + 40 + 40 + 3 + N_PRIVILEGED
    assert ObsLayout(n_percept=0).priv_range == (120, 125)


def test_base_config_not_mutated(base_config):
    before = repr(base_config)
    make_session_config(base_config, "P2")
    add_exo_obs_range(base_config, 1, 2)
    assert repr(base_config) == before


def test_bad_condition(base_config):
    with pytest.raises(ValueError):
        make_session_config(base_config, "P3")


def test_terrain_spec():
    assert terrain_spec("flat") is None
    t = terrain_spec("random", 0.04, 3)
    assert t["terrain"] == "random" and t["seed"] == 3 and t["extent"] >= 40  # upstream default (20 m) is too short
    assert terrain_spec("sinusoidal", 0.02, period=1.5)["period"] == 1.5
    with pytest.raises(ValueError):
        terrain_spec("stairs")
