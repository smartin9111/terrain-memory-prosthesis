import pytest

from terrain_memory_prosthesis.envs.terrain import terrain_spec
from terrain_memory_prosthesis.utils.session_config import PREVIEW_ENV_ID, add_exo_obs_range, make_session_config


def _ranges(d, net):
    return [tuple(o["range"]) for o in d["policy_params"]["custom_policy_params"]["net_indexing_info"][net]["observation"]]


def test_p1_vs_p2(base_config):
    p1 = make_session_config(base_config, "P1", terrain_spec("random", 0.03, 1))
    p2 = make_session_config(base_config, "P2", terrain_spec("random", 0.03, 1))
    assert p1["env_params"]["env_id"] == p2["env_params"]["env_id"] == PREVIEW_ENV_ID
    assert (40, 45) not in _ranges(p1, "exo_actor") and (40, 45) in _ranges(p2, "exo_actor")
    assert (40, 45) in _ranges(p1, "common_critic") and (40, 45) in _ranges(p2, "common_critic")  # same critic input
    assert (40, 45) not in _ranges(p1, "human_actor") and (40, 45) not in _ranges(p2, "human_actor")  # human stays blind
    assert _ranges(p1, "human_actor") == _ranges(p2, "human_actor")


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
