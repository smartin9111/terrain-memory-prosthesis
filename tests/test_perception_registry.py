import numpy as np
import pytest

from terrain_memory_prosthesis.common.interfaces import PerceptionModel
from terrain_memory_prosthesis.common.obs_layout import N_PRIVILEGED
from terrain_memory_prosthesis.perception import registry
from terrain_memory_prosthesis.perception.clean_preview import DEFAULT_DISTANCES


def test_builtin_models_registered():
    assert {"none", "clean_preview"} <= set(registry.available())


def test_none_is_empty_block():
    m = registry.build("none")
    assert isinstance(m, PerceptionModel) and m.size == 0 and m.name == "none"


def test_clean_preview_follows_distances():
    m = registry.build("clean_preview", distances=(0.2, 0.6))
    assert isinstance(m, PerceptionModel) and m.size == 2 and m.name == "clean_preview"


def test_privileged_block_matches_default_preview():
    # the critic's privileged block is the default clean preview: layout and default distances must agree
    assert len(DEFAULT_DISTANCES) == N_PRIVILEGED


def test_unknown_name():
    with pytest.raises(KeyError):
        registry.build("does_not_exist")


def test_env_var_selects_distances(monkeypatch):
    monkeypatch.setenv("MYO_PREVIEW_DISTANCES", "0.2,0.4,0.6")
    assert registry.build("clean_preview").size == 3
    monkeypatch.delenv("MYO_PREVIEW_DISTANCES")
    assert registry.build("clean_preview").size == N_PRIVILEGED
