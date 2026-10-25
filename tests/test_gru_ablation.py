import pytest
import torch

from terrain_memory_prosthesis.controllers.gru_module import HistoryGruResidual, ablate_window


def _w():
    torch.manual_seed(0)
    return torch.randn(3, 5, 4)   # B, K, F; frame 4 is the newest


def test_no_ablation_is_the_identity():
    w = _w()
    assert torch.equal(ablate_window(w, ""), w) and torch.equal(ablate_window(w, "none"), w)


def test_repeat_copies_the_newest_frame_everywhere():
    w = _w()
    r = ablate_window(w, "repeat")
    assert r.shape == w.shape and all(torch.equal(r[:, k], w[:, -1]) for k in range(w.shape[1]))


def test_shuffle_keeps_the_values_but_changes_the_order():
    w = _w()
    s = ablate_window(w, "shuffle")
    assert not torch.equal(s, w) and torch.equal(s.sort(dim=1).values, w.sort(dim=1).values)
    assert torch.equal(s, ablate_window(w, "shuffle"))   # fixed order: the same damage every call


def test_zero_and_unknown_mode():
    assert ablate_window(_w(), "zero").abs().sum() == 0
    with pytest.raises(ValueError):
        ablate_window(_w(), "nonsense")


def test_module_reads_the_mode_from_the_environment(monkeypatch):
    monkeypatch.setenv("MYO_GRU_ABLATE", "repeat")
    m = HistoryGruResidual(frame_idx=[0, 1], frame_width=3, n_hist=4, offset=0, hidden=8, n_out=1)
    obs = torch.randn(2, 12)
    w = m.window(obs)
    assert all(torch.equal(w[:, k], w[:, -1]) for k in range(4))
    monkeypatch.delenv("MYO_GRU_ABLATE")
    assert HistoryGruResidual(frame_idx=[0, 1], frame_width=3, n_hist=4, offset=0, hidden=8, n_out=1).ablate == ""
