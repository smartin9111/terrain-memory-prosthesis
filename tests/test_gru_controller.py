"""C2 building blocks that need no MyoAssist: the observation window (environment/history.py), the layout with a history block and
the GRU residual (controllers/gru_module.py)."""
import numpy as np
import pytest
import torch

from terrain_memory_prosthesis.common.obs_layout import ObsLayout
from terrain_memory_prosthesis.controllers.gru_module import HistoryGruResidual, n_percept_from
from terrain_memory_prosthesis.environment.history import HistoryBuffer


# ---------------------------------------------------------------- window
def _obs(t, n=6, extra=3):
    return np.concatenate([np.full(n, float(t)), np.full(extra, -1.0)])  # frame = first n numbers, then a "privileged" tail


def test_reset_fills_the_window_with_the_first_frame():
    h = HistoryBuffer(n_hist=4, frame_width=6)
    out = h.reset(_obs(7))
    assert out.shape == (9 + 4 * 6,) and np.all(out[:9] == _obs(7))
    assert np.all(out[9:] == 7.0)


def test_push_slides_oldest_first_and_ends_with_now():
    h = HistoryBuffer(n_hist=3, frame_width=6)
    h.reset(_obs(0))
    for t in (1, 2, 3, 4):
        out = h.push(_obs(t))
    w = out[9:].reshape(3, 6)
    assert [w[i, 0] for i in range(3)] == [2.0, 3.0, 4.0]  # oldest ... newest = the current observation
    assert np.all(out[:9] == _obs(4))  # the original observation is untouched at the front


def test_window_excludes_the_privileged_tail():
    h = HistoryBuffer(n_hist=2, frame_width=6)
    out = h.reset(_obs(1))
    assert -1.0 not in out[9:]  # the tail (privileged block) is never copied into the history


def test_frame_is_taken_from_frame_start_not_from_the_clean_block():
    # layout of the sensor-imperfection env: [clean 4 | sensor values 3 | validity 3 | privileged 2]; frame = values + validity
    h = HistoryBuffer(n_hist=2, frame_width=6, frame_start=4)
    obs = np.concatenate([np.full(4, 99.0), np.arange(6, dtype=float), np.full(2, -1.0)])
    out = h.reset(obs)
    assert 99.0 not in out[obs.size:] and -1.0 not in out[obs.size:]  # neither the clean block nor the privileged tail
    assert np.all(out[obs.size:].reshape(2, 6) == np.arange(6))
    out = h.push(obs + 100.0)
    assert np.all(out[obs.size:].reshape(2, 6)[-1] == np.arange(6) + 100.0)


def test_push_before_reset_does_not_crash_and_bad_length_is_rejected():
    assert HistoryBuffer(2, 6).push(_obs(3)).shape == (9 + 12,)
    with pytest.raises(ValueError):
        HistoryBuffer(0, 6)


# ---------------------------------------------------------------- layout
def test_layout_with_history_is_appended_last():
    lay = ObsLayout(n_percept=5, n_hist=16)
    assert lay.sensor_values_range == (40, 80) and lay.sensor_validity_range == (80, 120)
    assert lay.percept_range == (120, 125) and lay.priv_range == (125, 130)
    assert lay.frame_start == 40 and lay.frame_width == 85  # sensor values + validity + perception; never the clean 0:40
    assert lay.hist_range == (130, 130 + 16 * 85) and lay.total == 130 + 16 * 85
    assert ObsLayout(n_percept=0).total == 125  # without a window nothing changes (C1)


@pytest.mark.parametrize("n_percept,n_hist", [(0, 16), (5, 16), (3, 1), (5, 32)])
def test_n_percept_is_recovered_from_the_observation_size(n_percept, n_hist):
    assert n_percept_from(ObsLayout(n_percept=n_percept, n_hist=n_hist).total, n_hist) == n_percept


def test_inconsistent_observation_size_is_rejected():
    with pytest.raises(ValueError):
        n_percept_from(ObsLayout(n_percept=5, n_hist=16).total + 1, 16)


# ---------------------------------------------------------------- GRU residual
def _module(n_hist=4, n_percept=0, idx=(0, 1, 5), hidden=8):
    lay = ObsLayout(n_percept=n_percept, n_hist=n_hist)
    return HistoryGruResidual(frame_idx=list(idx), frame_width=lay.frame_width, n_hist=n_hist, offset=lay.hist_range[0], hidden=hidden), lay


def test_starts_at_exactly_zero_for_any_observation():
    torch.manual_seed(0)
    m, lay = _module()
    obs = torch.randn(32, lay.total) * 10
    assert float(m(obs).abs().max()) == 0.0


def test_window_selects_the_requested_inputs_in_time_order():
    m, lay = _module(n_hist=3, idx=(2, 0))
    obs = torch.zeros(1, lay.total)
    a = lay.hist_range[0]
    frames = torch.arange(3 * lay.frame_width, dtype=torch.float32).reshape(3, lay.frame_width)
    obs[0, a:] = frames.reshape(-1)
    w = m.window(obs)[0]
    assert w.shape == (3, 2)
    assert torch.equal(w, frames[:, [2, 0]])


def test_gradient_reaches_the_head_first_then_the_gru():
    torch.manual_seed(0)
    m, lay = _module()
    obs = torch.randn(16, lay.total)
    m(obs).sum().backward()
    assert m.head.weight.grad.abs().sum() > 0 and m.gru.weight_ih_l0.grad.abs().sum() == 0  # delta = W h with W = 0
    with torch.no_grad():
        m.head.weight.normal_()
    m.zero_grad()
    m(obs).sum().backward()
    assert m.gru.weight_ih_l0.grad.abs().sum() > 0


def test_output_depends_on_the_order_of_the_frames_once_trained():
    torch.manual_seed(1)
    m, lay = _module(n_hist=5, hidden=16)
    with torch.no_grad():
        m.head.weight.normal_()
    obs = torch.randn(1, lay.total)
    a = lay.hist_range[0]
    flipped = obs.clone()
    flipped[0, a:] = obs[0, a:].reshape(5, lay.frame_width).flip(0).reshape(-1)
    assert float((m(obs) - m(flipped)).abs().max()) > 1e-6  # memory: the sequence matters, not just the set of frames


def test_blocks_before_the_window_do_not_influence_the_branch():
    torch.manual_seed(2)
    m, lay = _module()
    with torch.no_grad():
        m.head.weight.normal_()
    obs = torch.randn(4, lay.total)
    other = obs.clone()
    other[:, : lay.hist_range[0]] += 5.0  # proprio / perception / privileged "now": only the window feeds the GRU
    assert torch.equal(m(obs), m(other))


def test_bad_frame_index_is_rejected():
    with pytest.raises(ValueError):
        HistoryGruResidual(frame_idx=[0, 99], frame_width=40, n_hist=4, offset=45)
