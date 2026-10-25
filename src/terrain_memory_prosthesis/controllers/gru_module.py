"""The memory branch of C2: a GRU over the observation window, added to the prosthesis actor as a zero-initialised residual.

    pre   = MLP_stage1(x_t)                       (the C1 actor, pre-tanh output)
    delta = Linear( GRU( window of the last K frames ) )          <- this module, starts at exactly 0
    a_t   = tanh(pre + delta)

At step 0 the head is zero, so C2 computes exactly the same torque as C1 (and as Stage 1): any later difference between C2 and C1
comes from what the GRU learned, not from a different start. The same trick as the zero-initialised preview columns of P2
(controllers/warm_start.py). Pure torch, no MyoAssist import, so it can be tested on its own.
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
from torch import nn

from terrain_memory_prosthesis.common.obs_layout import N_PRIVILEGED, N_PROPRIO, N_SENSOR_VALIDITY, N_SENSOR_VALUES

ABLATE_ENV_VAR = "MYO_GRU_ABLATE"
ABLATIONS = ("", "repeat", "shuffle", "zero")


def ablate_window(w: torch.Tensor, mode: str) -> torch.Tensor:
    """History ablation for evaluation: how much does the policy USE the past?  w = (B, K, F), oldest frame first.
    ''        the real window (normal behaviour)
    repeat    every frame = the newest one: the GRU sees the present only, no temporal information, but a realistic input
    shuffle   the K frames in a fixed random order: the same values, the time order destroyed
    zero      all zeros: the window is missing (not a realistic input, an upper bound on the damage)"""
    if mode in ("", "none"):
        return w
    if mode == "repeat":
        return w[:, -1:, :].expand_as(w).contiguous()
    if mode == "shuffle":
        g = torch.Generator().manual_seed(0)
        return w[:, torch.randperm(w.shape[1], generator=g), :]
    if mode == "zero":
        return torch.zeros_like(w)
    raise ValueError(f"unknown {ABLATE_ENV_VAR}={mode!r}; use one of {ABLATIONS}")


def n_percept_from(obs_dim: int, n_hist: int) -> int:
    """Invert obs_dim = 40 + 40 + 40 + n + 5 + K * (40 + 40 + n)   (clean | values | validity | perception | privileged | window)."""
    sensors = N_SENSOR_VALUES + N_SENSOR_VALIDITY
    n, rem = divmod(obs_dim - N_PROPRIO - sensors - N_PRIVILEGED - n_hist * sensors, n_hist + 1)
    if rem or n < 0:
        raise ValueError(f"observation size {obs_dim} does not fit a history window of {n_hist} frames")
    return n


class HistoryGruResidual(nn.Module):
    """obs (B, obs_dim) -> delta (B, n_out). Reads the window obs[:, offset : offset + K * frame_width] (K frames, oldest first)
    and, from each frame, only the entries `frame_idx` (the same inputs the prosthesis MLP itself sees)."""

    def __init__(self, frame_idx: Sequence[int], frame_width: int, n_hist: int, offset: int, hidden: int = 64, n_out: int = 1):
        super().__init__()
        if max(frame_idx) >= frame_width:
            raise ValueError(f"frame index {max(frame_idx)} outside a frame of width {frame_width}")
        self.frame_width, self.n_hist, self.offset = int(frame_width), int(n_hist), int(offset)
        self.ablate = os.environ.get(ABLATE_ENV_VAR, "")   # evaluation only (scripts/gru_ablation.py); empty = the normal window
        self.register_buffer("frame_idx", torch.as_tensor(list(frame_idx), dtype=torch.long), persistent=False)
        self.gru = nn.GRU(len(frame_idx), hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)
        self.zero_head()

    def zero_head(self) -> None:
        """delta == 0 for every input (call again after anything that re-initialises the weights, e.g. SB3's init_weights)."""
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def window(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, K, len(frame_idx)): the selected inputs of the K frames."""
        b = obs.shape[0]
        w = obs[:, self.offset : self.offset + self.n_hist * self.frame_width].reshape(b, self.n_hist, self.frame_width)
        w = w[:, :, self.frame_idx]
        return ablate_window(w, self.ablate)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(self.window(obs).to(self.head.weight.dtype))
        return self.head(h[-1])
