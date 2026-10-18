"""Sliding window of the last K observation frames (the controller's memory input for C2).

The recurrent controller (controllers/gru.py) does not keep a hidden state across environment steps. It reads a window of the
K most recent frames that the environment appends to every observation and runs a GRU over that window, from a zero state,
at every step. Consequences, all deliberate:

* PPO needs no change: every sample is self-contained, so SB3's shuffled minibatches, the rollout buffer, evaluation and the
  video tools work exactly as for the feed-forward C1;
* the memory horizon is explicit and tunable (K steps = K / 30 s), which is the interesting knob of the study;
* the cost is that the GRU is re-run over K frames for every sample, and that nothing older than K steps can be remembered.

Frame = the first `frame_width` numbers of the observation (proprioception + the perception block; NOT the privileged block).
The newest frame is the current observation, so the window always ends with "now". At an episode start the window is filled
with the first frame (a standing start, not an empty past).
"""
from __future__ import annotations

import numpy as np


class HistoryBuffer:
    def __init__(self, n_hist: int, frame_width: int):
        if n_hist < 1:
            raise ValueError("n_hist must be >= 1")
        self.n_hist, self.frame_width = int(n_hist), int(frame_width)
        self._buf: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.n_hist * self.frame_width

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Start of an episode: window = K copies of the first frame. Returns obs with the window appended."""
        frame = np.asarray(obs[: self.frame_width], dtype=np.float64)
        self._buf = np.tile(frame, (self.n_hist, 1))
        return self._append(obs)

    def push(self, obs: np.ndarray) -> np.ndarray:
        """One environment step: drop the oldest frame, add the newest. Returns obs with the window appended."""
        if self._buf is None:  # stepping before any reset (e.g. during env construction)
            return self.reset(obs)
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = np.asarray(obs[: self.frame_width], dtype=np.float64)
        return self._append(obs)

    def _append(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(obs, dtype=np.float64), self._buf.ravel()])
