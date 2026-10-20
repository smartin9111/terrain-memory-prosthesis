"""Reproducible sensor imperfections for fixed-size proprioceptive vectors.

This module is deliberately independent of MuJoCo, MyoAssist, and controller code.
It models imperfections on an already assembled observation vector; environment or
policy integration decides where the corrupted vector is routed.

Processing order per control step:
    clean measurement -> additive Gaussian noise -> fixed-step delay -> dropout

Dropout and delay warm-up are represented as zero-valued measurements plus a
validity mask. The caller should expose ``values`` and ``validity`` as separate
feature blocks when the controller needs to distinguish a real zero from a missing
measurement. Random draws use only the RNG initialized by ``reset(seed)``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorSample:
    """Corrupted values and per-channel validity (1 = valid, 0 = unavailable)."""

    values: np.ndarray
    validity: np.ndarray


class ProprioceptiveSensorModel:
    """Apply configurable noise, delay, and dropout to a numeric observation vector.

    Args:
        size: Number of input channels; output channel count never changes.
        noise_std: Gaussian standard deviation in each channel's native units. A
            scalar applies to every channel; a sequence must have ``size`` entries.
        delay_steps: Number of control steps by which measurements are delayed.
        dropout_probability: Independent per-channel probability of a dropout on
            each step. A scalar applies to every channel; a sequence must have
            ``size`` entries.

    Invalid configuration and non-finite input values raise ``ValueError`` rather
    than silently contaminating the controller input.
    """

    def __init__(self, size: int, noise_std: float | tuple | list | np.ndarray = 0.0,
                 delay_steps: int = 0,
                 dropout_probability: float | tuple | list | np.ndarray = 0.0):
        self._size = int(size)
        if self._size <= 0:
            raise ValueError("size must be positive")
        self.noise_std = self._channel_parameter(noise_std, "noise_std")
        self.dropout_probability = self._channel_parameter(dropout_probability, "dropout_probability")
        if np.any(self.noise_std < 0):
            raise ValueError("noise_std must be non-negative")
        if np.any((self.dropout_probability < 0) | (self.dropout_probability > 1)):
            raise ValueError("dropout_probability must be in [0, 1]")
        if isinstance(delay_steps, bool) or int(delay_steps) != delay_steps or int(delay_steps) < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        self.delay_steps = int(delay_steps)
        self.reset()

    @property
    def size(self) -> int:
        """Number of measurement channels (not counting validity-mask channels)."""
        return self._size

    @property
    def output_size(self) -> int:
        """Feature count when values and validity are concatenated for a policy."""
        return 2 * self._size

    def _channel_parameter(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self._size, float(arr), dtype=np.float64)
        if arr.shape != (self._size,):
            raise ValueError(f"{name} must be scalar or have shape ({self._size},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr.copy()

    def reset(self, seed: int | None = None) -> None:
        """Reset random state and delay history at the start of an episode."""
        self._rng = np.random.default_rng(seed)
        # Missing startup history is explicitly invalid, never mistaken for a true zero.
        self._history: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)
        self._history_valid: deque[bool] = deque(maxlen=self.delay_steps + 1)

    def observe(self, clean_values) -> SensorSample:
        """Return the noisy/delayed/dropout-corrupted measurement and validity mask."""
        x = np.asarray(clean_values, dtype=np.float64)
        if x.shape != (self._size,):
            raise ValueError(f"expected observation shape ({self._size},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("clean observation must contain only finite values")

        noisy = x + self._rng.normal(loc=0.0, scale=self.noise_std, size=self._size)
        self._history.append(noisy.copy())
        self._history_valid.append(True)

        if len(self._history) <= self.delay_steps:
            delayed = np.zeros(self._size, dtype=np.float64)
            startup_valid = False
        else:
            delayed = self._history[0].copy()
            startup_valid = self._history_valid[0]

        keep = self._rng.random(self._size) >= self.dropout_probability
        validity = keep.astype(np.float64) if startup_valid else np.zeros(self._size, dtype=np.float64)
        values = np.where(keep & startup_valid, delayed, 0.0)
        return SensorSample(values=values, validity=validity)

    def observe_vector(self, clean_values) -> np.ndarray:
        """Convenience API returning ``[corrupted values | validity mask]``."""
        sample = self.observe(clean_values)
        return np.concatenate((sample.values, sample.validity)).astype(np.float32, copy=False)
