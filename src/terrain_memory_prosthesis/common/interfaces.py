"""Interfaces. A perception model turns the simulator state into a fixed-size vector for the prosthesis actor.

Rules every perception model must follow (they keep the C1-vs-C2 comparison fair):
1. `size` is fixed for the lifetime of the object (the controller input size is derived from it).
2. No hidden memory a controller could exploit (no smoothing/filter state). A pure delay buffer is a sensor property and allowed
   (identical for every controller). `reset(seed)` clears it at every episode start.
3. Dropout = zeros + a validity bit (never "repeat the last value").
4. Randomness only from the RNG seeded in `reset(seed)`, so the same seed gives every controller the same disturbances.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PerceptionModel(Protocol):
    name: str

    @property
    def size(self) -> int: ...

    def reset(self, seed: int | None = None) -> None:
        """Called by the environment at every episode reset."""

    def observe(self, model, data) -> np.ndarray:
        """Called every control step with the MuJoCo model/data; returns a float vector of length `size`."""
