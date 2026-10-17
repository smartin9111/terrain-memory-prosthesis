"""P1: no exteroception (the prosthesis actor sees only the proprioceptive observation)."""
from __future__ import annotations

import numpy as np

from terrain_memory_prosthesis.perception.registry import register


@register("none")
class NoPerception:
    name = "none"

    @property
    def size(self) -> int:
        return 0

    def reset(self, seed=None) -> None:
        pass

    def observe(self, model, data) -> np.ndarray:
        return np.zeros(0)
