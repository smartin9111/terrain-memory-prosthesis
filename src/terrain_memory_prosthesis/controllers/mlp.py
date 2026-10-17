"""C1: feed-forward prosthesis controller = the Stage-1 prosthesis MLP (zero-extended inputs), human frozen."""
from __future__ import annotations

from terrain_memory_prosthesis.controllers.registry import register
from terrain_memory_prosthesis.controllers.warm_start import warm_start


@register("mlp")
def prepare(model, stage1_model, n_muscle: int = 18) -> dict:
    return warm_start(model, stage1_model, n_muscle=n_muscle)
