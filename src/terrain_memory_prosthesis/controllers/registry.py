"""Name -> controller recipe. A recipe turns (new model, Stage-1 model) into the Stage-2 start state.

C1 = "mlp": the Stage-1 prosthesis MLP, zero-extended for the extra inputs (controllers/warm_start.py).
C2 = "gru": TODO (controller owner). Register it in controllers/gru.py with `@register("gru")`; nothing else has to change.
"""
from __future__ import annotations

_REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def apply(name: str, model, stage1_model, **kwargs) -> dict:
    """Prepare `model` for Stage 2 with controller `name` (human frozen, prosthesis controller initialised)."""
    _load_builtin()
    if name not in _REGISTRY:
        raise NotImplementedError(f"controller {name!r} is not implemented; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](model, stage1_model, **kwargs)


def _load_builtin():
    from terrain_memory_prosthesis.controllers import mlp  # noqa: F401  (registers "mlp")
