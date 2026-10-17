"""Name -> perception model. Add a variant in your own file with `@register("name")`; nothing else has to change."""
from __future__ import annotations

_REGISTRY: dict = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def available() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def build(name: str, **kwargs):
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown perception model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def _load_builtin():
    from terrain_memory_prosthesis.perception import clean_preview, none  # noqa: F401  (registers)
