"""Name -> controller recipe. A recipe turns (new model, Stage-1 model) into the Stage-2 start state.

C1 = "mlp": the Stage-1 prosthesis MLP, zero-extended for the extra inputs (controllers/warm_start.py).
C2 = "gru": the same MLP plus a zero-initialised GRU branch that reads a window of the last K observation frames
           (controllers/gru.py, controllers/gru_module.py, environment/history.py).

A controller module registers itself with `@register("name")`; it may also provide
  * `build_model=fn(cfg, env) -> PPO`   when it needs its own policy class (C2 does; C1 uses MyoAssist's default),
  * `env_vars=fn(args) -> dict`         environment variables that define the controller's setting (stored in stage2_info.json
                                        and re-applied by every evaluation / video tool, like the perception variables).
Modules are imported only when asked for, so listing the names does not need MyoAssist.
"""
from __future__ import annotations

import importlib

_MODULES = {"mlp": "terrain_memory_prosthesis.controllers.mlp", "gru": "terrain_memory_prosthesis.controllers.gru"}
_REGISTRY: dict = {}
_BUILDERS: dict = {}
_ENV: dict = {}


def register(name: str, build_model=None, env_vars=None):
    def deco(fn):
        _REGISTRY[name] = fn
        if build_model is not None:
            _BUILDERS[name] = build_model
        if env_vars is not None:
            _ENV[name] = env_vars
        return fn

    return deco


def available() -> list[str]:
    return sorted(set(_MODULES) | set(_REGISTRY))


def _load(name: str) -> None:
    if name not in _REGISTRY and name in _MODULES:
        importlib.import_module(_MODULES[name])


def apply(name: str, model, stage1_model, **kwargs) -> dict:
    """Prepare `model` for Stage 2 with controller `name` (human frozen, prosthesis controller initialised)."""
    _load(name)
    if name not in _REGISTRY:
        raise NotImplementedError(f"controller {name!r} is not implemented; available: {available()}")
    return _REGISTRY[name](model, stage1_model, **kwargs)


def build_model(name: str, cfg, env):
    """The PPO model the controller needs, or None if MyoAssist's default model (C1) is the right one."""
    _load(name)
    return _BUILDERS[name](cfg, env) if name in _BUILDERS else None


def env_vars(name: str, args) -> dict:
    """Environment variables that define this controller's setting for the env and the policy (empty for C1)."""
    _load(name)
    return _ENV[name](args) if name in _ENV else {}
