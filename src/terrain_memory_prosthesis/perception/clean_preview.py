"""P2 exteroception: terrain height preview ahead of the walker (pure MuJoCo, no MyoAssist import). Owner: perception.

preview(model, data, distances, scale) -> K values
    h_k = (ground height at pelvis_x + d_k) - (ground height under the pelvis), divided by `scale` (default 0.1 m),
    look-ahead distances d_k along world +x (the walking direction); beyond the edge of the terrain -> 0.

Ground height: a downward ray from z = 2 m against the WORLD-body geoms only (hfield: mj_rayHfield, primitives:
mju_rayGeom), so the walker's own geoms can never hide the ground. Cost: a few microseconds per ray.

Environment variables (read at env construction, therefore inherited by subprocess workers):
    MYO_PREVIEW_DISTANCES  comma separated metres, default "0.2,0.4,0.6,0.8,1.0"
    MYO_PREVIEW_SCALE      default 0.1
    MYO_PREVIEW_ZERO       "1" -> the block is always zeros (ablation: same obs size, no information)

TODO (perception owner): noise / dropout / delay of the preview. Rule for a fair C1-vs-C2 comparison: the perception
must not carry hidden memory; a dropout is zeros + a validity bit, the same delay for both controllers.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_DISTANCES = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_SCALE = 0.1
_DOWN = np.array([0.0, 0.0, -1.0])


def preview_distances() -> tuple:
    s = os.environ.get("MYO_PREVIEW_DISTANCES")
    return tuple(float(x) for x in s.split(",") if x.strip()) if s else DEFAULT_DISTANCES


def ground_height(model, data, x: float, y: float, z_top: float = 2.0, max_hops: int = 12) -> float:
    """Height [m] of the highest WORLD geom (terrain/plane/box) below (x, y, z_top); nan if none.

    Only geoms of the world body (id 0) are intersected - hfield via mj_rayHfield, primitives via mju_rayGeom - so the
    walker's own geoms (dozens of them stack up above the pelvis at reset) can never hide the ground. (A first version
    hopped over walker hits with mj_ray and returned NaN once more than `max_hops` geoms were in the way; `max_hops`
    is kept for API compatibility and ignored.)
    """
    import mujoco

    model, data = getattr(model, "ptr", model), getattr(data, "ptr", data)  # dm_control wrappers -> raw mujoco
    origin = np.array([x, y, z_top], dtype=float)
    best = -1.0
    for g in np.flatnonzero(model.geom_bodyid == 0):
        gt = int(model.geom_type[g])
        if gt == int(mujoco.mjtGeom.mjGEOM_HFIELD):
            t = mujoco.mj_rayHfield(model, data, int(g), origin, _DOWN)
        elif gt == int(mujoco.mjtGeom.mjGEOM_MESH):
            continue  # not used by the terrains; mju_rayGeom does not support meshes
        else:
            t = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], origin, _DOWN, gt)
        if t >= 0 and (best < 0 or t < best):
            best = t
    return float("nan") if best < 0 else origin[2] - best


def terrain_preview(model, data, distances=None, scale: float | None = None, body: str = "pelvis") -> np.ndarray:
    distances = distances if distances is not None else preview_distances()
    scale = scale if scale is not None else float(os.environ.get("MYO_PREVIEW_SCALE", DEFAULT_SCALE))
    p = data.body(body).xpos
    h0 = ground_height(model, data, p[0], p[1])
    vals = np.array([ground_height(model, data, p[0] + d, p[1]) - h0 for d in distances])
    return np.nan_to_num(vals, nan=0.0) / scale


# --------------------------------------------------------------------------- the perception-model object (P2, clean)
from terrain_memory_prosthesis.perception.registry import register  # noqa: E402


@register("clean_preview")
class CleanPreview:
    """Ground heights ahead of the pelvis, relative to the ground under it, in units of `scale` (default 10 cm).

    distances / scale / zero default to the environment variables MYO_PREVIEW_DISTANCES / MYO_PREVIEW_SCALE / MYO_PREVIEW_ZERO;
    pass them explicitly (the env does so for the critic's privileged block, which must not depend on the variables).
    """

    name = "clean_preview"

    def __init__(self, distances=None, scale=None, zero=None):
        self.distances = tuple(distances) if distances is not None else preview_distances()
        self.scale = float(scale) if scale is not None else float(os.environ.get("MYO_PREVIEW_SCALE", DEFAULT_SCALE))
        self.zero = bool(zero) if zero is not None else os.environ.get("MYO_PREVIEW_ZERO", "0") == "1"

    @property
    def size(self) -> int:
        return len(self.distances)

    def reset(self, seed=None) -> None:
        pass

    def observe(self, model, data) -> np.ndarray:
        if self.zero:
            return np.zeros(self.size)
        return terrain_preview(model, data, self.distances, self.scale)
