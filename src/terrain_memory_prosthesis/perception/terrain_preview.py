"""P2 exteroception: terrain height preview ahead of the walker (pure MuJoCo, no MyoAssist import).

preview(model, data, distances, scale) -> K values
    h_k = (ground height at pelvis_x + d_k) - (ground height under the pelvis), divided by `scale` (default 0.1 m),
    look-ahead distances d_k along world +x (the walking direction); beyond the edge of the terrain -> 0.

Ground height comes from a downward `mujoco.mj_ray` from z = 2 m. The ray hits the walker's own geoms first
(hat_*, Cylinder_abdomen ...), so the cast is repeated from just below any hit that does not belong to the world body
(body id 0). Measured cost: ~0.3 % of one control step for 5 rays.

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
    """Height [m] of the first WORLD geom (terrain/plane) below (x, y, z_top); nan if none."""
    import mujoco

    model, data = getattr(model, "ptr", model), getattr(data, "ptr", data)  # dm_control wrappers -> raw mujoco
    gid = np.zeros(1, dtype=np.int32)
    origin = np.array([x, y, z_top], dtype=float)
    for _ in range(max_hops):
        t = mujoco.mj_ray(model, data, origin, _DOWN, None, 1, -1, gid)
        if t < 0:
            return float("nan")
        if model.geom_bodyid[gid[0]] == 0:
            return origin[2] - t
        origin[2] -= t + 1e-4  # skip a geom of the walker and continue below it
    return float("nan")


def terrain_preview(model, data, distances=None, scale: float | None = None, body: str = "pelvis") -> np.ndarray:
    distances = distances if distances is not None else preview_distances()
    scale = scale if scale is not None else float(os.environ.get("MYO_PREVIEW_SCALE", DEFAULT_SCALE))
    p = data.body(body).xpos
    h0 = ground_height(model, data, p[0], p[1])
    vals = np.array([ground_height(model, data, p[0] + d, p[1]) - h0 for d in distances])
    return np.nan_to_num(vals, nan=0.0) / scale
