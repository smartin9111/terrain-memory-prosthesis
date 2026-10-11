"""MyoAssist terrain dicts (pure python, no simulator import)."""
from __future__ import annotations


def terrain_spec(kind: str, amplitude: float = 0.0, seed: int = 0, period: float = 1.0, extent: float = 60.0, resolution: int = 768):
    """MyoAssist terrain dict. NOTE the upstream defaults (extent 20 m centred on the origin, 3 m flat safe zone) mean
    a walker leaves the heightfield after ~10 m (~250 control steps at 1.25 m/s); we therefore default to a 60 m field."""
    if kind == "flat":
        return None
    if kind == "random":
        return {"terrain": "random", "terrain_name": f"rnd_{amplitude}_{seed}", "amplitude": amplitude, "seed": seed, "extent": extent, "resolution": resolution}
    if kind == "sinusoidal":
        return {"terrain": "sinusoidal", "terrain_name": f"sin_{amplitude}_{period}", "amplitude": amplitude, "period": period, "extent": extent, "resolution": resolution}
    raise ValueError(kind)
