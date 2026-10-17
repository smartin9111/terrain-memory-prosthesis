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


EPISODIC_PREFIX = "episodic_"


def episodic_terrain(amp_lo: float, amp_hi: float, extent: float = 160.0, resolution: int = 640):
    """Terrain dict for PER-EPISODE random terrain (Stage 2 training).

    The heightfield is compiled once with relief `amp_hi`; at every reset the env (environment/preview_env.py) draws a new
    amplitude ~ U(amp_lo, amp_hi) and a new per-cell noise field and writes it into `model.hfield_data` (the geometry
    keeps its compiled bounds because only `amp <= amp_hi` is ever written). The range travels in `terrain_name`
    ("episodic_<lo>_<hi>") so it is recorded in session_config.json. Cells stay 25 cm (extent/resolution) like in Stage 1.
    """
    if not 0.0 <= amp_lo <= amp_hi:
        raise ValueError(f"need 0 <= amp_lo <= amp_hi, got {amp_lo}, {amp_hi}")
    return {
        "terrain": "random",
        "terrain_name": f"{EPISODIC_PREFIX}{amp_lo:g}_{amp_hi:g}",
        "amplitude": float(amp_hi),
        "seed": 0,
        "extent": float(extent),
        "resolution": int(resolution),
    }


def parse_episodic(terrain) -> tuple[float, float] | None:
    """(amp_lo, amp_hi) if `terrain` is an episodic_terrain() dict, else None."""
    name = terrain.get("terrain_name", "") if isinstance(terrain, dict) else ""
    if not name.startswith(EPISODIC_PREFIX):
        return None
    lo, hi = name[len(EPISODIC_PREFIX):].split("_")
    return float(lo), float(hi)
