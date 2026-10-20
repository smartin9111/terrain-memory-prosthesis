"""Pure-dict helpers that derive MyoAssist session configs (P1 / P2, terrain) from a base config.

Which network sees which part of the observation is decided HERE and nowhere else (layout: common/obs_layout.py).
"""
from __future__ import annotations

import copy

from terrain_memory_prosthesis.common.obs_layout import N_PRIVILEGED, N_PROPRIO, PREVIEW_ENV_ID, ObsLayout

N_BASE_OBS = N_PROPRIO  # kept for older imports


def add_exo_obs_range(d: dict, lo: int, hi: int) -> dict:
    """Return a copy of config dict `d` whose prosthesis actor ALSO sees obs[lo:hi]."""
    d = copy.deepcopy(d)
    d["policy_params"]["custom_policy_params"]["net_indexing_info"]["exo_actor"]["observation"].append({"type": "range", "range": [lo, hi]})
    return d


def make_session_config(base: dict, perception: str, terrain: dict | None = None, n_percept: int | None = None,
                        privileged_critic: bool = True) -> dict:
    """Build Stage-2 config with actor-only sensor corruption.

    The full observation layout is [clean 40 | sensor values 40 | validity 40 | perception | privileged].
    Human actor and critic retain clean inputs. The prosthesis actor receives the same 22 semantic channels as
    Stage 1, but selected from the corrupted-value block, followed by validity bits for those same channels.
    """
    if perception not in ("P1", "P2"):
        raise ValueError(f"perception must be 'P1' or 'P2', got {perception!r}")
    n = n_percept if n_percept is not None else (0 if perception == "P1" else N_PRIVILEGED)
    if perception == "P1" and n != 0:
        raise ValueError("P1 has no perception block (n_percept must be 0)")
    if perception == "P2" and n < 1:
        raise ValueError("P2 needs a perception block (n_percept >= 1)")

    lay = ObsLayout(n_percept=n)
    d = copy.deepcopy(base)
    d["env_params"]["env_id"] = PREVIEW_ENV_ID
    d["env_params"]["terrain"] = terrain
    ni = d["policy_params"]["custom_policy_params"]["net_indexing_info"]

    # Preserve the existing Stage-1 exo actor's selected channels and order, but route them to the
    # duplicated sensor-values block. Append matching validity channels as additional inputs.
    old_ranges = [item for item in ni["exo_actor"]["observation"] if item.get("type") == "range"]
    if not old_ranges:
        raise ValueError("base config exo_actor must define range-based observation inputs")
    sensor_ranges = []
    validity_ranges = []
    for item in old_ranges:
        lo, hi = item["range"]
        sensor_ranges.append({**item, "range": [lo + lay.sensor_values_range[0], hi + lay.sensor_values_range[0]],
                              "comment": f"sensor-corrupted: {item.get('comment', f'observation {lo}:{hi}')}"})
        validity_ranges.append({"type": "range", "range": [lo + lay.sensor_validity_range[0], hi + lay.sensor_validity_range[0]],
                               "comment": f"sensor validity for observation {lo}:{hi}"})
    ni["exo_actor"]["observation"] = sensor_ranges + validity_ranges

    if privileged_critic:
        ni["common_critic"]["observation"].append({"type": "range", "range": list(lay.priv_range),
                                                       "comment": "privileged clean terrain preview"})
    if perception == "P2":
        ni["exo_actor"]["observation"].append({"type": "range", "range": list(lay.percept_range),
                                                 "comment": "perception block (terrain preview)"})
    return d
