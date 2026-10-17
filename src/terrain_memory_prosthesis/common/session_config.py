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
    """Stage-2 config for perception condition "P1" or "P2" on the preview env.

    * env id -> myoAssistLegImitationExoPreview-v0, obs = [40 proprioceptive | n_percept | N_PRIVILEGED]
    * the critic sees the privileged clean preview in BOTH conditions (identical value baseline); the human actor never does
    * P1: n_percept = 0, the prosthesis actor sees proprioception only
    * P2: the prosthesis actor additionally sees the perception block obs[40:40+n_percept] (default n_percept = N_PRIVILEGED = 5)
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
    if privileged_critic:
        ni["common_critic"]["observation"].append({"type": "range", "range": list(lay.priv_range), "comment": "privileged clean terrain preview"})
    if perception == "P2":
        ni["exo_actor"]["observation"].append({"type": "range", "range": list(lay.percept_range), "comment": "perception block (terrain preview)"})
    return d
