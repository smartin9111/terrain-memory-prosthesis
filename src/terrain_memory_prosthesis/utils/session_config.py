"""Pure-dict helpers that derive MyoAssist session configs (P1 / P2, terrain) from a base config."""
from __future__ import annotations

import copy

from terrain_memory_prosthesis.envs.terrain import terrain_spec
from terrain_memory_prosthesis.perception.terrain_preview import preview_distances

PREVIEW_ENV_ID = "myoAssistLegImitationExoPreview-v0"
N_BASE_OBS = 40  # qpos[0:8] qvel[8:17] muscle_act[17:35] sensor[35:39] target_velocity[39]


def add_exo_obs_range(d: dict, lo: int, hi: int) -> dict:
    """Return a copy of config dict `d` whose prosthesis actor ALSO sees obs[lo:hi] (used to make a 'P2-sized' actor)."""
    import copy

    d = copy.deepcopy(d)
    d["policy_params"]["custom_policy_params"]["net_indexing_info"]["exo_actor"]["observation"].append({"type": "range", "range": [lo, hi]})
    return d


def make_session_config(base: dict, perception: str, terrain: dict | None = None, distances=None, privileged_critic: bool = True) -> dict:
    """Stage-2 config for perception condition "P1" or "P2" on the preview env.

    * env id -> myoAssistLegImitationExoPreview-v0 (obs = 40 proprioceptive + K preview values)
    * critic sees the preview in BOTH conditions (identical value baseline); the human actor never does
    * P2: the prosthesis actor additionally sees obs[40:40+K]
    """
    d = copy.deepcopy(base)
    d["env_params"]["env_id"] = PREVIEW_ENV_ID
    d["env_params"]["terrain"] = terrain
    k = len(distances if distances is not None else preview_distances())
    ni = d["policy_params"]["custom_policy_params"]["net_indexing_info"]
    if privileged_critic:
        ni["common_critic"]["observation"].append({"type": "range", "range": [N_BASE_OBS, N_BASE_OBS + k]})
    if perception == "P2":
        ni["exo_actor"]["observation"].append({"type": "range", "range": [N_BASE_OBS, N_BASE_OBS + k], "comment": "terrain preview"})
    elif perception != "P1":
        raise ValueError(f"perception must be 'P1' or 'P2', got {perception!r}")
    return d
