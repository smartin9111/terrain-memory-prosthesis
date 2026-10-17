"""Observation layout of the Stage-2 environment (the one place where the sizes live).

    obs = [ proprioceptive (40) | perception block (n_percept) | privileged block (n_priv) ]

* proprioceptive : MyoAssist's own observation (qpos 0:8, qvel 8:17, muscle activations 17:35, foot/toe forces 35:39,
                   target velocity 39). The Stage-1 human is frozen on exactly these 40 numbers: NEVER reorder/rescale them.
* perception     : what the PROSTHESIS ACTOR may additionally see (P1: nothing, n_percept = 0; P2: the terrain preview, possibly
                   degraded / encoded). Its size is `PerceptionModel.size`. It is always placed right after the 40 numbers and
                   is the LAST input block of the prosthesis actor, so the warm start can add zero-initialised columns for it.
* privileged     : the CLEAN terrain preview (always `N_PRIVILEGED` = 5 values at 0.2..1.0 m), seen by the CRITIC only, in
                   P1 and P2 alike (identical value baseline; asymmetric actor-critic).
"""
from __future__ import annotations

from dataclasses import dataclass

N_PROPRIO = 40
N_PRIVILEGED = 5
PREVIEW_ENV_ID = "myoAssistLegImitationExoPreview-v0"
PERCEPTION_ENV_VAR = "MYO_PERCEPTION"  # name of the perception model the env builds (default "clean_preview"; P1 = "none")


@dataclass(frozen=True)
class ObsLayout:
    n_percept: int
    n_priv: int = N_PRIVILEGED
    n_proprio: int = N_PROPRIO

    @property
    def percept_range(self) -> tuple[int, int]:
        return (self.n_proprio, self.n_proprio + self.n_percept)

    @property
    def priv_range(self) -> tuple[int, int]:
        a = self.n_proprio + self.n_percept
        return (a, a + self.n_priv)

    @property
    def total(self) -> int:
        return self.n_proprio + self.n_percept + self.n_priv
