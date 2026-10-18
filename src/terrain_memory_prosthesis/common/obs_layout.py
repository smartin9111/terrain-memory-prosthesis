"""Observation layout of the Stage-2 environment (the one place where the sizes live).

    obs = [ proprioceptive (40) | perception block (n_percept) | privileged block (n_priv) | history window (n_hist x frame) ]

* proprioceptive : MyoAssist's own observation (qpos 0:8, qvel 8:17, muscle activations 17:35, foot/toe forces 35:39,
                   target velocity 39). The Stage-1 human is frozen on exactly these 40 numbers: NEVER reorder/rescale them.
* perception     : what the PROSTHESIS ACTOR may additionally see (P1: nothing, n_percept = 0; P2: the terrain preview, possibly
                   degraded / encoded). Its size is `PerceptionModel.size`. It is always placed right after the 40 numbers and
                   is the LAST input block of the prosthesis actor, so the warm start can add zero-initialised columns for it.
* privileged     : the CLEAN terrain preview (always `N_PRIVILEGED` = 5 values at 0.2..1.0 m), seen by the CRITIC only, in
                   P1 and P2 alike (identical value baseline; asymmetric actor-critic).
* history        : only for the recurrent controller C2 (n_hist > 0): the last `n_hist` frames, oldest first, each frame being the
                   first `n_proprio + n_percept` numbers of the observation (environment/history.py). Placed LAST, so every index
                   of the blocks above is the same for C1 and C2.
"""
from __future__ import annotations

from dataclasses import dataclass

N_PROPRIO = 40
N_PRIVILEGED = 5
PREVIEW_ENV_ID = "myoAssistLegImitationExoPreview-v0"
PERCEPTION_ENV_VAR = "MYO_PERCEPTION"  # name of the perception model the env builds (default "clean_preview"; P1 = "none")
HISTORY_ENV_VAR = "MYO_HISTORY"  # C2 only: length K of the observation window the env appends (unset / 0 = no window = C1)
GRU_HIDDEN_ENV_VAR = "MYO_GRU_HIDDEN"  # C2 only: hidden size of the GRU


@dataclass(frozen=True)
class ObsLayout:
    n_percept: int
    n_priv: int = N_PRIVILEGED
    n_proprio: int = N_PROPRIO
    n_hist: int = 0

    @property
    def percept_range(self) -> tuple[int, int]:
        return (self.n_proprio, self.n_proprio + self.n_percept)

    @property
    def priv_range(self) -> tuple[int, int]:
        a = self.n_proprio + self.n_percept
        return (a, a + self.n_priv)

    @property
    def frame_width(self) -> int:
        """One history frame = proprioception + perception block."""
        return self.n_proprio + self.n_percept

    @property
    def hist_range(self) -> tuple[int, int]:
        a = self.n_proprio + self.n_percept + self.n_priv
        return (a, a + self.n_hist * self.frame_width)

    @property
    def total(self) -> int:
        return self.n_proprio + self.n_percept + self.n_priv + self.n_hist * self.frame_width
