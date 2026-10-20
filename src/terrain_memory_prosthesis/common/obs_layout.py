"""Observation layout of the Stage-2 environment (the one place where the sizes live).

    obs = [ clean proprioceptive (40) | corrupted sensor values (40) | validity (40) | perception | privileged | history window ]

* proprioceptive : MyoAssist's own observation (qpos 0:8, qvel 8:17, muscle activations 17:35, foot/toe forces 35:39,
                   target velocity 39). The Stage-1 human is frozen on exactly these 40 numbers: NEVER reorder/rescale them.
* sensor values  : duplicate of the 40-value observation after noise/delay/dropout; only the prosthesis actor reads its existing 22
                   semantic channels from this block.
* validity       : 40 binary availability flags; the prosthesis actor receives the flags of its 22 sensor channels.
* perception     : optional perception features (P1: zero values; P2: terrain preview), appended after the sensor blocks.
* privileged     : the CLEAN terrain preview (always `N_PRIVILEGED` = 5 values), read by the critic only in P1 and P2.
* history        : only for the recurrent controller C2 (n_hist > 0): the last `n_hist` frames, oldest first. A frame is what the
                   PROSTHESIS ACTOR may see: sensor values + validity + perception (never the clean proprioception, so sensor
                   faults cannot be bypassed by the memory; environment/history.py). Placed LAST, so every index of the blocks
                   above is the same for C1 and C2.
"""
from __future__ import annotations

from dataclasses import dataclass

N_PROPRIO = 40
N_PRIVILEGED = 5
N_SENSOR_VALUES = N_PROPRIO
N_SENSOR_VALIDITY = N_PROPRIO
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
    def sensor_values_range(self) -> tuple[int, int]:
        """Duplicate noisy proprioceptive measurements; clean baseline remains at [0:40]."""
        return (self.n_proprio, self.n_proprio + N_SENSOR_VALUES)

    @property
    def sensor_validity_range(self) -> tuple[int, int]:
        a = self.sensor_values_range[1]
        return (a, a + N_SENSOR_VALIDITY)

    @property
    def percept_range(self) -> tuple[int, int]:
        a = self.sensor_validity_range[1]
        return (a, a + self.n_percept)

    @property
    def priv_range(self) -> tuple[int, int]:
        a = self.percept_range[1]
        return (a, a + self.n_priv)

    @property
    def frame_start(self) -> int:
        """First obs index of one history frame (= the sensor values; the clean proprioception is NOT part of it)."""
        return self.sensor_values_range[0]

    @property
    def frame_width(self) -> int:
        """One history frame = sensor values + validity + perception block (obs[frame_start : frame_start + frame_width])."""
        return N_SENSOR_VALUES + N_SENSOR_VALIDITY + self.n_percept

    @property
    def hist_range(self) -> tuple[int, int]:
        a = self.priv_range[1]
        return (a, a + self.n_hist * self.frame_width)

    @property
    def total(self) -> int:
        return self.n_proprio + N_SENSOR_VALUES + N_SENSOR_VALIDITY + self.n_percept + self.n_priv + self.n_hist * self.frame_width
