"""Observation layout of the Stage-2 environment (the one place where the sizes live).

    obs = [ clean proprioceptive (40) | corrupted sensor values (40) | validity (40) | perception | privileged ]

* proprioceptive : MyoAssist's own observation (qpos 0:8, qvel 8:17, muscle activations 17:35, foot/toe forces 35:39,
                   target velocity 39). The Stage-1 human is frozen on exactly these 40 numbers: NEVER reorder/rescale them.
* sensor values  : duplicate of the 40-value observation after noise/delay/dropout; only the prosthesis actor reads its existing 22 semantic channels from this block.
* validity       : 40 binary availability flags; the prosthesis actor receives flags corresponding to its 22 sensor channels.
* perception     : optional perception features (P1: zero values; P2: terrain preview), appended after sensor blocks.
* privileged     : the CLEAN terrain preview (always `N_PRIVILEGED` = 5 values), read by the critic only in P1 and P2.
"""
from __future__ import annotations

from dataclasses import dataclass

N_PROPRIO = 40
N_PRIVILEGED = 5
N_SENSOR_VALUES = N_PROPRIO
N_SENSOR_VALIDITY = N_PROPRIO
PREVIEW_ENV_ID = "myoAssistLegImitationExoPreview-v0"
PERCEPTION_ENV_VAR = "MYO_PERCEPTION"  # name of the perception model the env builds (default "clean_preview"; P1 = "none")


@dataclass(frozen=True)
class ObsLayout:
    n_percept: int
    n_priv: int = N_PRIVILEGED
    n_proprio: int = N_PROPRIO

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
    def total(self) -> int:
        return self.n_proprio + N_SENSOR_VALUES + N_SENSOR_VALIDITY + self.n_percept + self.n_priv
