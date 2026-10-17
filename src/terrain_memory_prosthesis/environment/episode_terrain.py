"""Per-episode random heightfield: rewrite `model.hfield_data` at every reset (no model rebuild).

MyoAssist bakes ONE terrain into the model, so every env instance would keep the same ground for its whole life (and, with
the same seed, all instances the same ground). This writes a fresh field into the compiled hfield instead:
    height(row, col) = U(0, amp) * safe_zone_mask,  amp ~ U(amp_lo, amp_hi)       (same recipe as myoassist_terrains.uniform)
MuJoCo does not renormalise `hfield_data` at run time, so the physical height is `hfield_data * hfield_size[2]`; we store
`field / size_z`. `size_z` is the compiled relief (~ amp_hi), amplitudes never exceed it, so the compiled collision bounds stay valid.
"""
from __future__ import annotations

import numpy as np


class EpisodeTerrain:
    def __init__(self, model, amp_lo: float, amp_hi: float, safe_radius: float = 3.0, rng: np.random.Generator | None = None, hfield_id: int = 0):
        model = getattr(model, "ptr", model)
        if model.nhfield < 1:
            raise ValueError("model has no heightfield (terrain must be a random/sinusoidal hfield terrain)")
        self.id = int(hfield_id)
        self.nrow, self.ncol = int(model.hfield_nrow[self.id]), int(model.hfield_ncol[self.id])
        self.adr = int(model.hfield_adr[self.id])
        half_x, half_y, size_z = (float(v) for v in model.hfield_size[self.id][:3])
        self.size_z = size_z
        if amp_hi > size_z * 1.001:
            raise ValueError(f"amp_hi {amp_hi} exceeds the compiled relief {size_z}: compile the terrain with amplitude=amp_hi")
        self.amp_lo, self.amp_hi = float(amp_lo), float(amp_hi)
        self.rng = rng if rng is not None else np.random.default_rng()
        xs = np.linspace(-half_x, half_x, self.ncol)
        ys = np.linspace(-half_y, half_y, self.nrow)
        xx, yy = np.meshgrid(xs, ys)
        if safe_radius > 0:
            t = np.clip(np.sqrt(xx * xx + yy * yy) / safe_radius, 0.0, 1.0)
            self.mask = t * t * (3.0 - 2.0 * t)  # smoothstep, identical to myoassist_terrains.uniform
        else:
            self.mask = np.ones_like(xx)
        self.last_amplitude = float("nan")

    def resample(self, model) -> float:
        """Draw amplitude + a new field and write it into the model. Returns the amplitude."""
        model = getattr(model, "ptr", model)
        amp = float(self.rng.uniform(self.amp_lo, self.amp_hi)) if self.amp_hi > self.amp_lo else self.amp_hi
        field = self.rng.uniform(0.0, amp, size=(self.nrow, self.ncol)) * self.mask
        model.hfield_data[self.adr : self.adr + self.nrow * self.ncol] = (field / self.size_z).reshape(-1)
        self.last_amplitude = amp
        return amp
