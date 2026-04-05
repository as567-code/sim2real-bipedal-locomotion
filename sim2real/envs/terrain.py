from __future__ import annotations

import numpy as np
import mujoco


class TerrainGenerator:
    """Procedural terrain heightfield for the bipedal environment.

    Supports four terrain types applied to the MuJoCo hfield asset:
      - flat: zero heightfield
      - slope: linear gradient along x
      - steps: discrete step heights
      - rough: Perlin-like random bumps
    """

    TERRAIN_TYPES = ("flat", "slope", "steps", "rough")

    def __init__(self, model: mujoco.MjModel, max_height: float = 0.03):
        self.model = model
        self.max_height = max_height

        self._hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self.nrow = model.hfield_nrow[self._hfield_id]
        self.ncol = model.hfield_ncol[self._hfield_id]

    def generate(self, terrain_type: str | None = None, rng: np.random.Generator | None = None) -> str:
        """Write a new heightfield into the model. Returns the terrain type used."""
        rng = rng or np.random.default_rng()

        if terrain_type is None:
            terrain_type = rng.choice(self.TERRAIN_TYPES)

        hfield = np.zeros((self.nrow, self.ncol), dtype=np.float32)

        if terrain_type == "flat":
            pass
        elif terrain_type == "slope":
            slope_angle = rng.uniform(0.02, 0.08)
            for i in range(self.nrow):
                hfield[i, :] = slope_angle * (i / self.nrow)
        elif terrain_type == "steps":
            num_steps = rng.integers(3, 8)
            step_width = self.nrow // num_steps
            for s in range(num_steps):
                height = self.max_height * (s + 1) / num_steps * rng.uniform(0.5, 1.0)
                r_start = s * step_width
                r_end = min((s + 1) * step_width, self.nrow)
                hfield[r_start:r_end, :] = height
        elif terrain_type == "rough":
            hfield = self._fractal_noise(rng)
        else:
            raise ValueError(f"Unknown terrain type: {terrain_type}")

        hfield = np.clip(hfield, 0.0, self.max_height)

        adr = self.model.hfield_adr[self._hfield_id]
        size = self.nrow * self.ncol
        self.model.hfield_data[adr : adr + size] = hfield.ravel()

        return terrain_type

    def _fractal_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Multi-octave value noise for rough terrain."""
        hfield = np.zeros((self.nrow, self.ncol), dtype=np.float32)
        for octave in range(4):
            freq = 2 ** octave
            amplitude = self.max_height / (2 ** octave)
            rows = np.linspace(0, freq * np.pi, self.nrow)
            cols = np.linspace(0, freq * np.pi, self.ncol)
            phase_r = rng.uniform(0, 2 * np.pi)
            phase_c = rng.uniform(0, 2 * np.pi)
            grid = np.outer(np.sin(rows + phase_r), np.sin(cols + phase_c))
            hfield += (amplitude * grid).astype(np.float32)
        hfield -= hfield.min()
        if hfield.max() > 0:
            hfield = hfield / hfield.max() * self.max_height
        return hfield
