"""Showcase images for the hemocytometer backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the hemocytometer backend.

    Shows extreme viability contrast for obvious visual difference:
    1. BF high viability (95%) — mostly healthy cells
    2. Trypan blue high viability — few blue dead cells
    3. BF low viability (30%) — lots of debris/dead cells
    4. Trypan blue low viability — many blue cells
    """
    from virtual_microscope.backends.hemocytometer import create_sim

    # High viability (95%)
    sim_hi = create_sim(n_cells=150, viability=0.95, seed=seed)
    img1 = gray_to_rgb(snap_channel(sim_hi, mode=0, exposure=80.0), stretch=False)
    img2 = gray_to_rgb(snap_channel(sim_hi, mode=1, exposure=80.0), stretch=False)

    # Low viability (30%)
    sim_lo = create_sim(n_cells=150, viability=0.30, seed=seed + 3)
    img3 = gray_to_rgb(snap_channel(sim_lo, mode=0, exposure=80.0), stretch=False)
    img4 = gray_to_rgb(snap_channel(sim_lo, mode=1, exposure=80.0), stretch=False)

    del sim_hi, sim_lo
    return [img1, img2, img3, img4]
