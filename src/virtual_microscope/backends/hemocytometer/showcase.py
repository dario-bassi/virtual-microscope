"""Showcase images for the hemocytometer backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the hemocytometer backend.

    Native RGB — brightfield and trypan blue channels, different viabilities.
    """
    from virtual_microscope.backends.hemocytometer import create_sim

    sim = create_sim(n_cells=150, viability=0.85, seed=seed)
    img1 = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0), stretch=False)

    img2 = gray_to_rgb(snap_channel(sim, mode=1, exposure=80.0), stretch=False)

    sim_low = create_sim(n_cells=150, viability=0.50, seed=seed + 3)
    img3 = gray_to_rgb(snap_channel(sim_low, mode=0, exposure=80.0), stretch=False)

    img4 = gray_to_rgb(snap_channel(sim_low, mode=1, exposure=80.0), stretch=False)

    del sim_low
    return [img1, img2, img3, img4]
