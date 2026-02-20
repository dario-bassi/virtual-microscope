"""Showcase images for the voronoi backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import voronoi_showcase, GREEN, MAGENTA


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the voronoi backend."""
    from virtual_microscope.backends.voronoi import create_sim

    sim = create_sim(n_cells=60, seed=seed, internal_scale=4)
    return voronoi_showcase(sim, nuc_color=GREEN, mem_color=MAGENTA, n_steps=0)
