"""Showcase images for the viability backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the viability backend."""
    from virtual_microscope.backends.viability import create_sim

    sim = create_sim(n_cells=60, seed=seed, live_fraction=0.85)
    sim.enable_viability_staining(live_fraction=0.85, rng_seed=seed + 1000)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    calcein = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(calcein, GREEN)

    pi = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(pi, RED)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
