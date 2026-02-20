"""Showcase images for the dictyostelium backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, CYAN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the dictyostelium backend."""
    from virtual_microscope.backends.dictyostelium import create_sim

    sim = create_sim(n_cells=100, seed=seed)

    df = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(df, "gray")

    gfp = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(gfp, GREEN)

    for _ in range(10):
        sim.step(dt=1.0)
    camp = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(camp, CYAN)

    gfp2 = snap_channel(sim, mode=1, exposure=60.0)
    img4 = composite_max(apply_fluorescence(gfp2, GREEN), img3)

    return [img1, img2, img3, img4]
