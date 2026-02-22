"""Showcase images for the stress_granule backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, CYAN, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the stress_granule backend."""
    from virtual_microscope.backends.stress_granule import create_sim

    sim = create_sim(n_cells=30, seed=seed)
    sim.enable_stress_granules(
        foci_radius_range=(2.0, 4.0), max_foci_per_cell=12,
        formation_rate=0.25, dissolution_rate=0.12,
        heterogeneity=0.35, baseline_foci=0,
    )
    for _ in range(15):
        sim.step(dt=1.0)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(nuc, CYAN)

    sg = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_color(sg, GREEN)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
