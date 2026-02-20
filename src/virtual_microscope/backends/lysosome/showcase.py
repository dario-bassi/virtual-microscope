"""Showcase images for the lysosome backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, CYAN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the lysosome backend."""
    from virtual_microscope.backends.lysosome import create_sim

    sim = create_sim(n_cells=20, seed=seed)
    sim.enable_lysosomes(n_min=5, n_max=20, radius_min=1.5, radius_max=3.0, diffusion_rate=0.3)
    for _ in range(5):
        sim.step(dt=1.0)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(nuc, CYAN)

    lyso = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(lyso, RED)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
