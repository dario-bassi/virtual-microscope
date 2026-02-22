"""Showcase images for the particle backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, GREEN, MAGENTA,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the particle backend."""
    from virtual_microscope.backends.particle import create_sim

    sim = create_sim(n_cells=50, seed=seed)
    sim.auto_step = False

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(nuc, GREEN)

    mem = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_color(mem, MAGENTA)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
