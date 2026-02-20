"""Showcase images for the fibroblast backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, CYAN, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the fibroblast backend."""
    from virtual_microscope.backends.fibroblast import create_sim

    sim = create_sim(world_size=512, n_cells=8, seed=seed, internal_scale=4)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    dapi = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(dapi, CYAN)

    actin = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(actin, GREEN)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
