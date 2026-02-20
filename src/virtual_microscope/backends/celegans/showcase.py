"""Showcase images for the celegans backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the celegans backend."""
    from virtual_microscope.backends.celegans import create_sim

    sim = create_sim(world_size=2048, seed=seed)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    gfp = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(gfp, GREEN)

    mch = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(mch, RED)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
