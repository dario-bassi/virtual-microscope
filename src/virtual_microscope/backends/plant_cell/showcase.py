"""Showcase images for the plant_cell backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, gray_to_rgb, snap_channel, CYAN, MAGENTA,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the plant_cell backend."""
    from virtual_microscope.backends.plant_cell import create_sim

    sim = create_sim(world_size=512, staining="iodine", seed=seed, internal_scale=4)

    # Brightfield iodine — may be RGB
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = gray_to_rgb(bf, stretch=False) if bf.ndim == 3 else apply_cmap(bf, "gray")

    dapi = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(dapi, CYAN)

    cfw = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_color(cfw, MAGENTA)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
