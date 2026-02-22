"""Showcase images for the microfluidics backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, CYAN, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the microfluidics backend."""
    from virtual_microscope.backends.microfluidics import create_sim

    sim = create_sim(n_cells=30, channel_width=100, flow_speed=3.0, gradient=True, seed=seed, internal_scale=4)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    dapi = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(dapi, CYAN)

    grad = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_color(grad, GREEN)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
