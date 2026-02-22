"""Showcase images for the volvox backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the volvox backend."""
    from virtual_microscope.backends.volvox import create_sim

    sim = create_sim(n_somatic=300, n_gonidia=4, seed=seed, internal_scale=4)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    chl = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(chl, GREEN)

    for _ in range(10):
        sim.step(dt=1.0)
    bf2 = snap_channel(sim, mode=0, exposure=50.0)
    img3 = apply_cmap(bf2, "gray")

    chl2 = snap_channel(sim, mode=1, exposure=60.0)
    img4 = composite_max(apply_cmap(bf2, "gray"), apply_color(chl2, GREEN))

    return [img1, img2, img3, img4]
