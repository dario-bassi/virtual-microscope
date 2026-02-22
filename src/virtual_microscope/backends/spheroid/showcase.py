"""Showcase images for the spheroid backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, GREEN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the spheroid backend."""
    from virtual_microscope.backends.spheroid import create_sim

    sim = create_sim(radius=80, n_cells=2000, necrotic_fraction=0.45, seed=seed, internal_scale=4)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    calcein = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(calcein, GREEN)

    pi = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_color(pi, RED)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
