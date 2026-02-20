"""Showcase images for the flow_cytometry backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_fluorescence, gray_to_rgb, composite_max, snap_channel, GREEN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the flow_cytometry backend."""
    from virtual_microscope.backends.flow_cytometry import create_sim

    sim = create_sim(n_total=10000, events_per_snap=100, seed=seed)

    scatter = snap_channel(sim, mode=0, exposure=50.0)
    img1 = gray_to_rgb(scatter, stretch=False)

    fl1 = snap_channel(sim, mode=1, exposure=80.0)
    img2 = apply_fluorescence(fl1, GREEN)

    fl2 = snap_channel(sim, mode=2, exposure=80.0)
    img3 = apply_fluorescence(fl2, RED)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
