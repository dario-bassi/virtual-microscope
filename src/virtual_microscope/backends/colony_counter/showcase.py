"""Showcase images for the colony_counter backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_fluorescence, gray_to_rgb, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the colony_counter backend."""
    from virtual_microscope.backends.colony_counter import create_sim

    sim = create_sim(n_colonies=200, plate_type="spread", seed=seed)
    img1 = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0))

    sim_blue = create_sim(n_colonies=200, plate_type="spread", seed=seed, staining="xgal", blue_fraction=0.3)
    img2 = gray_to_rgb(snap_channel(sim_blue, mode=1, exposure=80.0))

    gfp = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(gfp, GREEN)

    sim_streak = create_sim(n_colonies=150, plate_type="streak", seed=seed + 1)
    img4 = gray_to_rgb(snap_channel(sim_streak, mode=0, exposure=80.0))

    del sim_blue, sim_streak
    return [img1, img2, img3, img4]
