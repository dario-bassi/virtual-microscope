"""Showcase images for the reaction_diffusion backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import apply_cmap, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the reaction_diffusion backend.

    Different Gray-Scott presets showing the variety of Turing patterns.
    1. Waves preset
    2. Spots preset
    3. Spirals preset
    4. Coral preset
    """
    from virtual_microscope.backends.reaction_diffusion import create_sim

    images = []
    for preset in ("waves", "spots", "spirals", "coral"):
        sim = create_sim(grid_size=512, preset=preset, seed=seed)
        # Step to develop patterns
        for _ in range(5):
            sim.step(dt=1.0)
        bf = snap_channel(sim, mode=0, exposure=50.0)
        images.append(apply_cmap(bf, "inferno"))
        del sim

    return images
