"""Showcase images for the celegans backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, GREEN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the celegans backend.

    Reduced world_size to zoom onto the worm body and show internal detail.
    1. DIC brightfield closeup (pharynx/intestine visible)
    2. GFP-pharynx channel (green)
    3. mCherry-body channel (red)
    4. Composite GFP + mCherry after stepping (different posture)
    """
    from virtual_microscope.backends.celegans import create_sim

    # Smaller world for zoomed-in view of worm body
    sim = create_sim(world_size=768, seed=seed)
    sim.auto_step = False  # decouple imaging from simulation

    bf = snap_channel(sim, mode=0, exposure=50.0)
    gfp = snap_channel(sim, mode=1, exposure=60.0)
    mch = snap_channel(sim, mode=2, exposure=60.0)

    img1 = apply_cmap(bf, "gray")
    img2 = apply_color(gfp, GREEN)
    img3 = apply_color(mch, RED)

    # 4 — Composite GFP + mCherry (same timepoint, channels overlap)
    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
