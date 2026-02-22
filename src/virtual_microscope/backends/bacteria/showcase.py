"""Showcase images for the bacteria backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the bacteria backend.

    1. Phase contrast — rod-shaped cells with halo
    2. GFP fluorescence (green) — glowing rods on dark background
    3. After 20 steps — colony has grown, cells dividing (phase contrast)
    4. Composite: phase contrast + GFP overlay
    """
    from virtual_microscope.backends.bacteria import create_sim

    sim = create_sim(n_cells=30, seed=seed, world_size=512, internal_scale=4)
    sim.auto_step = False  # decouple imaging from simulation

    # 1 — Phase contrast
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — GFP fluorescence (lower exposure for proper dark background)
    gfp = snap_channel(sim, mode=1, exposure=20.0)
    img2 = apply_color(gfp, GREEN)

    # 3 — After growth steps: larger colony
    for _ in range(20):
        sim.step(dt=1.0)
    bf_grown = snap_channel(sim, mode=0, exposure=50.0)
    img3 = apply_cmap(bf_grown, "gray")

    # 4 — Composite: brightfield + GFP overlay
    gfp_grown = snap_channel(sim, mode=1, exposure=20.0)
    gray_bg = apply_cmap(bf_grown, "gray")
    green_fg = apply_color(gfp_grown, GREEN)
    img4 = composite_max(gray_bg, green_fg)

    return [img1, img2, img3, img4]
