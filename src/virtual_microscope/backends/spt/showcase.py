"""Showcase images for the spt (single-particle tracking) backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_color, gray_to_rgb, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the spt backend.

    1. Epifluorescence overview — all particles visible
    2. SPT/TIRF channel (green) — bright individual spots
    3. After stepping — particles have moved
    4. After more steps — diffusion visible
    """
    from virtual_microscope.backends.spt import create_sim

    sim = create_sim(n_free=20, n_confined=10, n_directed=5, seed=seed)
    # Center camera on the 128x128 world (default offset shows only top-left corner)
    sim.camera_offset = np.array(
        [sim.width // 2 - sim.viewport_width // 2,
         sim.height // 2 - sim.viewport_height // 2],
        dtype=float,
    )

    # 1 — Epifluorescence overview
    epi = snap_channel(sim, mode=0, exposure=80.0)
    img1 = gray_to_rgb(epi, stretch=False)

    # 2 — SPT/TIRF channel
    spt = snap_channel(sim, mode=1, exposure=100.0)
    img2 = apply_color(spt, GREEN)

    # 3 — After stepping
    for _ in range(10):
        sim.step(dt=1.0)
    spt2 = snap_channel(sim, mode=1, exposure=100.0)
    img3 = apply_color(spt2, GREEN)

    # 4 — After more steps
    for _ in range(20):
        sim.step(dt=1.0)
    spt3 = snap_channel(sim, mode=1, exposure=100.0)
    img4 = apply_color(spt3, GREEN)

    return [img1, img2, img3, img4]
