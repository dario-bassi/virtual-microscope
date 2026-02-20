"""Showcase images for the calcium backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the calcium backend.

    1. Brightfield — cell monolayer
    2. GCaMP fluorescence (green) — resting state
    3. GCaMP after stepping — bright propagating wave
    4. GCaMP later — wave has passed, refractory region
    """
    from virtual_microscope.backends.calcium import create_sim

    sim = create_sim(grid_size=512, n_cells=200, seed=seed, internal_scale=4)

    # 1 — Brightfield
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — GCaMP at rest
    gcamp_rest = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(gcamp_rest, GREEN)

    # 3 — Step to propagate waves
    for _ in range(8):
        sim.step(dt=1.0)
    gcamp_wave = snap_channel(sim, mode=1, exposure=60.0)
    img3 = apply_fluorescence(gcamp_wave, GREEN)

    # 4 — Step further to see refractory region
    for _ in range(8):
        sim.step(dt=1.0)
    gcamp_late = snap_channel(sim, mode=1, exposure=60.0)
    img4 = apply_fluorescence(gcamp_late, GREEN)

    return [img1, img2, img3, img4]
