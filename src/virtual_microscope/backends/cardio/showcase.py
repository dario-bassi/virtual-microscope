"""Showcase images for the cardio backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the cardio backend.

    Shows cardiac wave propagation over time:
    1. Resting tissue (brightfield)
    2. Early wave front (GCaMP after ~8 steps)
    3. Wave mid-propagation with ectopic interference (~23 steps)
    4. Composite BF + GCaMP
    """
    from virtual_microscope.backends.cardio import create_sim

    sim = create_sim(grid_size=512, n_cells=300, seed=seed, internal_scale=4)
    sim.auto_step = False  # decouple imaging from simulation

    # 1 — Resting tissue brightfield
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — Early wave front (GCaMP after a few steps)
    for _ in range(8):
        sim.step(dt=1.0)
    gcamp1 = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(gcamp1, GREEN)

    # 3 — Wave mid-propagation with ectopic focus collision
    for _ in range(15):
        sim.step(dt=1.0)
    gcamp2 = snap_channel(sim, mode=1, exposure=60.0)
    img3 = apply_color(gcamp2, GREEN)

    # 4 — Composite BF + GCaMP (same timepoint)
    bf2 = snap_channel(sim, mode=0, exposure=50.0)
    img4 = composite_max(apply_cmap(bf2, "gray"), apply_color(gcamp2, GREEN))

    return [img1, img2, img3, img4]
