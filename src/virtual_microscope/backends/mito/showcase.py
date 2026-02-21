"""Showcase images for the mito backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, CYAN, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the mito backend.

    Uses apply_color (no bg subtraction) to show MitoTracker fluorescence.
    1. BF overview
    2. MitoTracker (green) — tubular mitochondrial network
    3. DAPI nuclei (cyan)
    4. Composite MitoTracker + DAPI
    """
    from virtual_microscope.backends.mito import create_sim

    sim = create_sim(world_size=512, n_tubules=40, seed=seed, internal_scale=4)

    # 1 — Brightfield overview
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — MitoTracker (green) — apply_color bypasses bg subtraction
    mito = snap_channel(sim, mode=2, exposure=60.0)
    img2 = apply_color(mito, GREEN)

    # 3 — DAPI nuclei (cyan)
    dapi = snap_channel(sim, mode=1, exposure=60.0)
    img3 = apply_color(dapi, CYAN)

    # 4 — Composite MitoTracker + DAPI
    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
