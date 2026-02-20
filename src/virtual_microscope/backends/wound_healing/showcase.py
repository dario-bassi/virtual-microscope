"""Showcase images for the wound_healing backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, snap_channel, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of wound_healing.

    1. Brightfield at t=0 — wound gap visible
    2. Nucleus fluorescence (green) at t=0
    3. Brightfield after stepping — wound closing
    4. Nucleus fluorescence after stepping — migration front
    """
    from virtual_microscope.backends.wound_healing import create_sim

    sim = create_sim(n_cells=100, seed=seed, wound_width=120, migration_speed=2.0)
    # Pre-equilibrate tissue
    for _ in range(5):
        sim.step(dt=1.0)
    # Create vertical scratch wound
    sim.create_wound(
        shape="rectangle",
        center=(256, 256),
        size=(120, 512),
        jagged=0.2,
    )
    sim._bf_full = None
    sim._nuc_full = None
    sim._mem_full = None

    # 1 — Brightfield at t=0 (wound gap visible)
    bf0 = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf0, "gray")

    # 2 — Nucleus fluorescence at t=0
    nuc0 = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(nuc0, GREEN)

    # 3 — Step to let wound close
    for _ in range(30):
        sim.step(dt=1.0)
    bf_healed = snap_channel(sim, mode=0, exposure=50.0)
    img3 = apply_cmap(bf_healed, "gray")

    # 4 — Nucleus fluorescence after wound closure
    nuc_healed = snap_channel(sim, mode=1, exposure=60.0)
    img4 = apply_fluorescence(nuc_healed, GREEN)

    return [img1, img2, img3, img4]
