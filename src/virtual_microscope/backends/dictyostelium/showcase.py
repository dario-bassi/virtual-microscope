"""Showcase images for the dictyostelium backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, CYAN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the dictyostelium backend.

    Shows 4 progressive timepoints of cAMP-driven aggregation:
    1. Dispersed cells (t=0, darkfield)
    2. Early cAMP waves (t≈20)
    3. Streaming begins (t≈50)
    4. Late aggregation (t≈80, composite GFP + cAMP)
    """
    from virtual_microscope.backends.dictyostelium import create_sim

    sim = create_sim(n_cells=100, seed=seed)

    # 1 — Dispersed cells (t=0, darkfield)
    img1 = apply_cmap(snap_channel(sim, mode=0, exposure=50.0), "gray")

    # 2 — Early cAMP waves (t≈20)
    for _ in range(20):
        sim.step(dt=1.0)
    gfp = snap_channel(sim, mode=1, exposure=60.0)
    camp = snap_channel(sim, mode=2, exposure=60.0)
    img2 = composite_max(
        apply_fluorescence(gfp, GREEN), apply_fluorescence(camp, CYAN),
    )

    # 3 — Streaming begins (t≈50)
    for _ in range(30):
        sim.step(dt=1.0)
    gfp = snap_channel(sim, mode=1, exposure=60.0)
    camp = snap_channel(sim, mode=2, exposure=60.0)
    img3 = composite_max(
        apply_fluorescence(gfp, GREEN), apply_fluorescence(camp, CYAN),
    )

    # 4 — Late aggregation (t≈80)
    for _ in range(30):
        sim.step(dt=1.0)
    gfp = snap_channel(sim, mode=1, exposure=60.0)
    camp = snap_channel(sim, mode=2, exposure=60.0)
    img4 = composite_max(
        apply_fluorescence(gfp, GREEN), apply_fluorescence(camp, CYAN),
    )

    return [img1, img2, img3, img4]
