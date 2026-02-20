"""Showcase images for the fucci backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, RED, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the fucci backend.

    1. Brightfield — tissue monolayer
    2. RFP-Cdt1 (red) — G1 phase nuclei
    3. GFP-Geminin (green) — S/G2/M phase nuclei
    4. Composite: red G1 + green S/G2/M (FUCCI merge, yellow = early S)
    """
    from virtual_microscope.backends.fucci import create_sim

    sim = create_sim(n_cells=60, seed=seed)
    sim.enable_fucci_reporter(g1_duration=30, s_duration=15, g2_duration=10, m_duration=8)
    for _ in range(10):
        sim.step(dt=1.0)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    rfp = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(rfp, RED)

    gfp = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(gfp, GREEN)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
