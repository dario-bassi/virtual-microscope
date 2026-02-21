"""Showcase images for the fucci backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, RED, GREEN,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the fucci backend.

    Uses apply_color (no bg subtraction) for clear FUCCI cell cycle visualization.
    1. Brightfield — tissue monolayer
    2. RFP-Cdt1 (red) — G1 phase nuclei bright
    3. mVenus-Geminin (green) — S/G2/M phase membranes bright
    4. Composite: red G1 + green S/G2/M (yellow = early S overlap)
    """
    from virtual_microscope.backends.fucci import create_sim

    sim = create_sim(n_cells=60, seed=seed)
    sim.enable_fucci_reporter(g1_duration=30, s_duration=15, g2_duration=10, m_duration=8)
    # Run more steps for good phase distribution
    for _ in range(25):
        sim.step(dt=1.0)

    # 1 — Brightfield tissue
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — RFP-Cdt1 (red hot LUT) — G1 cells bright
    rfp = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_color(rfp, RED)

    # 3 — mVenus-Geminin (green hot LUT) — S/G2/M cells bright
    gfp = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_color(gfp, GREEN)

    # 4 — Composite: red + green + yellow mix
    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
