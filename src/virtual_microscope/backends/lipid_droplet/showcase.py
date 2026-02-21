"""Showcase images for the lipid_droplet backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_color, composite_max, snap_channel, CYAN, YELLOW,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the lipid_droplet backend.

    Shows actual BODIPY-stained lipid droplets (mode 3) instead of generic membrane.
    1. BF showing hepatocyte morphology
    2. Nile Red / BODIPY lipid droplets (yellow) — round droplets
    3. DAPI nuclei (cyan) — displaced by steatotic droplets
    4. Composite Nile Red + DAPI
    """
    from virtual_microscope.backends.lipid_droplet import create_sim

    sim = create_sim(n_cells=25, seed=seed, steatotic_fraction=0.4)
    sim.enable_lipid_droplets(
        normal_n_range=(1, 5), steatotic_n_range=(15, 40),
        steatotic_fraction=0.4, radius_range=(3.0, 8.0),
    )

    # 1 — BF showing hepatocyte morphology
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — Nile Red / BODIPY lipid droplets (yellow) — mode 3 is BODIPY channel
    ld = snap_channel(sim, mode=3, exposure=60.0)
    img2 = apply_color(ld, YELLOW)

    # 3 — DAPI nuclei (cyan)
    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img3 = apply_color(nuc, CYAN)

    # 4 — Composite Nile Red + DAPI
    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
