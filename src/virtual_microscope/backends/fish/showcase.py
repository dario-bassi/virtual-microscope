"""Showcase images for the fish (FISH probes) backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_color, composite_max, snap_channel, CYAN, ORANGE,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the fish backend.

    Shows actual FISH probes (mode 3) instead of generic nucleus/membrane.
    1. DAPI nuclei (cyan)
    2. FISH probes (orange dots inside nuclei)
    3. Composite DAPI + FISH showing copy number variation
    4. Different gene config (higher amplification)
    """
    from virtual_microscope.backends.fish import create_sim

    sim = create_sim(n_cells=40, seed=seed)
    sim.enable_fish_probes(
        locus_copies=2, amplified_fraction=0.15, deleted_fraction=0.10,
    )

    # 1 — DAPI nuclei (cyan)
    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img1 = apply_color(nuc, CYAN)

    # 2 — FISH probes (orange) — mode 3 is the registered FISH channel
    fish_ch = snap_channel(sim, mode=3, exposure=60.0)
    img2 = apply_color(fish_ch, ORANGE)

    # 3 — Composite DAPI + FISH
    img3 = composite_max(img1, img2)

    # 4 — Different gene config: more amplification, more copies
    sim2 = create_sim(n_cells=40, seed=seed + 5)
    sim2.enable_fish_probes(
        locus_copies=3, amplified_fraction=0.25, deleted_fraction=0.05,
        amplified_copies_range=(4, 8),
    )
    nuc2 = snap_channel(sim2, mode=1, exposure=60.0)
    fish2 = snap_channel(sim2, mode=3, exposure=60.0)
    img4 = composite_max(apply_color(nuc2, CYAN), apply_color(fish2, ORANGE))

    del sim2
    return [img1, img2, img3, img4]
