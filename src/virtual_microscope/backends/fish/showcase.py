"""Showcase images for the fish (FISH probes) backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, MAGENTA,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the fish backend."""
    from virtual_microscope.backends.fish import create_sim

    sim = create_sim(n_cells=40, seed=seed)
    sim.enable_fish_probes(locus_copies=2, amplified_fraction=0.15, deleted_fraction=0.10)

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(nuc, GREEN)

    mem = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(mem, MAGENTA)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
