"""Showcase images for the particle backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, MAGENTA,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the particle backend."""
    from virtual_microscope.backends.particle import create_sim

    sim = create_sim(nb_cells=50, rng_seed=seed)
    # ScatteredCellSim requires state_devices for mode selection
    sim.state_devices = {
        "Filter Wheel": {"label": "Electra1(402/454)"},
        "LED": {"label": "CYAN"},
        "Objective": {"label": "10x"},
    }

    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    sim.state_devices["Filter Wheel"]["label"] = "mScarlet3(569/582)"
    sim.state_devices["LED"]["label"] = "ORANGE"
    nuc = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(nuc, GREEN)

    sim.state_devices["Filter Wheel"]["label"] = "miRFP670(642/670)"
    sim.state_devices["LED"]["label"] = "RED"
    mem = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(mem, MAGENTA)

    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
