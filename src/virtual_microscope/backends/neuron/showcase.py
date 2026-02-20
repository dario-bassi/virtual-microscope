"""Showcase images for the neuron backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, MAGENTA,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the neuron backend.

    1. Phase contrast — soma + dendrites + axon
    2. MAP2 dendrites (green) — full arbor
    3. Synaptophysin puncta (magenta) — synaptic markers
    4. Composite: green dendrites + magenta synapses merged
    """
    from virtual_microscope.backends.neuron import create_sim

    sim = create_sim(n_neurons=8, world_size=512, seed=seed, internal_scale=4)

    # 1 — Phase contrast
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — MAP2 dendrites (green)
    map2 = snap_channel(sim, mode=1, exposure=40.0)
    img2 = apply_fluorescence(map2, GREEN)

    # 3 — Synaptophysin puncta (magenta)
    syn = snap_channel(sim, mode=2, exposure=40.0)
    img3 = apply_fluorescence(syn, MAGENTA)

    # 4 — Composite: dendrites (green) + synapses (magenta)
    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
