"""Showcase images for the malaria backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the malaria backend.

    Native Giemsa RGB staining — different parasitemia levels.
    1. Low parasitemia (~5%) — ring stages
    2. Higher parasitemia (~15%)
    3. Different field of view
    4. Very high parasitemia (~25%)
    """
    from virtual_microscope.backends.malaria import create_sim

    # 1 — Low parasitemia
    sim = create_sim(world_size=512, n_rbc=2000, parasitemia=0.05, seed=seed, internal_scale=4)
    img1 = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0), stretch=False)

    # 2 — Higher parasitemia
    sim2 = create_sim(world_size=512, n_rbc=2000, parasitemia=0.15, seed=seed, internal_scale=4)
    img2 = gray_to_rgb(snap_channel(sim2, mode=0, exposure=80.0), stretch=False)

    # 3 — Different field
    sim3 = create_sim(world_size=512, n_rbc=2000, parasitemia=0.05, seed=seed + 10, internal_scale=4)
    img3 = gray_to_rgb(snap_channel(sim3, mode=0, exposure=80.0), stretch=False)

    # 4 — Very high parasitemia
    sim4 = create_sim(world_size=512, n_rbc=2000, parasitemia=0.25, seed=seed + 3, internal_scale=4)
    img4 = gray_to_rgb(snap_channel(sim4, mode=0, exposure=80.0), stretch=False)

    del sim, sim2, sim3, sim4
    return [img1, img2, img3, img4]
