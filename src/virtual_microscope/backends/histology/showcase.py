"""Showcase images for the histology backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the histology backend.

    Native H&E RGB — different grades showing disease progression.
    1. Normal glandular tissue (grade 0)
    2. Low-grade dysplasia (grade 1)
    3. High-grade dysplasia (grade 2)
    4. Different tissue field (varied morphology)
    """
    from virtual_microscope.backends.histology import create_sim

    # 1 — Normal glandular tissue
    sim = create_sim(tissue_type="glandular", seed=seed, grade=0, internal_scale=4)
    img1 = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0), stretch=False)

    # 2 — Low-grade
    sim2 = create_sim(tissue_type="glandular", seed=seed, grade=1, internal_scale=4)
    img2 = gray_to_rgb(snap_channel(sim2, mode=0, exposure=80.0), stretch=False)

    # 3 — High-grade
    sim3 = create_sim(tissue_type="glandular", seed=seed, grade=2, internal_scale=4)
    img3 = gray_to_rgb(snap_channel(sim3, mode=0, exposure=80.0), stretch=False)

    # 4 — Different field
    sim4 = create_sim(tissue_type="glandular", seed=seed + 10, grade=1, internal_scale=4)
    img4 = gray_to_rgb(snap_channel(sim4, mode=0, exposure=80.0), stretch=False)

    del sim, sim2, sim3, sim4
    return [img1, img2, img3, img4]
