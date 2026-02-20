"""Showcase images for the blood_smear backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import contrast_stretch, gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the blood_smear backend.

    All panels show native RGB Giemsa stain — no false-color channel splitting.
    1. Normal blood smear (native RGB)
    2. Smear with pathology: sickle cells + rouleaux
    3. Different field of view (different seed)
    4. Dense smear with more WBCs
    """
    from virtual_microscope.backends.blood_smear import create_sim

    # 1 — Normal blood smear
    sim = create_sim(world_size=512, n_rbc=800, n_wbc=15, seed=seed, internal_scale=4)
    bf = snap_channel(sim, mode=0, exposure=80.0)
    img1 = gray_to_rgb(bf, stretch=False)

    # 2 — Pathology: sickle cells + rouleaux
    sim2 = create_sim(
        world_size=512, n_rbc=800, n_wbc=15, seed=seed + 7,
        abnormal_rbc={"sickle": 0.08, "target": 0.05},
        rouleaux_fraction=0.15, internal_scale=4,
    )
    bf2 = snap_channel(sim2, mode=0, exposure=80.0)
    img2 = gray_to_rgb(bf2, stretch=False)

    # 3 — Different field of view
    sim3 = create_sim(world_size=512, n_rbc=800, n_wbc=15, seed=seed + 20, internal_scale=4)
    bf3 = snap_channel(sim3, mode=0, exposure=80.0)
    img3 = gray_to_rgb(bf3, stretch=False)

    # 4 — Dense smear with more WBCs
    sim4 = create_sim(world_size=512, n_rbc=1000, n_wbc=25, seed=seed + 3, internal_scale=4)
    bf4 = snap_channel(sim4, mode=0, exposure=80.0)
    img4 = gray_to_rgb(bf4, stretch=False)

    del sim, sim2, sim3, sim4
    return [img1, img2, img3, img4]
