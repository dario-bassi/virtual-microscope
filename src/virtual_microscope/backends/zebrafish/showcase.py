"""Showcase images for the zebrafish backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import (
    apply_cmap, apply_fluorescence, composite_max, snap_channel, GREEN, RED,
)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the zebrafish backend.

    1. Brightfield — transparent embryo anatomy
    2. Tg(flk1:GFP) — vascular endothelium (green)
    3. Tg(myl7:mCherry) — cardiac myocytes (red)
    4. Composite: GFP vasculature + mCherry heart
    """
    from virtual_microscope.backends.zebrafish import create_sim

    sim = create_sim(n_rbc=30, cardiac_freq=2.5, seed=seed, internal_scale=2)

    # 1 — Brightfield
    bf = snap_channel(sim, mode=0, exposure=50.0)
    img1 = apply_cmap(bf, "gray")

    # 2 — GFP vasculature
    gfp = snap_channel(sim, mode=1, exposure=60.0)
    img2 = apply_fluorescence(gfp, GREEN)

    # 3 — mCherry cardiac
    mch = snap_channel(sim, mode=2, exposure=60.0)
    img3 = apply_fluorescence(mch, RED)

    # 4 — Composite: GFP + mCherry
    img4 = composite_max(img2, img3)

    return [img1, img2, img3, img4]
