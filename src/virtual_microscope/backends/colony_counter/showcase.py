"""Showcase images for the colony_counter backend."""

from __future__ import annotations

import cv2
import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the colony_counter backend.

    1-2: Whole-plate overview (spread, xgal)
    3-4: Zoomed-in detail of colony morphology (crop center quarter + resize)
    """
    from virtual_microscope.backends.colony_counter import create_sim

    # 1 — Whole-plate spread
    sim = create_sim(n_colonies=200, plate_type="spread", seed=seed)
    img1 = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0))

    # 2 — Whole-plate X-gal staining
    sim_blue = create_sim(
        n_colonies=200, plate_type="spread", seed=seed,
        staining="xgal", blue_fraction=0.3,
    )
    img2 = gray_to_rgb(snap_channel(sim_blue, mode=1, exposure=80.0))

    # 3 — Zoomed BF detail (crop center quarter → upscale to 512)
    bf_full = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0))
    h, w = bf_full.shape[:2]
    crop3 = bf_full[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    img3 = cv2.resize(crop3, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # 4 — Zoomed X-gal detail
    xgal_full = gray_to_rgb(snap_channel(sim_blue, mode=1, exposure=80.0))
    crop4 = xgal_full[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    img4 = cv2.resize(crop4, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    del sim, sim_blue
    return [img1, img2, img3, img4]
