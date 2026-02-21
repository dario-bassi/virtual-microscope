"""Showcase images for the malaria backend."""

from __future__ import annotations

import cv2
import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the malaria backend.

    Differentiates from blood_smear by showing parasitemia progression
    and zoomed-in views of infected RBCs with visible parasites.
    1. Low parasitemia (~5%) Giemsa overview
    2. High parasitemia (~20%) — many infected RBCs
    3. Zoomed detail showing parasite ring/trophozoite inside individual RBCs
    4. Zoomed view at high parasitemia
    """
    from virtual_microscope.backends.malaria import create_sim

    # 1 — Low parasitemia overview
    sim = create_sim(world_size=512, n_rbc=2000, parasitemia=0.05, seed=seed, internal_scale=4)
    img1 = gray_to_rgb(snap_channel(sim, mode=0, exposure=80.0), stretch=False)

    # 2 — High parasitemia overview
    sim2 = create_sim(world_size=512, n_rbc=2000, parasitemia=0.20, seed=seed, internal_scale=4)
    img2 = gray_to_rgb(snap_channel(sim2, mode=0, exposure=80.0), stretch=False)

    # 3 — Zoomed detail: larger world, offset camera, crop center
    sim3 = create_sim(world_size=1024, n_rbc=3000, parasitemia=0.10, seed=seed + 5, internal_scale=4)
    sim3.camera_offset = np.array([80.0, 60.0])
    full3 = gray_to_rgb(snap_channel(sim3, mode=0, exposure=80.0), stretch=False)
    h, w = full3.shape[:2]
    crop3 = full3[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    img3 = cv2.resize(crop3, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # 4 — Zoomed high parasitemia detail
    sim4 = create_sim(world_size=1024, n_rbc=3000, parasitemia=0.20, seed=seed + 8, internal_scale=4)
    sim4.camera_offset = np.array([-60.0, 80.0])
    full4 = gray_to_rgb(snap_channel(sim4, mode=0, exposure=80.0), stretch=False)
    crop4 = full4[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    img4 = cv2.resize(crop4, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    del sim, sim2, sim3, sim4
    return [img1, img2, img3, img4]
