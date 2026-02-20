"""Showcase images for the plate_reader backend."""

from __future__ import annotations

import cv2
import numpy as np
from virtual_microscope._showcase_utils import snap_channel


def _plate_to_512(well_data: np.ndarray) -> np.ndarray:
    """Render 8x12 well data as a 512x512 RGB plate visualization."""
    # Normalize to 0-255 float
    arr = well_data.astype(np.float64)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.full_like(arr, 128.0)
    arr = arr.astype(np.uint8)

    # Create a nice plate image at 512x512
    canvas = np.full((512, 512, 3), 30, dtype=np.uint8)  # dark background

    # Well layout: 8 rows x 12 cols with padding
    pad = 20
    well_w = (512 - pad * 2) // 12
    well_h = (512 - pad * 2) // 8
    radius = min(well_w, well_h) // 2 - 2

    for r in range(8):
        for c in range(12):
            cx = pad + c * well_w + well_w // 2
            cy = pad + r * well_h + well_h // 2
            val = int(arr[r, c])
            # Color: blue → green → yellow → red (heat-like)
            if val < 85:
                color = (val * 3, 0, 0)  # blue ramp
            elif val < 170:
                t = (val - 85) * 3
                color = (255 - t, t, 0)  # blue→green
            else:
                t = (val - 170) * 3
                color = (0, 255, min(t, 255))  # green→yellow/red
            # BGR for cv2
            bgr = (color[0], color[1], color[2])
            cv2.circle(canvas, (cx, cy), radius, bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), radius, (80, 80, 80), 1, cv2.LINE_AA)

    # Convert BGR to RGB
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the plate_reader backend.

    1. Viability assay — 96-well heatmap
    2. ELISA assay
    3. Luminescence assay
    4. Viability with different seed
    """
    from virtual_microscope.backends.plate_reader import create_sim

    # 1 — Viability assay
    sim_v = create_sim(assay_type="viability", seed=seed)
    v1 = snap_channel(sim_v, mode=0, exposure=50.0)
    img1 = _plate_to_512(v1)

    # 2 — ELISA assay
    sim_e = create_sim(assay_type="elisa", seed=seed)
    e1 = snap_channel(sim_e, mode=0, exposure=50.0)
    img2 = _plate_to_512(e1)

    # 3 — Luminescence assay
    sim_l = create_sim(assay_type="fluorescence", seed=seed)
    l1 = snap_channel(sim_l, mode=0, exposure=50.0)
    img3 = _plate_to_512(l1)

    # 4 — Viability variant
    sim_v2 = create_sim(assay_type="viability", seed=seed + 5)
    v2 = snap_channel(sim_v2, mode=0, exposure=50.0)
    img4 = _plate_to_512(v2)

    del sim_v, sim_e, sim_l, sim_v2
    return [img1, img2, img3, img4]
