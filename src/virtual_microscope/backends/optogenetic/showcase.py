"""Showcase images for the optogenetic backend.

4-frame timeseries showing cells being steered upward by per-cell SLM
stimulation at their top edge.  Each frame is a brightfield image with the
stimulation mask overlaid in blue and cell tracks showing displacement.
"""

from __future__ import annotations

import cv2
import numpy as np
from virtual_microscope._showcase_utils import apply_cmap, snap_channel


def _build_top_mask(sim) -> np.ndarray:
    """Create an SLM mask that illuminates only the top edge of each cell.

    For every cell a small disc is drawn ~base_radius above the cell center,
    so stimulated vertices pull the cell upward.
    """
    h, w = sim.viewport_height, sim.viewport_width
    mask = np.zeros((h, w), dtype=np.uint8)
    offset = tuple(sim.camera_offset)
    r_spot = int(sim.base_radius * 0.55)

    for cell in sim._cells:
        cx = int(cell.center[0] - offset[0])
        cy = int(cell.center[1] - offset[1]) - int(sim.base_radius * 0.8)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(mask, (cx, cy), r_spot, 255, -1)
    return mask


def _get_centers(sim) -> np.ndarray:
    """Return (N, 2) array of cell centers in viewport coordinates."""
    offset = np.array(sim.camera_offset)
    return np.array([c.center - offset for c in sim._cells])


def _overlay_mask(bf_rgb: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int] = (80, 140, 255),
                  alpha: float = 0.45) -> np.ndarray:
    """Overlay *mask* onto *bf_rgb* as a semi-transparent colour wash."""
    out = bf_rgb.copy()
    region = mask > 0
    overlay = np.zeros_like(out)
    overlay[region, 0] = color[0]
    overlay[region, 1] = color[1]
    overlay[region, 2] = color[2]
    out[region] = (out[region].astype(np.float32) * (1 - alpha)
                   + overlay[region].astype(np.float32) * alpha
                   ).clip(0, 255).astype(np.uint8)
    return out


def _draw_tracks(img: np.ndarray, history: list[np.ndarray],
                 color: tuple[int, int, int] = (0, 220, 255),
                 thickness: int = 1) -> np.ndarray:
    """Draw polyline tracks from position history onto *img*.

    Skips segments longer than 80 px (world-wrap artefacts).
    """
    out = img.copy()
    if len(history) < 2:
        return out
    n_cells = history[0].shape[0]
    for i in range(n_cells):
        pts = [h[i].astype(int) for h in history]
        # origin marker (small open circle)
        cv2.circle(out, tuple(pts[0]), 3, color, 1, cv2.LINE_AA)
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) > 80:
                continue  # skip world-wrap jumps
            cv2.line(out, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)
        # filled dot at current position
        cv2.circle(out, tuple(pts[-1]), 3, color, -1, cv2.LINE_AA)
    return out


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 BF+mask+tracks timeseries frames showing upward cell steering."""
    from virtual_microscope.backends.optogenetic import create_sim

    sim = create_sim(n_cells=30, world_size=512, base_radius=20.0, seed=seed)
    sim.auto_step = False

    steps_between = 120
    dt = 0.01
    frames: list[np.ndarray] = []
    history: list[np.ndarray] = []

    for _ in range(4):
        # record positions
        history.append(_get_centers(sim))

        # render BF + mask overlay + tracks
        mask = _build_top_mask(sim)
        bf = snap_channel(sim, mode=0, exposure=50.0)
        bf_rgb = apply_cmap(bf, "gray")
        img = _overlay_mask(bf_rgb, mask)
        img = _draw_tracks(img, history)
        frames.append(img)

        # stimulate + advance physics
        for _ in range(steps_between):
            mask = _build_top_mask(sim)
            sim.apply_optogenetic_stimulator(mask)
            sim.step(dt=dt)

    return frames
