"""Shared utilities for backend showcase images."""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib import colormaps as mpl_cmaps


# ── Contrast & normalization ──────────────────────────────────────────

def contrast_stretch(img: np.ndarray, plow: float = 0.5, phigh: float = 99.5) -> np.ndarray:
    """Stretch image contrast using percentile-based min/max scaling."""
    f = img.astype(np.float32)
    lo = np.percentile(f, plow)
    hi = np.percentile(f, phigh)
    if hi <= lo:
        return img
    stretched = (f - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.ndim == 3:
        return img[:, :, 0]
    return img


# ── Colormap application ─────────────────────────────────────────────

def apply_cmap(img: np.ndarray, cmap: str = "gray", stretch: bool = True) -> np.ndarray:
    """Apply a matplotlib colormap to a grayscale image.

    Returns (H, W, 3) uint8 RGB.

    For fluorescence channels (dark bg, bright signal), use colormaps like
    "Greens", "Reds", "Magentas", etc. — 0 maps to black, 255 to full color.

    For brightfield (bright bg, dark features), use "gray" or "gray_r".
    """
    gray = to_gray(img)
    if stretch:
        gray = contrast_stretch(gray)
    norm = gray.astype(np.float32) / 255.0
    cmap_obj = mpl_cmaps[cmap]
    rgba = cmap_obj(norm)  # (H, W, 4) float [0,1]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def apply_fluorescence(img: np.ndarray, color: tuple[int, int, int],
                       invert: str = "auto") -> np.ndarray:
    """Map a grayscale fluorescence image to a single RGB color.

    Feature signal → color, background → black.  Uses local background
    subtraction (Gaussian blur) to remove both the optical pipeline's baseline
    brightness and vignetting gradients, isolating only real features.

    Parameters
    ----------
    invert : "auto" | True | False
        If "auto" (default), detect whether features are darker or brighter
        than the local background and handle accordingly.  Most simulator
        fluorescence channels produce features darker than a bright baseline.
    """
    gray = to_gray(img)
    f = gray.astype(np.float32)

    # Estimate LOCAL background with large-sigma Gaussian blur.
    # This follows the vignetting curve while smoothing over small features.
    sigma = max(f.shape) // 10  # ~50 px for 512x512
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma)

    if invert == "auto":
        # Features darker than local bg → absorption-like → need inversion
        diff = f - bg
        below = (diff < -2).sum()
        above = (diff > 2).sum()
        invert = bool(below >= above)

    if invert:
        signal = np.clip(bg - f, 0, None)
    else:
        signal = np.clip(f - bg, 0, None)

    # Stretch signal to 0-255
    hi = np.percentile(signal, 99.5)
    if hi > 0:
        signal = signal / hi * 255.0
    signal = np.clip(signal, 0, 255)

    norm = signal / 255.0
    out = np.zeros((*gray.shape, 3), dtype=np.uint8)
    out[:, :, 0] = (norm * color[0]).clip(0, 255).astype(np.uint8)
    out[:, :, 1] = (norm * color[1]).clip(0, 255).astype(np.uint8)
    out[:, :, 2] = (norm * color[2]).clip(0, 255).astype(np.uint8)
    return out


# Standard microscopy false-color presets (R, G, B)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)


# ── Compositing ──────────────────────────────────────────────────────

def composite_max(*channels: np.ndarray) -> np.ndarray:
    """Max-intensity projection of multiple RGB images (fluorescence merge)."""
    result = channels[0].copy()
    for ch in channels[1:]:
        result = np.maximum(result, ch)
    return result


def composite_min(*channels: np.ndarray) -> np.ndarray:
    """Min-intensity projection (for inverted / brightfield-style merge)."""
    result = channels[0].copy()
    for ch in channels[1:]:
        result = np.minimum(result, ch)
    return result


def composite_add(*channels: np.ndarray) -> np.ndarray:
    """Additive merge of multiple (H, W, 3) uint8 RGB images, clamped to 255."""
    acc = np.zeros_like(channels[0], dtype=np.uint16)
    for ch in channels:
        acc += ch.astype(np.uint16)
    return np.clip(acc, 0, 255).astype(np.uint8)


# ── Image utilities ──────────────────────────────────────────────────

def gray_to_rgb(img: np.ndarray, stretch: bool = True) -> np.ndarray:
    """Ensure image is (H, W, 3) uint8 RGB, with optional contrast stretching."""
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 2:
        if stretch:
            img = contrast_stretch(img)
        return np.stack([img, img, img], axis=-1)
    return img


def make_montage(images: list[np.ndarray], gap: int = 2) -> np.ndarray:
    """Assemble images into a horizontal strip with dark gap between panels."""
    n = len(images)
    h, w = images[0].shape[:2]
    out_w = n * w + (n - 1) * gap
    canvas = np.zeros((h, out_w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        x = i * (w + gap)
        rgb = gray_to_rgb(img, stretch=False)
        canvas[:, x : x + w] = rgb
    return canvas


# ── Snapping helpers ─────────────────────────────────────────────────

def snap_channel(sim, mode: int, exposure: float = 50.0, intensity: float = 1.0) -> np.ndarray:
    """Snap a single channel by setting sim.mode directly and calling snap_frame."""
    sim.mode = mode
    return sim.snap_frame(exposure=exposure, intensity=intensity)


def voronoi_showcase(
    sim,
    nuc_color: tuple[int, int, int] = GREEN,
    mem_color: tuple[int, int, int] = MAGENTA,
    n_steps: int = 5,
) -> list[np.ndarray]:
    """Standard 4-image showcase for DynamicVoronoiSim-based backends.

    1. Brightfield (gray)
    2. Nucleus channel (nuc_color)
    3. Membrane channel (mem_color)
    4. Composite: nucleus + membrane max projection
    """
    bf = snap_channel(sim, mode=0, exposure=80.0)
    img1 = apply_cmap(bf, "gray")

    nuc = snap_channel(sim, mode=1, exposure=100.0)
    img2 = apply_fluorescence(nuc, nuc_color)

    if n_steps > 0 and hasattr(sim, "step"):
        for _ in range(n_steps):
            sim.step(dt=1.0)
    mem = snap_channel(sim, mode=2, exposure=100.0)
    img3 = apply_fluorescence(mem, mem_color)

    nuc2 = snap_channel(sim, mode=1, exposure=100.0)
    nuc_rgb = apply_fluorescence(nuc2, nuc_color)
    img4 = composite_max(nuc_rgb, img3)

    return [img1, img2, img3, img4]
