"""
Nuclear Texture — Heterochromatin, nucleoli, and chromatin structure.

Replaces the plain filled-circle nucleus rendering with textured nuclei
that include:
  - Heterochromatin foci (bright clumps near periphery)
  - Euchromatin (diffuse, dimmer interior)
  - Nucleoli (1-2 dark voids)
  - Nuclear envelope brightening (rim)

Works by generating a texture map for each nucleus, then masking it
into a circular region. The texture is built from multi-scale noise
modulated by radial distance, with added bright/dark spots.

Usage:
    from nuclear_texture import render_textured_nucleus

    # Render a single textured nucleus into an image
    render_textured_nucleus(
        img, cx=100, cy=100, radius=15, intensity=0.8,
        rng=np.random.default_rng(42),
    )

    # Or render all nuclei in a VoronoiSim
    render_textured_nuclei(img, sim, marker_positive, marker_intensity, rng)
"""

import numpy as np
import cv2


def render_textured_nucleus(
    img: np.ndarray,
    cx: float, cy: float, radius: float,
    intensity: float = 0.8,
    rng: np.random.Generator = None,
    n_nucleoli: int = -1,
    n_foci: int = -1,
    aspect: float = 1.0,
    orient: float = 0.0,
) -> None:
    """Render a single textured nucleus into img (in-place).

    Args:
        img: Target image (HxWx3, uint8), modified in-place
        cx, cy: Nucleus center coordinates
        radius: Nucleus radius in pixels (semi-minor axis)
        intensity: Overall intensity multiplier (0-1)
        rng: Random number generator
        n_nucleoli: Number of nucleoli (-1 = auto: 0-2 based on radius)
        n_foci: Number of heterochromatin foci (-1 = auto)
        aspect: Aspect ratio (semi-major / semi-minor). 1.0 = circle.
        orient: Orientation angle in radians (rotation of major axis).
    """
    if rng is None:
        rng = np.random.default_rng()

    r = max(3, int(radius))
    ra = max(3, int(radius * aspect))  # semi-major axis
    ix, iy = int(round(cx)), int(round(cy))
    h, w = img.shape[:2]

    # Bounding box with margin — use larger axis
    rmax = max(r, ra)
    margin = 2
    x1 = max(0, ix - rmax - margin)
    y1 = max(0, iy - rmax - margin)
    x2 = min(w, ix + rmax + margin + 1)
    y2 = min(h, iy + rmax + margin + 1)
    if x2 <= x1 or y2 <= y1:
        return

    bw = x2 - x1
    bh = y2 - y1

    # Local coordinates relative to center
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dx = xx - cx
    dy = yy - cy

    # Rotate into ellipse-aligned coordinates
    cos_a = np.cos(-orient)
    sin_a = np.sin(-orient)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    # Elliptical distance: normalized so that boundary = 1.0
    # semi-major axis = ra along rotated x, semi-minor = r along rotated y
    edist = np.sqrt((dx_rot / max(ra, 1))**2 + (dy_rot / max(r, 1))**2)

    # Elliptical mask
    mask = edist <= 1.0
    # Use elliptical distance as "normalized distance" for radial profile
    dist = edist

    if not mask.any():
        return

    # Normalized distance from center (0 = center, 1 = edge)
    # Already normalized via elliptical dist (boundary = 1.0)
    norm_dist = np.clip(dist, 0, 1)

    # Effective radius for sub-structure placement (geometric mean of axes)
    reff = max(3, int(np.sqrt(r * ra)))

    # --- Base texture: multi-scale noise ---
    # Coarse pattern (heterochromatin-like clumps)
    if reff >= 6:
        noise_size = max(3, reff // 2)
        coarse = rng.standard_normal((noise_size, noise_size)).astype(np.float32)
        coarse = cv2.resize(coarse, (bw, bh), interpolation=cv2.INTER_CUBIC)
        coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=reff * 0.3)
    else:
        coarse = rng.standard_normal((bh, bw)).astype(np.float32) * 0.3

    # Fine noise (speckle)
    fine = rng.standard_normal((bh, bw)).astype(np.float32) * 0.15
    if reff >= 5:
        fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=0.8)

    # Combine: texture = base_profile + coarse + fine
    texture = coarse * 0.35 + fine

    # --- Radial profile: brighter at periphery (envelope) and center ---
    # Base: dome shape (brightest at center, fades at edge)
    radial = 1.0 - 0.4 * norm_dist**2

    # Nuclear envelope: bright ring near edge
    envelope = np.exp(-((norm_dist - 0.85) ** 2) / (2 * 0.08**2)) * 0.3
    radial += envelope

    # --- Nucleoli: dark voids ---
    if n_nucleoli < 0:
        n_nucleoli = rng.choice([0, 1, 1, 2]) if reff >= 8 else rng.choice([0, 0, 1])

    for _ in range(n_nucleoli):
        # Place nucleolus off-center in ellipse-aligned space
        nuc_r = reff * rng.uniform(0.15, 0.25)
        # Generate position in normalized ellipse space (unit circle)
        frac = rng.uniform(0.1, 0.5)
        angle = rng.uniform(0, 2 * np.pi)
        # Position in ellipse-aligned coords, then rotate to world
        local_x = frac * ra * np.cos(angle)
        local_y = frac * r * np.sin(angle)
        cos_o = np.cos(orient)
        sin_o = np.sin(orient)
        nuc_cx = cx + local_x * cos_o - local_y * sin_o - x1
        nuc_cy = cy + local_x * sin_o + local_y * cos_o - y1

        ndx = np.arange(bw)[None, :] - nuc_cx
        ndy = np.arange(bh)[:, None] - nuc_cy
        nuc_dist_map = np.sqrt(ndx**2 + ndy**2)
        # Smooth dark void — nucleoli are clearly dark in DAPI
        void = np.exp(-(nuc_dist_map**2) / (2 * nuc_r**2)) * 0.7
        radial -= void

    # --- Heterochromatin foci: bright puncta near periphery ---
    if n_foci < 0:
        n_foci = rng.integers(2, max(3, reff // 3) + 1) if reff >= 8 else rng.integers(1, 3)

    for _ in range(n_foci):
        foci_r = reff * rng.uniform(0.08, 0.18)
        # Place in ellipse-aligned space at 40-85% of boundary
        frac = rng.uniform(0.4, 0.85)
        angle = rng.uniform(0, 2 * np.pi)
        local_x = frac * ra * np.cos(angle)
        local_y = frac * r * np.sin(angle)
        cos_o = np.cos(orient)
        sin_o = np.sin(orient)
        foci_cx = cx + local_x * cos_o - local_y * sin_o - x1
        foci_cy = cy + local_x * sin_o + local_y * cos_o - y1

        fdx = np.arange(bw)[None, :] - foci_cx
        fdy = np.arange(bh)[:, None] - foci_cy
        foci_dist_map = np.sqrt(fdx**2 + fdy**2)
        bright_spot = np.exp(-(foci_dist_map**2) / (2 * foci_r**2)) * rng.uniform(0.3, 0.6)
        radial += bright_spot

    # --- Combine and apply ---
    combined = (radial + texture) * intensity
    combined = np.clip(combined, 0, 1.2)

    # Scale to pixel values
    base_max = 200  # Same as VoronoiSim convention
    pixel_values = (combined * base_max).clip(0, 255).astype(np.uint8)

    # Apply with circular mask
    for c in range(img.shape[2] if img.ndim == 3 else 1):
        if img.ndim == 3:
            channel = img[y1:y2, x1:x2, c]
        else:
            channel = img[y1:y2, x1:x2]
        channel[mask] = pixel_values[mask]


def render_textured_nuclei(
    img: np.ndarray,
    sim,
    positive: np.ndarray | None = None,
    intensities: np.ndarray | None = None,
    rng: np.random.Generator = None,
    renderable: np.ndarray | None = None,
) -> np.ndarray:
    """Render all nuclei in a VoronoiSim with texture.

    Args:
        img: Target image (HxWx3, uint8)
        sim: VoronoiSim instance
        positive: Bool array of which cells to render (default: all)
        intensities: Float array of per-cell intensities (default: from sim)
        rng: Random generator
        renderable: Bool array; False = ghost cell, skip rendering

    Returns:
        Modified image
    """
    if rng is None:
        rng = np.random.default_rng()

    if positive is None:
        positive = sim.has_nucleus_marker
    if intensities is None:
        intensities = sim.nucleus_intensity

    # Autofluorescence background in all cells
    for i, poly in enumerate(sim.cell_polygons[:sim.nb_cells]):
        if len(poly) < 3 or (renderable is not None and not renderable[i]):
            continue
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        auto = int(rng.uniform(5, 12))
        cv2.fillPoly(img, [pts], (auto, auto, auto))

    # Get ellipsoidal parameters if available
    has_aspect = hasattr(sim, '_nuc_aspect')
    has_orient = hasattr(sim, '_nuc_orient')

    # Textured nuclei
    for i in range(sim.nb_cells):
        if renderable is not None and not renderable[i]:
            continue
        if not positive[i]:
            continue
        cx, cy = sim.cell_centroids[i]
        r = max(3, float(sim.nucleus_radii[i]))
        asp = float(sim._nuc_aspect[i]) if has_aspect else 1.0
        ori = float(sim._nuc_orient[i]) if has_orient else 0.0
        if 0 <= cx < sim.width and 0 <= cy < sim.height:
            render_textured_nucleus(
                img, cx, cy, r,
                intensity=float(intensities[i]),
                rng=rng,
                aspect=asp,
                orient=ori,
            )

    return img
