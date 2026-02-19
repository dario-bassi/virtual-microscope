"""
GelDocSim — Gel electrophoresis / Western blot imaging simulator.

Simulates gel images as they would appear on a gel documentation system.
Supports:
  - SDS-PAGE (Coomassie blue staining) — blue bands on pale background
  - Western blot (chemiluminescent) — bright bands on dark background
  - Agarose (EtBr/UV transillumination) — bright bands on dark background

This is fundamentally different from microscopy: instead of cells in a
spatial field, the agent sees protein/DNA bands in gel lanes and must perform
band detection, molecular weight calibration, and densitometry.

Single channel: always returns the gel image.
Objective zoom: 10x=full gel, 20x=4 lanes, 40x=2 lanes.
No stage panning — always centered.

Usage via SimulationBridge:
    sim = GelDocSim(n_lanes=8, gel_type='western', seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2


# Molecular weight ladder (kDa) — standard protein ladder
MW_LADDER_BROAD = [250, 150, 100, 75, 50, 37, 25, 20, 15, 10]
MW_LADDER_PRECISION = [260, 160, 110, 80, 60, 50, 40, 30, 20, 15, 10]

# DNA size ladder (bp)
DNA_LADDER_1KB = [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 1000, 750, 500, 250]
DNA_LADDER_100BP = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]

# Migration distance follows log-linear relationship:
# distance = a - b * log10(MW)   (higher MW → less migration)

# Gel color schemes
COOMASSIE_BG = np.array([220, 225, 235], dtype=np.float32)     # pale blue-gray
COOMASSIE_BAND = np.array([40, 50, 130], dtype=np.float32)     # Coomassie blue
COOMASSIE_LANE_BG = np.array([230, 232, 240], dtype=np.float32)  # lane background

WESTERN_BG = np.array([15, 15, 20], dtype=np.float32)          # dark background
WESTERN_BAND = np.array([240, 240, 220], dtype=np.float32)     # chemiluminescence
WESTERN_LANE_BG = np.array([18, 18, 22], dtype=np.float32)     # lane background

AGAROSE_BG = np.array([10, 10, 12], dtype=np.float32)          # UV transilluminator dark
AGAROSE_BAND = np.array([255, 180, 60], dtype=np.float32)      # EtBr orange fluorescence
AGAROSE_LANE_BG = np.array([14, 12, 14], dtype=np.float32)     # lane background


# ── Protein experiment presets ──────────────────────────────────────────

DEFAULT_EXPERIMENT = {
    "name": "Drug dose-response",
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Control", "bands": {50: 1.0, 37: 0.8, 25: 0.3}},
        {"label": "Drug 1x", "bands": {50: 0.7, 37: 0.8, 25: 0.3}},
        {"label": "Drug 2x", "bands": {50: 0.4, 37: 0.7, 25: 0.3}},
        {"label": "Drug 4x", "bands": {50: 0.15, 37: 0.6, 25: 0.3}},
        {"label": "Drug 8x", "bands": {50: 0.05, 37: 0.4, 25: 0.3}},
        {"label": "Recovery", "bands": {50: 0.6, 37: 0.75, 25: 0.3}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

APOPTOSIS_EXPERIMENT = {
    "name": "Apoptosis markers (UV-induced)",
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Ctrl 2h", "bands": {116: 0.9, 89: 0.05, 42: 0.85, 26: 0.7, 19: 0.05, 17: 0.05, 12: 0.1}},
        {"label": "UV 2h", "bands": {116: 0.5, 89: 0.4, 42: 0.80, 26: 0.4, 19: 0.3, 17: 0.25, 12: 0.5}},
        {"label": "UV 4h", "bands": {116: 0.2, 89: 0.7, 42: 0.75, 26: 0.2, 19: 0.6, 17: 0.5, 12: 0.8}},
        {"label": "UV 6h", "bands": {116: 0.08, 89: 0.85, 42: 0.70, 26: 0.08, 19: 0.8, 17: 0.7, 12: 0.9}},
        {"label": "UV+zVAD", "bands": {116: 0.8, 89: 0.1, 42: 0.82, 26: 0.6, 19: 0.08, 17: 0.06, 12: 0.2}},
        {"label": "Ctrl 6h", "bands": {116: 0.85, 89: 0.06, 42: 0.83, 26: 0.65, 19: 0.06, 17: 0.05, 12: 0.12}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

SIGNALING_EXPERIMENT = {
    "name": "EGF signaling time course",
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Serum-", "bands": {60: 0.05, 56: 0.8, 54: 0.05, 46: 0.05, 44: 0.08, 42: 0.85}},
        {"label": "EGF 5m", "bands": {60: 0.9, 56: 0.8, 54: 0.1, 46: 0.15, 44: 0.95, 42: 0.85}},
        {"label": "EGF 15m", "bands": {60: 0.7, 56: 0.8, 54: 0.08, 46: 0.1, 44: 0.75, 42: 0.85}},
        {"label": "EGF 30m", "bands": {60: 0.35, 56: 0.8, 54: 0.05, 46: 0.06, 44: 0.4, 42: 0.85}},
        {"label": "EGF 60m", "bands": {60: 0.15, 56: 0.8, 54: 0.04, 46: 0.05, 44: 0.15, 42: 0.85}},
        {"label": "EGF+U0", "bands": {60: 0.85, 56: 0.8, 54: 0.04, 46: 0.05, 44: 0.05, 42: 0.85}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

EXPRESSION_EXPERIMENT = {
    "name": "Recombinant protein expression",
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Uninduced", "bands": {75: 0.3, 50: 0.5, 37: 0.15, 25: 0.6, 20: 0.4, 15: 0.3}},
        {"label": "IPTG 1h", "bands": {75: 0.3, 50: 0.5, 37: 0.4, 25: 0.6, 20: 0.4, 15: 0.3}},
        {"label": "IPTG 2h", "bands": {75: 0.28, 50: 0.45, 37: 0.7, 25: 0.55, 20: 0.38, 15: 0.28}},
        {"label": "IPTG 4h", "bands": {75: 0.25, 50: 0.4, 37: 0.95, 25: 0.5, 20: 0.35, 15: 0.25}},
        {"label": "Soluble", "bands": {75: 0.1, 50: 0.3, 37: 0.6, 25: 0.4, 20: 0.2, 15: 0.15}},
        {"label": "Pellet", "bands": {75: 0.15, 50: 0.1, 37: 0.35, 25: 0.1, 20: 0.15, 15: 0.1}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

# ── DNA experiment presets ──────────────────────────────────────────────

# Restriction digest: plasmid cut with different enzymes
# Vector = 4500 bp, insert = 1200 bp. Different enzymes yield different fragments.
RESTRICTION_DIGEST_EXPERIMENT = {
    "name": "Restriction digest verification",
    "is_dna": True,
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Uncut", "bands": {5700: 0.9}},
        {"label": "EcoRI", "bands": {4500: 0.85, 1200: 0.6}},
        {"label": "BamHI", "bands": {3200: 0.8, 2500: 0.75}},
        {"label": "EcoRI+BamHI", "bands": {3200: 0.8, 1300: 0.65, 1200: 0.6}},
        {"label": "HindIII", "bands": {5700: 0.85}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

# PCR product analysis: different primer pairs, expected sizes
PCR_EXPERIMENT = {
    "name": "PCR product analysis",
    "is_dna": True,
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Gene A", "bands": {850: 0.9}},
        {"label": "Gene B", "bands": {1200: 0.85}},
        {"label": "Gene C", "bands": {420: 0.95}},
        {"label": "Multiplex", "bands": {1200: 0.6, 850: 0.7, 420: 0.8}},
        {"label": "No template", "bands": {}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

# Cloning verification: colony PCR screening for insert
CLONING_EXPERIMENT = {
    "name": "Colony PCR insert screening",
    "is_dna": True,
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Clone 1", "bands": {1500: 0.8}},
        {"label": "Clone 2", "bands": {300: 0.7}},
        {"label": "Clone 3", "bands": {1500: 0.75}},
        {"label": "Clone 4", "bands": {1500: 0.85}},
        {"label": "Clone 5", "bands": {300: 0.65}},
        {"label": "+ctrl", "bands": {1500: 0.9}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

# RT-PCR: gene expression across tissues
RT_PCR_EXPERIMENT = {
    "name": "RT-PCR tissue expression",
    "is_dna": True,
    "conditions": [
        {"label": "Ladder", "is_ladder": True},
        {"label": "Brain", "bands": {600: 0.9, 300: 0.85}},
        {"label": "Liver", "bands": {600: 0.3, 300: 0.9}},
        {"label": "Heart", "bands": {600: 0.7, 300: 0.4}},
        {"label": "Kidney", "bands": {600: 0.5, 300: 0.6}},
        {"label": "Lung", "bands": {600: 0.15, 300: 0.75}},
        {"label": "Ladder", "is_ladder": True},
    ],
}

# Lookup by name
EXPERIMENT_PRESETS = {
    "dose_response": DEFAULT_EXPERIMENT,
    "apoptosis": APOPTOSIS_EXPERIMENT,
    "signaling": SIGNALING_EXPERIMENT,
    "expression": EXPRESSION_EXPERIMENT,
    "restriction_digest": RESTRICTION_DIGEST_EXPERIMENT,
    "pcr": PCR_EXPERIMENT,
    "cloning": CLONING_EXPERIMENT,
    "rt_pcr": RT_PCR_EXPERIMENT,
}


class GelDocSim:
    """Gel electrophoresis / western blot simulator.

    Parameters
    ----------
    n_lanes : int
        Number of lanes (including ladder lanes).
    gel_type : str
        'coomassie' for SDS-PAGE or 'western' for chemiluminescent blot.
    experiment : dict, optional
        Experiment definition with conditions and band patterns.
    ladder : list of float
        Molecular weight markers (kDa) for the ladder lane.
    seed : int
        Random seed for reproducible noise and artifacts.
    viewport_width, viewport_height : int
        Output image dimensions.
    """

    def __init__(
        self,
        n_lanes: int = 8,
        gel_type: str = "western",
        experiment=None,
        ladder: list = None,
        seed: int = 42,
        viewport_width: int = 512,
        viewport_height: int = 512,
    ):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.n_lanes = n_lanes
        self.gel_type = gel_type
        self.rgb_mode = True
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # SimulationBridge interface
        self.mode = 0
        self.camera_offset = [0, 0]
        self.state_devices = {}
        self.current_objectiv = 10
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40}

        # Gel geometry (internal coordinates)
        self.internal_scale = 4
        self._gel_width = 512  # world units
        self._gel_height = 512
        self._iw = self._gel_width * self.internal_scale
        self._ih = self._gel_height * self.internal_scale

        # Experiment (accept dict or preset name string)
        if isinstance(experiment, str):
            self._experiment = EXPERIMENT_PRESETS.get(experiment, DEFAULT_EXPERIMENT)
        else:
            self._experiment = experiment or DEFAULT_EXPERIMENT
        self._is_dna = self._experiment.get("is_dna", False)
        self._conditions = self._experiment["conditions"][:n_lanes]
        while len(self._conditions) < n_lanes:
            self._conditions.append({"label": "Empty", "bands": {}})

        # Auto-select gel type for DNA experiments if not explicitly set
        if self._is_dna and gel_type not in ("agarose",):
            self.gel_type = "agarose"

        # Ladder — auto-select DNA or protein ladder
        if ladder:
            self._ladder = ladder
        elif self._is_dna:
            self._ladder = DNA_LADDER_1KB
        else:
            self._ladder = MW_LADDER_BROAD

        # Color scheme
        if self.gel_type == "western":
            self._bg = WESTERN_BG.copy()
            self._band_color = WESTERN_BAND.copy()
            self._lane_bg = WESTERN_LANE_BG.copy()
        elif self.gel_type == "agarose":
            self._bg = AGAROSE_BG.copy()
            self._band_color = AGAROSE_BAND.copy()
            self._lane_bg = AGAROSE_LANE_BG.copy()
        else:
            self._bg = COOMASSIE_BG.copy()
            self._band_color = COOMASSIE_BAND.copy()
            self._lane_bg = COOMASSIE_LANE_BG.copy()

        # Migration parameters (log-linear: dist = a - b * log10(size))
        if self._is_dna:
            # DNA: log10(100 bp)=2 → near bottom, log10(10000 bp)=4 → near top
            self._mig_a = 1.6
            self._mig_b = 0.38
        else:
            # Protein: MW=10 kDa near bottom, MW=250 near top
            self._mig_a = 1.4
            self._mig_b = 0.58

        # Lane geometry
        self._lane_margin = 0.06  # fraction of gel width for outer margins
        self._lane_gap = 0.015    # fraction between lanes
        self._compute_lane_geometry()

        # Gel "smile" — edge lanes migrate slightly faster (thermal gradient)
        # Migration offset per lane: edge lanes get +offset, center lanes 0
        center = (n_lanes - 1) / 2.0
        smile_strength = 0.02 + self._rng.uniform(0, 0.015)
        self._lane_mig_offset = np.array([
            smile_strength * ((i - center) / max(1, center)) ** 2
            for i in range(n_lanes)
        ])
        # Per-lane random migration jitter (loading/buffer variation)
        self._lane_mig_offset += self._rng.normal(0, 0.005, n_lanes)

        # Pre-render
        self._gel_full = None    # (ih, iw, 3) RGB gel image
        self._render_full()

    def _compute_lane_geometry(self):
        """Compute lane positions and widths."""
        n = self.n_lanes
        total_margin = 2 * self._lane_margin
        total_gap = (n - 1) * self._lane_gap
        available = 1.0 - total_margin - total_gap
        self._lane_width = available / n

        self._lane_centers = []
        for i in range(n):
            cx = self._lane_margin + (i + 0.5) * self._lane_width + i * self._lane_gap
            self._lane_centers.append(cx)

    def _mw_to_distance(self, mw_kda):
        """Convert molecular weight (kDa) to migration distance (0-1 fraction)."""
        if mw_kda <= 0:
            return 1.0
        d = self._mig_a - self._mig_b * np.log10(mw_kda)
        return float(np.clip(d, 0.02, 0.95))

    def _render_full(self):
        """Pre-render the complete gel image at internal resolution."""
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        rng = self._rng

        # === Gel image ===
        img = np.empty((ih, iw, 3), dtype=np.float32)
        img[:, :] = self._bg

        # Gel region (slight gradient top to bottom for realism)
        gel_top = int(0.08 * ih)   # wells at ~8% from top
        gel_bot = int(0.92 * ih)   # dye front at ~92%
        gel_h = gel_bot - gel_top

        # Background gradient (slightly brighter near bottom for Coomassie)
        gradient = np.linspace(0, 1, gel_h).reshape(-1, 1, 1)
        if self.gel_type == "coomassie":
            img[gel_top:gel_bot, :] = self._lane_bg * (1.0 + 0.05 * gradient)
        else:
            img[gel_top:gel_bot, :] = self._lane_bg

        # Background noise (fine grain)
        noise = rng.normal(0, 2.0, (ih, iw, 3)).astype(np.float32)
        img += noise

        # Uneven background (large-scale blotchiness)
        blotch = rng.normal(0, 3, (ih // (8 * s), iw // (8 * s))).astype(np.float32)
        blotch = cv2.GaussianBlur(blotch, (0, 0), 2)
        blotch = cv2.resize(blotch, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img += blotch[:, :, np.newaxis]

        # === Draw each lane ===
        for lane_idx, (cx_frac, condition) in enumerate(
            zip(self._lane_centers, self._conditions)
        ):
            lane_left = int((cx_frac - self._lane_width / 2) * iw)
            lane_right = int((cx_frac + self._lane_width / 2) * iw)
            lane_w = lane_right - lane_left

            # Well (loading slot at top)
            well_top = gel_top - int(0.015 * ih)
            well_bot = gel_top + int(0.01 * ih)
            well_color = self._bg * 0.85
            img[well_top:well_bot, lane_left:lane_right] = well_color

            mig_offset = self._lane_mig_offset[lane_idx]
            if condition.get("is_ladder"):
                self._draw_ladder_lane(img, lane_left, lane_right, gel_top,
                                       gel_h, rng, mig_offset=mig_offset)
            elif condition.get("bands"):
                self._draw_sample_lane(img, lane_left, lane_right, gel_top,
                                       gel_h, condition, lane_idx, rng,
                                       mig_offset=mig_offset)
            # else: empty lane (no bands)

        # === Dye front ===
        dye_y = gel_bot - int(0.02 * gel_h)
        dye_h = max(2, int(1.5 * s))
        if self.gel_type == "coomassie":
            # Bromophenol blue dye front — visible dark blue line
            dye_color = np.array([60, 60, 160], dtype=np.float32)
            dye_alpha = 0.3
        elif self.gel_type == "agarose":
            # Bromophenol blue faint under UV
            dye_color = np.array([30, 30, 60], dtype=np.float32)
            dye_alpha = 0.15
        else:
            dye_color = self._bg
            dye_alpha = 0.0
        if dye_alpha > 0:
            for c in range(3):
                img[dye_y:dye_y + dye_h, :, c] = (
                    img[dye_y:dye_y + dye_h, :, c] * (1 - dye_alpha) +
                    dye_color[c] * dye_alpha
                )

        # === Gel border ===
        border_t = max(1, s)
        border_color = self._bg * 0.7
        img[gel_top - border_t:gel_top, :] = border_color
        img[gel_bot:gel_bot + border_t, :] = border_color
        # Left/right gel edges
        gel_left = int(self._lane_margin * 0.3 * iw)
        gel_right = iw - gel_left
        img[gel_top:gel_bot, gel_left:gel_left + border_t] = border_color
        img[gel_top:gel_bot, gel_right - border_t:gel_right] = border_color

        self._gel_full = np.clip(img, 0, 255).astype(np.uint8)

    def _draw_band(self, img, cx, cy, band_width, band_height, intensity,
                   rng, smear=0.0):
        """Draw a single protein/DNA band as a Gaussian profile.

        Parameters
        ----------
        cx, cy : int
            Band center in internal pixels.
        band_width : int
            Band width (horizontal extent) in internal pixels.
        band_height : int
            Band height (vertical extent / sigma) in internal pixels.
        intensity : float
            Band intensity (0-1).
        smear : float
            Downward smearing (0-1), simulates overloading or degradation.
        """
        s = self.internal_scale
        ih, iw = self._ih, self._iw

        # Band curvature: slight parabolic smile/frown within each band
        # Higher intensity bands tend to smile more
        curvature = rng.uniform(-0.8, 1.5) * s  # pixels of y-shift at edges

        # Horizontal extent (flat with rounded edges)
        x0 = max(0, cx - band_width // 2)
        x1 = min(iw, cx + band_width // 2)
        if x0 >= x1:
            return

        xs = np.arange(x0, x1).astype(np.float32)
        h_profile = np.ones_like(xs)
        # Smooth edge falloff
        edge_w = max(1, band_width * 0.08)
        left_edge = np.clip((xs - x0) / edge_w, 0, 1)
        right_edge = np.clip((x1 - 1 - xs) / edge_w, 0, 1)
        h_profile *= left_edge * right_edge

        # Per-column y-offset for curvature (parabolic)
        x_norm = (xs - cx) / max(1, band_width / 2)  # -1..1
        y_offsets = (curvature * x_norm ** 2).astype(np.float32)

        # Vertical range (accommodate curvature)
        max_curve = int(abs(curvature) + 1)
        y0 = max(0, cy - band_height * 3 - max_curve)
        y1 = min(ih, cy + band_height * 3 + int(smear * band_height * 4) + max_curve)
        if y0 >= y1:
            return

        ys = np.arange(y0, y1).astype(np.float32)

        # 2D band with per-column center shift
        band_2d = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        for xi in range(x1 - x0):
            cy_shifted = cy + y_offsets[xi]
            col_profile = np.exp(-0.5 * ((ys - cy_shifted) / max(1, band_height)) ** 2)
            # Smear tail
            if smear > 0.01:
                tail_mask = ys > cy_shifted
                tail = np.exp(-(ys - cy_shifted) / max(1, smear * band_height * 3))
                col_profile[tail_mask] = np.maximum(col_profile[tail_mask],
                                                    tail[tail_mask] * 0.4 * smear)
            band_2d[:, xi] = col_profile * h_profile[xi]

        band_2d *= intensity

        # Per-pixel noise on the band
        band_noise = rng.normal(1.0, 0.03, band_2d.shape).astype(np.float32)
        band_2d *= band_noise

        # Apply band color
        for c in range(3):
            channel = img[y0:y1, x0:x1, c]
            if self.gel_type in ("western", "agarose"):
                # Additive (bright on dark)
                channel += band_2d * self._band_color[c]
            else:
                # Subtractive (dark on light) — Coomassie stains darker
                channel -= band_2d * (self._lane_bg[c] - self._band_color[c])

    def _draw_ladder_lane(self, img, left, right, gel_top, gel_h, rng,
                          mig_offset=0.0):
        """Draw molecular weight ladder bands."""
        s = self.internal_scale
        band_w = right - left - 4 * s
        cx = (left + right) // 2

        max_size = max(self._ladder)
        for mw in self._ladder:
            dist = self._mw_to_distance(mw) + mig_offset
            dist = np.clip(dist, 0.02, 0.95)
            cy = gel_top + int(dist * gel_h)

            # Size-dependent band height: larger fragments diffuse more
            size_factor = np.log10(mw) / np.log10(max_size)  # 0..1
            bh = max(2, int((2.0 + 1.5 * size_factor) * s))

            # Ladder bands are moderately bright, uniform
            intensity = 0.65 + rng.uniform(-0.05, 0.05)
            self._draw_band(img, cx, cy, band_w, bh, intensity, rng)

    def _draw_sample_lane(self, img, left, right, gel_top, gel_h,
                          condition, lane_idx, rng, mig_offset=0.0):
        """Draw sample bands for a given condition."""
        s = self.internal_scale
        band_w = right - left - 4 * s
        cx = (left + right) // 2

        # Per-lane loading variation
        loading = 1.0 + rng.normal(0, 0.08)

        # Reference size for band width scaling
        all_sizes = [float(k) for k in condition.get("bands", {}).keys()]
        max_size = max(all_sizes) if all_sizes else 100

        bands = condition.get("bands", {})
        for mw_kda, rel_intensity in bands.items():
            mw_kda = float(mw_kda)
            dist = self._mw_to_distance(mw_kda) + mig_offset
            dist = np.clip(dist, 0.02, 0.95)
            cy = gel_top + int(dist * gel_h)

            # Size-dependent band height: larger fragments = broader bands
            size_factor = np.log10(max(mw_kda, 1)) / np.log10(max(max_size, 10))
            bh = max(2, int((2.5 + 2.0 * size_factor) * s))

            # Scale intensity by loading and add noise
            intensity = rel_intensity * loading * (0.85 + rng.uniform(0, 0.3))

            # Higher loading = more smearing
            smear = max(0, (intensity - 0.7) * 0.3) + rng.uniform(0, 0.05)

            self._draw_band(img, cx, cy, band_w, bh, intensity, rng,
                            smear=smear)

    # -- SimulationBridge interface --

    def _update_objectif(self):
        """Update objective magnification from state_devices."""
        if "Objective" not in self.state_devices:
            return
        obj = self.state_devices["Objective"]
        lbl = obj.get("label", obj.get("Label", ""))
        if lbl in self._objectif_dict:
            self.current_objectiv = self._objectif_dict[lbl]

    def set_focal_plane(self, z):
        pass

    def snap_frame(self, mask=None, exposure=50, intensity=100, **kwargs):
        """Return a viewport-cropped gel image.

        Always returns (H, W, 3) uint8 in RGB mode.
        """
        self._update_objectif()
        return self._crop_fov(self._gel_full)

    def _crop_fov(self, full: np.ndarray) -> np.ndarray:
        """Crop FOV from high-res buffer, always centered, downsample to viewport."""
        s = self.internal_scale
        ih, iw = full.shape[:2]
        out_w, out_h = self.viewport_width, self.viewport_height
        obj = self.current_objectiv

        if obj == 40:
            fov_world = 128
        elif obj == 20:
            fov_world = 256
        else:
            fov_world = 512

        fov_internal = fov_world * s

        # Always centered — no stage panning
        cx_i = iw // 2
        cy_i = ih // 2
        half = fov_internal // 2

        x0 = cx_i - half
        y0 = cy_i - half
        x1 = x0 + fov_internal
        y1 = y0 + fov_internal

        # Clamp
        pad_l = max(0, -x0)
        pad_t = max(0, -y0)
        pad_r = max(0, x1 - iw)
        pad_b = max(0, y1 - ih)

        x0c = max(0, x0)
        y0c = max(0, y0)
        x1c = min(iw, x1)
        y1c = min(ih, y1)

        region = full[y0c:y1c, x0c:x1c]

        if pad_l or pad_t or pad_r or pad_b:
            if full.ndim == 3:
                padded = np.full((fov_internal, fov_internal, 3),
                                 self._bg.astype(np.uint8), dtype=np.uint8)
            else:
                padded = np.full((fov_internal, fov_internal),
                                 int(self._bg[0]), dtype=np.uint8)
            padded[pad_t:pad_t + (y1c - y0c),
                   pad_l:pad_l + (x1c - x0c)] = region
            region = padded

        if region.shape[0] != out_h or region.shape[1] != out_w:
            region = cv2.resize(region, (out_w, out_h),
                                interpolation=cv2.INTER_AREA)

        return region

    def get_ground_truth(self):
        """Return ground truth for grading.

        Returns dict with:
        - conditions: list of {label, bands: {mw: intensity}}
        - ladder_mw: list of ladder molecular weights
        - n_lanes: number of lanes
        - gel_type: 'western' or 'coomassie'
        - experiment_name: name of the experiment
        - band_positions: {mw_kda: migration_fraction} for all MWs
        """
        # Collect all unique MWs and their positions
        all_mws = set(self._ladder)
        for cond in self._conditions:
            if cond.get("bands"):
                all_mws.update(float(mw) for mw in cond["bands"].keys())

        band_positions = {}
        for mw in sorted(all_mws, reverse=True):
            band_positions[float(mw)] = round(self._mw_to_distance(mw), 4)

        result = {
            "experiment_name": self._experiment.get("name", "Unknown"),
            "gel_type": self.gel_type,
            "is_dna": self._is_dna,
            "n_lanes": self.n_lanes,
            "ladder_sizes": sorted(self._ladder, reverse=True),
            "size_unit": "bp" if self._is_dna else "kDa",
            "conditions": [
                {
                    "label": c.get("label", f"Lane {i+1}"),
                    "is_ladder": c.get("is_ladder", False),
                    "bands": {float(k): round(float(v), 3)
                              for k, v in c.get("bands", {}).items()},
                }
                for i, c in enumerate(self._conditions)
            ],
            "band_positions": band_positions,
        }
        # Keep backward compat
        result["ladder_mw"] = result["ladder_sizes"]
        return result
