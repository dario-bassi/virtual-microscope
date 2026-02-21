"""
BloodSmearSim — Peripheral blood smear simulation.

Simulates a Wright-Giemsa-stained blood smear with:
  - Red blood cells (RBCs): biconcave discs, ~7um diameter
  - White blood cells (WBCs): neutrophils, lymphocytes, monocytes, eosinophils
  - Platelets: tiny purple dots, often in small clusters

This is a STAINED, DRIED preparation — no dynamics, no fluorescence.
Brightfield only (RGB). The challenge is morphological classification.

Wright-Giemsa staining produces characteristic colors:
  - Background: pale pink/cream
  - RBCs: salmon pink with pale central pallor (biconcave disc)
  - WBC nuclei: dark purple/violet (basophilic chromatin)
  - WBC cytoplasm: varies by type (pale blue, pink, gray-blue)
  - Neutrophil granules: fine pink/lilac
  - Eosinophil granules: bright orange-red (refractile)
  - Platelets: purple granular

High-resolution rendering: all channels rendered at internal_scale x world_size
(default 4x), so higher objectives reveal genuine subcellular detail.

Usage via SimulationBridge:
    sim = BloodSmearSim(world_size=512, seed=42, internal_scale=4)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class BloodSmearSim(SimBase):
    """Peripheral blood smear simulation with high-resolution rendering.

    Channels:
      - mode 0: Brightfield (Wright-Giemsa stain) — the ONLY useful channel
      - mode 1: "nucleus channel" — renders WBC nuclei only (training aid)
      - mode 2: "membrane channel" — renders RBC outlines only (training aid)

    The primary workflow is brightfield-only morphological classification.
    """

    # WBC subtypes with RGB color parameters (Wright-Giemsa stain)
    WBC_TYPES = {
        "neutrophil": {
            "fraction": 0.60,
            "size_um": 13.0,
            "nucleus_lobes": (2, 5),
            "nucleus_rgb": (68, 40, 105),      # dark blue-purple
            "cyto_rgb": (220, 195, 210),       # pale pink
            "granule_rgb": (200, 165, 185),    # fine pink/lilac granules
        },
        "lymphocyte": {
            "fraction": 0.30,
            "size_um": 9.0,
            "nucleus_lobes": (1, 1),
            "nucleus_rgb": (48, 30, 85),       # very dark blue-purple
            "cyto_rgb": (185, 195, 220),       # thin pale blue rim
            "granule_rgb": None,
        },
        "monocyte": {
            "fraction": 0.05,
            "size_um": 17.0,
            "nucleus_lobes": (1, 1),
            "nucleus_rgb": (82, 55, 115),      # lighter blue-purple
            "cyto_rgb": (195, 195, 210),       # blue-gray
            "granule_rgb": None,
        },
        "eosinophil": {
            "fraction": 0.04,
            "size_um": 13.0,
            "nucleus_lobes": (2, 2),
            "nucleus_rgb": (62, 35, 100),      # dark blue-purple
            "cyto_rgb": (225, 210, 215),       # pale
            "granule_rgb": (235, 130, 50),     # vivid orange-red (coarse)
        },
        "basophil": {
            "fraction": 0.01,
            "size_um": 12.0,
            "nucleus_lobes": (2, 2),
            "nucleus_rgb": (30, 18, 60),       # very dark blue-purple (obscured)
            "cyto_rgb": (195, 185, 215),       # pale lilac
            "granule_rgb": (22, 12, 55),       # very dark blue-black granules
        },
    }

    # Abnormal RBC morphology types
    RBC_MORPHOLOGIES = ["normal", "sickle", "target", "spherocyte"]

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_rbc: int = 600,
        n_wbc: int = 12,
        n_platelets: int = 20,
        wbc_differential: dict = None,
        abnormal_rbc: dict = None,
        rouleaux_fraction: float = 0.0,
        seed: int = 42,
        fixed_dt: float = 0.0,
        internal_scale: int = 4,
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=fixed_dt,
            auto_step=False, snaps_per_step=2,
            mode_map={
                ("mScarlet3(569/582)", "ORANGE"): 1,   # nuclei-aid
                ("miRFP670(642/670)", "RED"): 2,       # membrane-aid
            },
        )

        # Override DOF table (blood smear uses 0.5 for 100x instead of default 0.6)
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.5}

        self._noise_rng = np.random.default_rng(seed + 7777)

        # Cell counts
        self.n_rbc = n_rbc
        self.n_wbc = n_wbc
        self.n_platelets = n_platelets

        # WBC differential (fractions)
        if wbc_differential is not None:
            self._wbc_diff = wbc_differential
        else:
            self._wbc_diff = {k: v["fraction"] for k, v in self.WBC_TYPES.items()}

        # Abnormal RBC morphology: {"sickle": 0.05, "target": 0.03, "spherocyte": 0.02}
        self._abnormal_rbc = abnormal_rbc or {}
        self._rouleaux_fraction = rouleaux_fraction

        # RGB camera mode (Wright-Giemsa staining is color)
        self.rgb_mode = True

        # Giemsa stain colors (R, G, B) — calibrated to real Wright-Giemsa
        self._bg_color = np.array([236, 230, 230], dtype=np.float32)  # neutral pinkish-gray
        self._rbc_body = np.array([218, 158, 155], dtype=np.float32)  # salmon-pink (not orange)
        self._rbc_pallor = np.array([234, 220, 220], dtype=np.float32)  # stark pale pink center
        self._rbc_edge = np.array([198, 140, 140], dtype=np.float32)  # darker pink rim
        self._plt_color = np.array([130, 85, 145], dtype=np.float32)  # purple granules

        # Optical pipeline (single instance, not per-channel dict)
        self._blood_pipeline = OpticalPipeline()
        self._blood_pipeline.noise = {"photon_scale": 400, "read_noise": 2.0}
        self._blood_pipeline.vignette = 0.06  # subtle vignetting

        # Generate cells
        self._generate_cells()

        # Pre-render full images at internal resolution
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None
        self._render_full()

    def _generate_cells(self):
        """Generate positions and types for all blood cells (world coordinates)."""
        margin = 20

        # -- RBCs -- quasi-regular monolayer via jittered grid
        avg_spacing = (self.width - 2 * margin) / np.sqrt(self.n_rbc)
        grid_n = int(np.ceil(np.sqrt(self.n_rbc)))
        xs = np.linspace(margin + avg_spacing / 2, self.width - margin - avg_spacing / 2, grid_n)
        ys = np.linspace(margin + avg_spacing / 2, self.height - margin - avg_spacing / 2, grid_n)
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        # Shuffle and take n_rbc points
        idx_all = self.rng.permutation(len(gx))[:self.n_rbc]
        self._rbc_x = gx[idx_all] + self.rng.normal(0, avg_spacing * 0.25, self.n_rbc)
        self._rbc_y = gy[idx_all] + self.rng.normal(0, avg_spacing * 0.25, self.n_rbc)
        self._rbc_x = np.clip(self._rbc_x, margin, self.width - margin)
        self._rbc_y = np.clip(self._rbc_y, margin, self.height - margin)
        self._rbc_radius = self.rng.normal(3.5, 0.4, self.n_rbc).clip(2.5, 5.0)
        # Assign RBC morphologies
        self._rbc_morph = ["normal"] * self.n_rbc
        for morph, frac in self._abnormal_rbc.items():
            if morph in self.RBC_MORPHOLOGIES:
                n_abnormal = int(round(frac * self.n_rbc))
                indices = self.rng.choice(
                    self.n_rbc, min(n_abnormal, self.n_rbc), replace=False
                )
                for idx in indices:
                    if self._rbc_morph[idx] == "normal":
                        self._rbc_morph[idx] = morph
        # Sickle cells get elongated shape + random angle
        self._rbc_angle = self.rng.uniform(0, np.pi, self.n_rbc)
        # Spherocytes are smaller, denser
        for i in range(self.n_rbc):
            if self._rbc_morph[i] == "spherocyte":
                self._rbc_radius[i] *= 0.65

        # Rouleaux: stacks of RBCs resembling coins stacked on edge.
        # Must be after morphology assignment (only normal RBCs form stacks).
        self._rbc_in_rouleaux = np.zeros(self.n_rbc, dtype=bool)
        self._rouleaux_stacks = []
        if self._rouleaux_fraction > 0:
            self._form_rouleaux(margin)

        # -- WBCs --
        wbc_types = []
        remaining = self.n_wbc
        type_names = list(self._wbc_diff.keys())
        for i, name in enumerate(type_names):
            if i == len(type_names) - 1:
                count = remaining
            else:
                count = round(self._wbc_diff[name] * self.n_wbc)
                count = min(count, remaining)
            wbc_types.extend([name] * count)
            remaining -= count

        self.rng.shuffle(wbc_types)
        self._wbc_types = wbc_types[: self.n_wbc]

        self._wbc_x = self.rng.uniform(
            margin + 15, self.width - margin - 15, self.n_wbc
        )
        self._wbc_y = self.rng.uniform(
            margin + 15, self.height - margin - 15, self.n_wbc
        )
        self._wbc_radius = np.array(
            [self.WBC_TYPES[t]["size_um"] / 2.0 for t in self._wbc_types]
        )

        # -- Platelets --
        self._plt_x = self.rng.uniform(margin, self.width - margin, self.n_platelets)
        self._plt_y = self.rng.uniform(margin, self.height - margin, self.n_platelets)
        n_clusters = max(1, self.n_platelets // 4)
        cluster_cx = self.rng.uniform(margin, self.width - margin, n_clusters)
        cluster_cy = self.rng.uniform(margin, self.height - margin, n_clusters)
        for i in range(self.n_platelets):
            if self.rng.random() < 0.4:
                ci = self.rng.integers(0, n_clusters)
                self._plt_x[i] = cluster_cx[ci] + self.rng.normal(0, 3)
                self._plt_y[i] = cluster_cy[ci] + self.rng.normal(0, 3)

    def _form_rouleaux(self, margin: int):
        """Arrange a fraction of RBCs into linear rouleaux stacks.

        Rouleaux are columns of RBCs stacked like coins. They form when
        elevated plasma proteins (fibrinogen, immunoglobulins) make RBCs
        sticky. In the smear, they appear as overlapping chains.

        Each stack: 3-7 cells, linear arrangement, ~40% diameter overlap.
        """
        n_in_rouleaux = int(self.n_rbc * self._rouleaux_fraction)
        if n_in_rouleaux < 3:
            return

        # Only recruit normal-morphology RBCs (sickle/spherocyte don't stack)
        normal_idx = [i for i in range(self.n_rbc)
                      if self._rbc_morph[i] == "normal"]
        n_available = min(n_in_rouleaux, len(normal_idx))
        recruited = self.rng.choice(normal_idx, n_available, replace=False)

        # Partition into stacks of 3-7 cells
        stacks = []
        ptr = 0
        while ptr < len(recruited):
            remaining = len(recruited) - ptr
            if remaining < 3:
                break
            size = min(remaining, int(self.rng.integers(3, 8)))
            stacks.append(recruited[ptr:ptr + size].tolist())
            ptr += size

        # Arrange each stack as a linear chain
        for stack_idx in stacks:
            n = len(stack_idx)
            # Stack center: use the first cell's position as anchor
            anchor = stack_idx[0]
            cx = self._rbc_x[anchor]
            cy = self._rbc_y[anchor]

            # Random stack orientation
            angle = self.rng.uniform(0, np.pi)
            ca, sa = np.cos(angle), np.sin(angle)

            # Overlap: cells spaced at 60% of diameter apart
            mean_r = float(np.mean(self._rbc_radius[stack_idx]))
            spacing = mean_r * 2 * 0.60  # 40% overlap between adjacent

            # Curvature: scales with stack length (longer → more arc)
            curvature = self.rng.uniform(-0.08, 0.08) * (1 + 0.15 * n)

            for j, cell_idx in enumerate(stack_idx):
                # Position along the stack
                offset = (j - (n - 1) / 2.0) * spacing
                # Slight lateral wobble for naturalism
                lat = self.rng.normal(0, 0.3)
                # Curvature: quadratic bend
                frac = j / max(1, n - 1) * 2 - 1  # -1 to 1
                lat += curvature * offset * offset * 0.1

                new_x = cx + offset * ca - lat * sa
                new_y = cy + offset * sa + lat * ca
                new_x = np.clip(new_x, margin, self.width - margin)
                new_y = np.clip(new_y, margin, self.height - margin)

                self._rbc_x[cell_idx] = new_x
                self._rbc_y[cell_idx] = new_y
                self._rbc_in_rouleaux[cell_idx] = True

            self._rouleaux_stacks.append({
                "indices": stack_idx,
                "center": (cx, cy),
                "angle_deg": float(np.degrees(angle)),
                "n_cells": n,
            })

    def _render_full(self):
        """Pre-render full-resolution images for all channels."""
        self._bf_full = self._render_bf()
        self._nuc_full = self._render_nuc()
        self._mem_full = self._render_mem()

    # -- RBC rendering --

    def _make_rbc_stamp(self, radius_internal: float) -> np.ndarray:
        """Create a pre-computed RGB RBC biconcave disc stamp.

        Biconcave disc produces characteristic central pallor:
        - Edge ring: darker salmon (hemoglobin density)
        - Central pallor: pale pink (thin region)
        - Rim: slight darkening at membrane
        """
        r = int(np.ceil(radius_internal)) + 1
        size = 2 * r + 1
        stamp = np.full((size, size, 3), self._bg_color, dtype=np.float32)

        body = self._rbc_body
        pallor = self._rbc_pallor
        edge = self._rbc_edge
        bg = self._bg_color

        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= radius_internal:
                    t = dist / radius_internal
                    # Biconcave disc profile: darker ring, pale center
                    ring_factor = np.sin(np.pi * t) ** 0.7
                    # Blend body (ring) and pallor (center)
                    color = pallor * (1 - ring_factor) + body * ring_factor
                    # Outer edge: slightly darker
                    if t > 0.75:
                        edge_blend = (t - 0.75) / 0.25
                        color = color * (1 - edge_blend * 0.3) + edge * edge_blend * 0.3
                    # Fade to background at very edge
                    if t > 0.82:
                        edge_fade = (1.0 - t) / 0.18
                        color = color * edge_fade + bg * (1 - edge_fade)
                    stamp[cy + dy, cx + dx] = color

        return stamp

    def _render_sickle_rbc(self, img, cx, cy, radius, angle):
        """Render a sickle (crescent) shaped RBC in RGB."""
        r = max(2, int(round(radius)))
        s = self.internal_scale
        axes = (int(r * 2.5), max(1, int(r * 0.45)))
        ang_deg = np.degrees(angle)
        # Dark salmon crescent body
        body_color = tuple(int(c) for c in (self._rbc_body * 0.75).clip(0, 255))
        cv2.ellipse(img, (cx, cy), axes, ang_deg, 0, 360, body_color, -1)
        # Lighter interior (curved membrane overlap)
        offset_x = int(np.cos(angle + np.pi / 2) * r * 0.5)
        offset_y = int(np.sin(angle + np.pi / 2) * r * 0.5)
        lighter = tuple(int(c) for c in self._bg_color.clip(0, 255))
        cv2.ellipse(
            img, (cx + offset_x, cy + offset_y),
            (int(r * 1.8), max(1, int(r * 0.35))),
            ang_deg, 0, 360, lighter, -1,
        )
        # Pointed tips
        tip_dist = r * 2.2
        tip_r = max(1, int(r * 0.25))
        tip_color = tuple(int(c) for c in (self._rbc_body * 0.80).clip(0, 255))
        for sign in [-1, 1]:
            tx = cx + int(sign * tip_dist * np.cos(angle))
            ty = cy + int(sign * tip_dist * np.sin(angle))
            cv2.circle(img, (tx, ty), tip_r, tip_color, -1)

    def _render_target_rbc(self, img, cx, cy, radius):
        """Render a target cell (codocyte) — bull's eye in RGB."""
        r = max(2, int(round(radius)))
        outer = tuple(int(c) for c in (self._rbc_body * 0.85).clip(0, 255))
        ring_light = tuple(int(c) for c in self._rbc_pallor.clip(0, 255))
        center_dark = tuple(int(c) for c in (self._rbc_body * 0.82).clip(0, 255))
        center_light = tuple(int(c) for c in (self._rbc_pallor * 0.97).clip(0, 255))
        cv2.circle(img, (cx, cy), r, outer, -1)
        cv2.circle(img, (cx, cy), max(1, int(r * 0.7)), ring_light, -1)
        cv2.circle(img, (cx, cy), max(1, int(r * 0.4)), center_dark, -1)
        cv2.circle(img, (cx, cy), max(1, int(r * 0.15)), center_light, -1)

    def _render_spherocyte_rbc(self, img, cx, cy, radius):
        """Render a spherocyte in RGB — uniformly dark, no central pallor."""
        r = max(1, int(round(radius)))
        lt = max(1, self.internal_scale // 2)
        # Uniformly stained (no biconcave = no pallor)
        body = tuple(int(c) for c in (self._rbc_body * 0.82).clip(0, 255))
        rim = tuple(int(c) for c in (self._rbc_body * 0.75).clip(0, 255))
        cv2.circle(img, (cx, cy), r, body, -1)
        cv2.circle(img, (cx, cy), r, rim, lt)

    # -- Brightfield rendering --

    def _render_bf(self) -> np.ndarray:
        """Render Wright-Giemsa stained brightfield in RGB at internal resolution."""
        s = self.internal_scale
        iw, ih = self._iw, self._ih

        # RGB background (cream/pink)
        img = np.empty((ih, iw, 3), dtype=np.float32)
        img[:] = self._bg_color

        # Background noise (correlated across channels for natural look)
        bg_noise = self._noise_rng.normal(
            0, 1.5, (self.height, self.width)
        ).astype(np.float32)
        if s > 1:
            bg_noise = cv2.resize(
                bg_noise, (iw, ih), interpolation=cv2.INTER_LINEAR
            )
        img += bg_noise[:, :, None]

        # -- RBC stamps (cached per quantized scaled radius) --
        stamp_cache = {}
        for i in range(self.n_rbc):
            r_scaled = self._rbc_radius[i] * s
            r_key = round(r_scaled * 2) / 2
            if r_key not in stamp_cache:
                stamp_cache[r_key] = self._make_rbc_stamp(r_key)

        # -- RBCs --
        # Pre-compute transmittance reference for rouleaux stacking
        bg_f = self._bg_color.astype(np.float32)

        for i in range(self.n_rbc):
            cx = self._s(self._rbc_x[i])
            cy = self._s(self._rbc_y[i])
            r_scaled = self._sf(self._rbc_radius[i])
            morph = self._rbc_morph[i]

            if morph == "sickle":
                self._render_sickle_rbc(
                    img, cx, cy, r_scaled, self._rbc_angle[i]
                )
            elif morph == "target":
                self._render_target_rbc(img, cx, cy, r_scaled)
            elif morph == "spherocyte":
                self._render_spherocyte_rbc(img, cx, cy, r_scaled)
            else:
                r_key = round(r_scaled * 2) / 2
                stamp = stamp_cache[r_key]
                sh, sw = stamp.shape[:2]
                hr, wr = sh // 2, sw // 2
                y0, y1 = cy - hr, cy - hr + sh
                x0, x1 = cx - wr, cx - wr + sw
                sy0, sx0 = 0, 0
                if y0 < 0:
                    sy0 = -y0
                    y0 = 0
                if x0 < 0:
                    sx0 = -x0
                    x0 = 0
                if y1 > ih:
                    y1 = ih
                if x1 > iw:
                    x1 = iw
                region = stamp[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)]
                if region.size > 0:
                    target = img[y0:y1, x0:x1]
                    if self._rbc_in_rouleaux[i]:
                        # Rouleaux: transmittance multiplication for optical
                        # density accumulation. Where stamp equals bg, T=1
                        # (no change); where stamp is cell body, T<1 (darken).
                        # Overlapping cells compound: bg * T1 * T2.
                        transmittance = region / bg_f[None, None, :]
                        np.clip(transmittance, 0.01, 1.0, out=transmittance)
                        img[y0:y1, x0:x1] = target * transmittance
                    else:
                        # Normal: per-channel minimum (darker wins)
                        img[y0:y1, x0:x1] = np.minimum(target, region)

        # -- WBCs --
        for i in range(self.n_wbc):
            self._render_wbc_bf(img, i)

        # -- Platelets --
        plt_r = max(1, s // 2)
        plt_color = tuple(int(c) for c in self._plt_color)
        for i in range(self.n_platelets):
            cx = self._s(self._plt_x[i])
            cy = self._s(self._plt_y[i])
            if 0 <= cx < iw and 0 <= cy < ih:
                cv2.circle(img, (cx, cy), plt_r, plt_color, -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_wbc_bf(self, img: np.ndarray, idx: int):
        """Render a single WBC on the RGB brightfield image at internal resolution."""
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        lt = max(1, s // 2)

        wtype = self._wbc_types[idx]
        params = self.WBC_TYPES[wtype]
        cx = self._s(self._wbc_x[idx])
        cy = self._s(self._wbc_y[idx])
        r = self._sf(self._wbc_radius[idx])

        nuc_rgb = params["nucleus_rgb"]
        cyto_rgb = params["cyto_rgb"]
        gran_rgb = params["granule_rgb"]

        # Clear area (WBCs displace RBCs)
        clear_color = tuple(int(c) for c in (self._bg_color * 0.97))
        cv2.circle(img, (cx, cy), int(r) + s, clear_color, -1)
        # Cytoplasm
        cv2.circle(img, (cx, cy), int(r), cyto_rgb, -1)
        # Cell membrane ring (slightly darker)
        mem_rgb = tuple(max(0, c - 15) for c in cyto_rgb)
        cv2.circle(img, (cx, cy), int(r), mem_rgb, lt)

        rng_local = np.random.default_rng(idx * 1000 + 42)

        if wtype == "neutrophil":
            n_lobes = rng_local.integers(2, 5)
            # Wider separation (0.50r) + smaller lobes (0.26r) ensures no overlap
            # even for 4-lobe cells: min gap = 2×0.50r×sin(45°) - 2×0.26r = 0.188r > 0
            lobe_r = r * 0.26
            dist_from_center = r * 0.50
            lobe_positions = []
            for j in range(n_lobes):
                angle = (2 * np.pi * j / n_lobes) + rng_local.normal(0, 0.12)
                lx = cx + int(dist_from_center * np.cos(angle))
                ly = cy + int(dist_from_center * np.sin(angle))
                lobe_positions.append((lx, ly))
                cv2.circle(img, (lx, ly), max(2, int(lobe_r)), nuc_rgb, -1)
                # Lobe rim (slightly lighter) to emphasise each lobe boundary
                lobe_rim = tuple(min(255, c + 18) for c in nuc_rgb)
                cv2.circle(img, (lx, ly), max(2, int(lobe_r)), lobe_rim, lt)

            # Thin chromatin strands connecting adjacent lobes
            strand_rgb = tuple(min(255, c + 45) for c in nuc_rgb)
            for j in range(len(lobe_positions)):
                p1 = lobe_positions[j]
                p2 = lobe_positions[(j + 1) % len(lobe_positions)]
                # Cytoplasm-coloured pinch at the inter-lobe waist makes gap visible
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                cv2.circle(img, (mx, my), max(1, lt), cyto_rgb, -1)
                cv2.line(img, p1, p2, strand_rgb, max(1, lt // 2))

            # Granules (fine pink/lilac)
            if gran_rgb:
                n_granules = rng_local.integers(8, 20)
                gran_r = max(1, s // 2)
                for _ in range(n_granules):
                    gx = cx + int(rng_local.normal(0, r * 0.4))
                    gy = cy + int(rng_local.normal(0, r * 0.4))
                    if 0 <= gx < iw and 0 <= gy < ih:
                        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                        if d < r * 0.85:
                            cv2.circle(img, (gx, gy), gran_r, gran_rgb, -1)

        elif wtype == "lymphocyte":
            nuc_r = r * 0.78
            cv2.circle(img, (cx, cy), max(2, int(nuc_r)), nuc_rgb, -1)
            # Dense chromatin clump
            darker_nuc = tuple(max(0, c - 10) for c in nuc_rgb)
            cv2.circle(img, (cx, cy), max(1, int(nuc_r * 0.4)),
                       darker_nuc, -1)

        elif wtype == "monocyte":
            nuc_r = r * 0.50
            offset = int(nuc_r * 0.55)
            cv2.circle(img, (cx - offset, cy), max(2, int(nuc_r)),
                       nuc_rgb, -1)
            cv2.circle(img, (cx + offset, cy), max(2, int(nuc_r)),
                       nuc_rgb, -1)
            # Lighter chromatin bridge
            lighter_nuc = tuple(min(255, c + 30) for c in nuc_rgb)
            cv2.circle(img, (cx, cy - int(nuc_r * 0.3)),
                       max(1, int(nuc_r * 0.45)), lighter_nuc, -1)
            # Cytoplasmic vacuoles
            vac_r = max(1, s // 2)
            vac_rgb = tuple(min(255, c + 10) for c in cyto_rgb)
            for _ in range(3):
                vx = cx + int(rng_local.normal(0, r * 0.3))
                vy = cy + int(rng_local.normal(0, r * 0.3))
                cv2.circle(img, (vx, vy), vac_r, vac_rgb, -1)

        elif wtype == "eosinophil":
            lobe_r = r * 0.35
            offset = int(r * 0.38)
            cv2.circle(img, (cx - offset, cy), max(2, int(lobe_r)),
                       nuc_rgb, -1)
            cv2.circle(img, (cx + offset, cy), max(2, int(lobe_r)),
                       nuc_rgb, -1)
            # Chromatin bridge
            bridge_rgb = tuple(min(255, c + 10) for c in nuc_rgb)
            cv2.line(img, (cx - offset, cy), (cx + offset, cy),
                     bridge_rgb, lt)

            # Eosinophil granules: LARGE, coarse, densely packed orange-red
            if gran_rgb:
                n_granules = rng_local.integers(50, 80)
                gran_r = max(2, int(s * 1.3))
                for _ in range(n_granules):
                    gx = cx + int(rng_local.normal(0, r * 0.45))
                    gy = cy + int(rng_local.normal(0, r * 0.45))
                    if 0 <= gx < iw and 0 <= gy < ih:
                        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                        if d < r * 0.90:
                            d_lobe1 = np.sqrt(
                                (gx - (cx - offset)) ** 2 + (gy - cy) ** 2)
                            d_lobe2 = np.sqrt(
                                (gx - (cx + offset)) ** 2 + (gy - cy) ** 2)
                            if d_lobe1 > lobe_r * 0.8 and d_lobe2 > lobe_r * 0.8:
                                cv2.circle(img, (gx, gy), gran_r,
                                           gran_rgb, -1)

        elif wtype == "basophil":
            # Basophil: bilobed nucleus largely obscured by coarse dark granules
            lobe_r = r * 0.30
            offset = int(r * 0.30)
            cv2.circle(img, (cx - offset, cy), max(2, int(lobe_r)),
                       nuc_rgb, -1)
            cv2.circle(img, (cx + offset, cy), max(2, int(lobe_r)),
                       nuc_rgb, -1)
            # Very dark, coarse, large granules filling most of cytoplasm
            if gran_rgb:
                n_granules = rng_local.integers(15, 30)
                gran_r = max(2, int(s * 1.5))
                for _ in range(n_granules):
                    gx = cx + int(rng_local.normal(0, r * 0.40))
                    gy = cy + int(rng_local.normal(0, r * 0.40))
                    if 0 <= gx < iw and 0 <= gy < ih:
                        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
                        if d < r * 0.88:
                            cv2.circle(img, (gx, gy), gran_r, gran_rgb, -1)

    # -- Nucleus channel --

    def _render_nuc(self) -> np.ndarray:
        """Render WBC nuclei only (fluorescence channel) at internal resolution.

        Nuclear morphology is the key differentiator:
        - Neutrophil: 2-4 separate lobes
        - Lymphocyte: single large round nucleus
        - Monocyte: kidney/bean-shaped
        - Eosinophil: bilobed
        """
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        img = np.zeros((ih, iw), dtype=np.float32)

        for i in range(self.n_wbc):
            wtype = self._wbc_types[i]
            params = self.WBC_TYPES[wtype]
            cx = self._s(self._wbc_x[i])
            cy = self._s(self._wbc_y[i])
            r = self._sf(self._wbc_radius[i])
            # Darker nucleus RGB → brighter in fluorescence
            nuc_bright = 255 - int(np.mean(params["nucleus_rgb"]))

            rng_local = np.random.default_rng(i * 1000 + 42)

            if wtype == "neutrophil":
                n_lobes = rng_local.integers(2, 5)
                lobe_r = r * 0.30
                dist_from_center = r * 0.48
                for j in range(n_lobes):
                    angle = (2 * np.pi * j / n_lobes) + rng_local.normal(0, 0.15)
                    lx = cx + int(dist_from_center * np.cos(angle))
                    ly = cy + int(dist_from_center * np.sin(angle))
                    cv2.circle(img, (lx, ly), max(2, int(lobe_r)), nuc_bright, -1)

            elif wtype == "lymphocyte":
                nuc_r = r * 0.75
                cv2.circle(img, (cx, cy), max(2, int(nuc_r)), nuc_bright, -1)

            elif wtype == "monocyte":
                nuc_r = r * 0.50
                offset = int(nuc_r * 0.55)
                cv2.circle(
                    img, (cx - offset, cy), max(2, int(nuc_r)), nuc_bright, -1
                )
                cv2.circle(
                    img, (cx + offset, cy), max(2, int(nuc_r)), nuc_bright, -1
                )
                cv2.circle(
                    img,
                    (cx, cy - int(nuc_r * 0.3)),
                    max(1, int(nuc_r * 0.35)),
                    nuc_bright * 0.4,
                    -1,
                )

            elif wtype == "eosinophil":
                lobe_r = r * 0.32
                offset = int(r * 0.48)
                cv2.circle(
                    img, (cx - offset, cy), max(2, int(lobe_r)), nuc_bright, -1
                )
                cv2.circle(
                    img, (cx + offset, cy), max(2, int(lobe_r)), nuc_bright, -1
                )

            elif wtype == "basophil":
                # Bilobed nucleus, largely obscured by granules in BF
                lobe_r = r * 0.28
                offset = int(r * 0.30)
                cv2.circle(
                    img, (cx - offset, cy), max(2, int(lobe_r)), nuc_bright, -1
                )
                cv2.circle(
                    img, (cx + offset, cy), max(2, int(lobe_r)), nuc_bright, -1
                )

        return np.clip(img, 0, 255).astype(np.uint8)

    # -- Membrane (RBC morphology) channel --

    def _render_mem(self) -> np.ndarray:
        """Render RBC morphology channel at internal resolution.

        Different morphologies render with distinct patterns:
        - Normal: bright ring + dim center (donut = biconcave disc)
        - Sickle: elongated bright outline (crescent)
        - Target: concentric rings (bull's eye)
        - Spherocyte: uniform bright circle (no central pallor)
        """
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        lt = max(1, s // 2)  # scaled line thickness
        img = np.zeros((ih, iw), dtype=np.float32)

        for i in range(self.n_rbc):
            cx = self._s(self._rbc_x[i])
            cy = self._s(self._rbc_y[i])
            r = self._sf(self._rbc_radius[i])
            morph = self._rbc_morph[i]

            if morph == "sickle":
                angle = self._rbc_angle[i]
                axes = (int(r * 2.5), max(1, int(r * 0.45)))
                ang_deg = np.degrees(angle)
                cv2.ellipse(img, (cx, cy), axes, ang_deg, 0, 360, 200, lt)
            elif morph == "target":
                cv2.circle(img, (cx, cy), max(1, int(r)), 180, lt)
                cv2.circle(img, (cx, cy), max(1, int(r * 0.6)), 120, lt)
                cv2.circle(img, (cx, cy), max(1, int(r * 0.3)), 180, -1)
            elif morph == "spherocyte":
                cv2.circle(img, (cx, cy), max(1, int(r)), 160, -1)
            else:
                # Normal: donut (biconcave disc)
                cv2.circle(img, (cx, cy), max(1, int(r)), 180, lt)
                cv2.circle(img, (cx, cy), max(1, int(r * 0.5)), 60, -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    # -- SimulationBridge interface --

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs):
        """Capture a frame — compatible with SimulationBridge.

        Returns (H, W, 3) uint8 for BF (RGB Giemsa stain).
        Nucleus and membrane channels return grayscale expanded to 3ch.
        """
        self._update_mode()
        self._update_objectif()
        self._snap_count += 1

        if self.mode == 0:
            full = self._bf_full
        elif self.mode == 1:
            full = self._nuc_full
        elif self.mode == 2:
            full = self._mem_full
        else:
            if self.mode in self._extra_channels:
                full = self._extra_channels[self.mode].get("image", self._bf_full)
            else:
                full = self._bf_full

        crop = self._crop_fov(full)
        crop = self._blood_pipeline.apply(crop)

        # Ensure 3-channel output for RGB mode
        if self.rgb_mode and crop.ndim == 2:
            crop = np.stack([crop, crop, crop], axis=-1)

        return crop

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._generate_cells()
            self._render_full()

    # -- Ground truth --

    def get_ground_truth(self) -> dict:
        """Return ground truth for grading."""
        differential = {}
        for wtype in self._wbc_types:
            differential[wtype] = differential.get(wtype, 0) + 1

        wbc_positions = [
            {
                "x": float(self._wbc_x[i]),
                "y": float(self._wbc_y[i]),
                "type": self._wbc_types[i],
            }
            for i in range(self.n_wbc)
        ]

        rbc_morphology = {}
        for morph in self._rbc_morph:
            rbc_morphology[morph] = rbc_morphology.get(morph, 0) + 1

        gt = {
            "n_rbc": self.n_rbc,
            "n_wbc": self.n_wbc,
            "n_platelets": self.n_platelets,
            "wbc_differential": differential,
            "wbc_positions": wbc_positions,
            "wbc_types": list(set(self._wbc_types)),
            "rbc_morphology": rbc_morphology,
        }
        if self._rouleaux_stacks:
            gt["n_rouleaux_stacks"] = len(self._rouleaux_stacks)
            gt["n_rbc_in_rouleaux"] = int(np.sum(self._rbc_in_rouleaux))
            gt["rouleaux_stacks"] = self._rouleaux_stacks
        return gt

    # -- Convenience --

    def add_channel(
        self,
        channel_id: int,
        name: str,
        filter_label: str,
        led_label: str,
        image: np.ndarray = None,
    ):
        """Register an extra imaging channel."""
        self._extra_channels[channel_id] = {
            "name": name,
            "filter": filter_label,
            "led": led_label,
            "image": image,
        }
