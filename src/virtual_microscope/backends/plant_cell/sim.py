"""
PlantCellSim — Onion epidermis (iodine-stained) for virtual microscopy.

Simulates an onion epidermal peel:
  - Elongated hexagonal cells in a staggered pattern (150-250µm × 30-50µm)
  - Thick cell walls (~7µm) with middle lamella double-line effect
  - Large central vacuole (80-90% of cell volume)
  - Peripheral nucleus (pushed to cell wall by vacuole)
  - Iodine staining: amber walls, yellow-brown nuclei

Channels:
  - mode 0: Brightfield (iodine-stained)
  - mode 1: "nucleus-channel" — DAPI nuclear stain (bright nuclei)
  - mode 2: "membrane-channel" — Calcofluor White cell wall stain

This is the classic introductory microscopy sample. The agent must learn
to work with rectangular cells, measure cell dimensions, and locate
peripheral nuclei — a fundamentally different morphology from Voronoi tissue.

Usage via SimulationBridge:
    sim = PlantCellSim(seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.optical_pipeline import OpticalPipeline


class PlantCellSim:
    """Onion epidermis simulation — elongated hexagonal plant cells with thick walls."""

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        cell_length_range: tuple = (150, 250),
        cell_width_range: tuple = (30, 50),
        wall_thickness: float = 7.0,
        staining: str = "iodine",      # "iodine", "unstained", "toluidine"
        seed: int = 42,
        internal_scale: int = 4,
    ):
        self.width = world_size
        self.height = world_size
        self.internal_scale = internal_scale
        self._iw = world_size * internal_scale
        self._ih = world_size * internal_scale
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # Camera / state
        self.camera_offset = np.array([0.0, 0.0])
        self.focal_plane = 0.0
        self.tissue_z = 0.0
        self.state_devices = {}
        self.mode = 0
        self.current_objectiv = 10
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40, "100x": 100}
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.5}
        self._dof = 6.0
        self._extra_channels = {}
        self._snap_count = 0

        self.rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 5555)
        self.fixed_dt = 0.0
        self._time = 0.0
        self.auto_step = False
        self.snaps_per_step = 2

        # Z-drift
        self.z_drift_rate = 0.0   # µm/s
        self.z_drift_noise = 0.0  # σ of z-jitter (µm·s⁻½, Brownian)

        # Cell parameters
        self.cell_length_range = cell_length_range
        self.cell_width_range = cell_width_range
        self.wall_thickness = wall_thickness
        self.staining = staining

        # Plasmolysis state (0.0 = turgid, 1.0 = fully plasmolyzed)
        self._plasmolysis_level = 0.0
        self._plasmolysis_target = 0.0
        self._plasmolysis_rate = 0.05  # per step

        # Optical pipeline
        self._pipeline = OpticalPipeline()
        self._pipeline.noise = {"photon_scale": 600, "read_noise": 1.2}
        self._pipeline.vignette = 0.06

        # Generate cells and render
        self._generate_cells()
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None
        self._render_full()

    def _s(self, v):
        """Scale world coordinate to internal resolution (int)."""
        return int(round(v * self.internal_scale))

    def _sf(self, v):
        """Scale world coordinate to internal resolution (float)."""
        return v * self.internal_scale

    def _wavy_rect_contour(self, half_l, half_w, cell_idx, level=0,
                           amplitude_world=1.5):
        """Generate wavy elongated-hexagonal polygon in local coords.

        Real onion epidermis cells have tapered/pointed short ends
        (elongated hexagonal shape), not rectangular.  The taper ratio
        is deterministic per cell_idx so outer/inner/vacuole contours
        all share the same hex shape at different insets.

        Returns np.float32 array of shape (N, 2).
        """
        # Taper ratio — deterministic per cell, consistent across levels
        taper_rng = np.random.default_rng(cell_idx * 1237 + 99991)
        taper_ratio = float(taper_rng.uniform(0.4, 0.65))
        taper = half_w * taper_ratio
        taper = min(taper, half_l * 0.4)  # safety cap

        rng = np.random.default_rng(cell_idx * 7919 + level * 3571 + 31337)
        amp = self._sf(amplitude_world)

        straight_len = 2 * (half_l - taper)
        spacing = self._sf(25)
        n_long = max(6, min(14, int(straight_len / spacing) + 1))
        diag_len = np.sqrt(taper ** 2 + half_w ** 2)
        n_diag = max(3, min(6, int(diag_len / spacing) + 1))

        def _smooth(n):
            n_ctrl = max(3, n // 2)
            ctrl = rng.normal(0, amp * 0.6, n_ctrl)
            ctrl[0] = ctrl[-1] = 0.0
            return np.interp(np.linspace(0, 1, n),
                             np.linspace(0, 1, n_ctrl), ctrl)

        pts = []

        # Top straight edge: TL(-hl+t, -hw) → TR(hl-t, -hw)
        dy = _smooth(n_long)
        for i in range(n_long):
            t = i / max(1, n_long - 1)
            pts.append([(-half_l + taper) + t * straight_len,
                        -half_w + dy[i]])

        # Upper-right diagonal: TR → R(hl, 0)
        d = _smooth(n_diag)
        x0, y0 = half_l - taper, -half_w
        x1, y1 = half_l, 0.0
        for i in range(1, n_diag):
            t = i / max(1, n_diag - 1)
            pts.append([x0 + t * (x1 - x0) + d[i] * 0.4,
                        y0 + t * (y1 - y0) + d[i] * 0.4])

        # Lower-right diagonal: R(hl, 0) → BR(hl-t, hw)
        d = _smooth(n_diag)
        x0, y0 = half_l, 0.0
        x1, y1 = half_l - taper, half_w
        for i in range(1, n_diag):
            t = i / max(1, n_diag - 1)
            pts.append([x0 + t * (x1 - x0) + d[i] * 0.4,
                        y0 + t * (y1 - y0) + d[i] * 0.4])

        # Bottom straight edge: BR(hl-t, hw) → BL(-hl+t, hw)
        dy = _smooth(n_long)
        for i in range(1, n_long):
            t = i / max(1, n_long - 1)
            pts.append([(half_l - taper) - t * straight_len,
                        half_w + dy[i]])

        # Lower-left diagonal: BL(-hl+t, hw) → L(-hl, 0)
        d = _smooth(n_diag)
        x0, y0 = -half_l + taper, half_w
        x1, y1 = -half_l, 0.0
        for i in range(1, n_diag):
            t = i / max(1, n_diag - 1)
            pts.append([x0 + t * (x1 - x0) - d[i] * 0.4,
                        y0 + t * (y1 - y0) + d[i] * 0.4])

        # Upper-left diagonal: L(-hl, 0) → TL(-hl+t, -hw)
        d = _smooth(n_diag)
        x0, y0 = -half_l, 0.0
        x1, y1 = -half_l + taper, -half_w
        for i in range(1, n_diag):
            t = i / max(1, n_diag - 1)
            pts.append([x0 + t * (x1 - x0) - d[i] * 0.4,
                        y0 + t * (y1 - y0) - d[i] * 0.4])

        return np.array(pts, dtype=np.float32)

    def _transform_contour(self, local_pts, cx, cy, angle):
        """Rotate and translate a local contour to global coordinates."""
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        return (local_pts @ R.T + np.array([cx, cy])).astype(np.int32)

    def _generate_cells(self):
        """Generate rectangular cells in a staggered brick pattern."""
        rng = self.rng
        margin = 10
        cells = []

        # Global tissue angle: slight rotation (±8°) for realism
        tissue_angle = rng.uniform(-8, 8)
        self._tissue_angle = tissue_angle
        cos_ta = np.cos(np.radians(tissue_angle))
        sin_ta = np.sin(np.radians(tissue_angle))

        # Generate rows of cells (staggered brick pattern)
        y_pos = -margin
        row_idx = 0
        while y_pos < self.height + margin + 300:
            # Each row has a consistent cell width (cells in a row are similar)
            row_width = rng.uniform(*self.cell_width_range)

            # Stagger offset for alternating rows
            x_offset = rng.uniform(0, 80) if row_idx % 2 else 0

            x_pos = -margin - x_offset
            while x_pos < self.width + margin + 300:
                cell_len = rng.uniform(*self.cell_length_range)
                # Per-cell angle jitter (±3°)
                cell_angle = tissue_angle + rng.uniform(-3, 3)

                # Cell center in unrotated coords
                cx_raw = x_pos + cell_len / 2
                cy_raw = y_pos + row_width / 2

                # Rotate around world center
                wc = self.width / 2
                hc = self.height / 2
                dx, dy = cx_raw - wc, cy_raw - hc
                cx = wc + dx * cos_ta - dy * sin_ta
                cy = hc + dx * sin_ta + dy * cos_ta

                # Nucleus: peripheral position (pushed to wall by vacuole)
                # Real onion cells: nucleus pressed against a long wall (~75%)
                nuc_side = rng.choice(["top", "bottom", "top", "bottom",
                                       "top", "bottom", "left", "right"])
                nuc_r = rng.uniform(4, 6)  # nucleus radius
                if nuc_side == "top":
                    nuc_dx = rng.uniform(-cell_len * 0.3, cell_len * 0.3)
                    nuc_dy = -(row_width / 2 - self.wall_thickness - nuc_r - 1)
                elif nuc_side == "bottom":
                    nuc_dx = rng.uniform(-cell_len * 0.3, cell_len * 0.3)
                    nuc_dy = row_width / 2 - self.wall_thickness - nuc_r - 1
                elif nuc_side == "left":
                    nuc_dx = -(cell_len / 2 - self.wall_thickness - nuc_r - 1)
                    nuc_dy = rng.uniform(-row_width * 0.2, row_width * 0.2)
                else:
                    nuc_dx = cell_len / 2 - self.wall_thickness - nuc_r - 1
                    nuc_dy = rng.uniform(-row_width * 0.2, row_width * 0.2)

                # Rotate nucleus offset by cell angle
                ca = np.radians(cell_angle)
                cos_ca, sin_ca = np.cos(ca), np.sin(ca)
                nuc_rx = nuc_dx * cos_ca - nuc_dy * sin_ca
                nuc_ry = nuc_dx * sin_ca + nuc_dy * cos_ca

                # Cytoplasmic strands (0-2 per cell, sparse)
                n_strands = int(rng.choice([0, 0, 0, 1, 1, 2]))
                cell_data = {
                    "cx": cx, "cy": cy,
                    "length": cell_len,
                    "width": row_width,
                    "angle": cell_angle,
                    "nuc_x": cx + nuc_rx,
                    "nuc_y": cy + nuc_ry,
                    "nuc_r": nuc_r,
                    "nuc_aspect": rng.uniform(1.2, 1.8),  # lens-shaped
                    "vacuole_clarity": rng.uniform(0.85, 0.95),
                    "n_strands": n_strands,
                }
                for s in range(n_strands):
                    cell_data[f"strand_angle_{s}"] = float(rng.uniform(0, 2 * np.pi))
                cells.append(cell_data)

                x_pos += cell_len + self.wall_thickness
            y_pos += row_width + self.wall_thickness
            row_idx += 1

        self._cells = cells
        self.n_cells = len(cells)

    def _render_full(self):
        """Pre-render all channels at full resolution."""
        self._bf_full = self._render_bf()
        self._nuc_full = self._render_nuc()
        self._mem_full = self._render_mem()

    def _render_bf(self) -> np.ndarray:
        """Render brightfield image at internal resolution."""
        # Compressed palette — subtle contrast like real tissue
        if self.staining == "iodine":
            wall_color = 140
            vacuole_color = 195
            cyto_color = 175
            nuc_color = 130
            tissue_bg = 155
        elif self.staining == "toluidine":
            wall_color = 135
            vacuole_color = 195
            cyto_color = 180
            nuc_color = 110
            tissue_bg = 155
        else:  # unstained — nearly transparent, very low contrast
            wall_color = 195
            vacuole_color = 215
            cyto_color = 210
            nuc_color = 205
            tissue_bg = 210

        self._tissue_bg = tissue_bg

        # Start with tissue base (middle lamella) — no glass-slide gaps
        img = np.full((self._ih, self._iw), float(tissue_bg), dtype=np.float32)

        # Subtle background texture
        bg_noise = self._noise_rng.normal(0, 1.5, (self.height, self.width)).astype(np.float32)
        if self.internal_scale > 1:
            bg_noise = cv2.resize(bg_noise, (self._iw, self._ih),
                                  interpolation=cv2.INTER_LINEAR)
        img += bg_noise

        # Draw each cell with per-cell intensity jitter
        cell_rng = np.random.default_rng(12345)
        for ci, cell in enumerate(self._cells):
            jitter = int(cell_rng.integers(-3, 4))
            self._render_cell_bf(img, cell, ci,
                                 wall_color + jitter,
                                 vacuole_color + jitter,
                                 cyto_color + jitter,
                                 nuc_color + jitter)

        # Junction thickening at wall meeting points
        self._add_junctions(img, wall_color)

        # Staining gradient artifact — darker near one edge (common real artifact)
        if self.staining != "unstained":
            grad_rng = np.random.default_rng(77771)
            grad_angle = grad_rng.uniform(0, 2 * np.pi)
            grad_strength = grad_rng.uniform(4.0, 10.0)
            # Linear gradient across the image
            yy, xx = np.mgrid[:self._ih, :self._iw]
            cx_mid, cy_mid = self._iw / 2, self._ih / 2
            grad = ((xx - cx_mid) * np.cos(grad_angle) +
                    (yy - cy_mid) * np.sin(grad_angle))
            grad = grad / max(self._iw, self._ih)  # normalize to [-0.5, 0.5]
            img -= (grad * grad_strength).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_cell_bf(self, img, cell, cell_idx, wall_color, vacuole_color,
                         cyto_color, nuc_color):
        """Render a single cell on the BF image with wavy walls.

        During plasmolysis, the protoplast shrinks inside the rigid cell wall,
        creating a visible gap filled with external solution (lighter than cyto).
        """
        s = self.internal_scale
        cx, cy = self._sf(cell["cx"]), self._sf(cell["cy"])
        half_l = self._sf(cell["length"] / 2)
        half_w = self._sf(cell["width"] / 2)
        angle = cell["angle"]
        wt = self._sf(self.wall_thickness)
        plasm = self._plasmolysis_level

        # Wavy outer wall contour
        outer_local = self._wavy_rect_contour(half_l, half_w, cell_idx, level=0)
        outer_pts = self._transform_contour(outer_local, cx, cy, angle)
        cv2.fillPoly(img, [outer_pts], wall_color)

        # Inner cell area — wall_thickness inset
        inner_half_l = half_l - wt
        inner_half_w = half_w - wt
        if inner_half_l <= 0 or inner_half_w <= 0:
            return

        inner_local = self._wavy_rect_contour(inner_half_l, inner_half_w,
                                               cell_idx, level=1)
        inner_pts = self._transform_contour(inner_local, cx, cy, angle)

        # Middle lamella — thin lighter line through wall center
        # creates the characteristic double-line effect at higher mag
        mid_half_l = (half_l + inner_half_l) / 2
        mid_half_w = (half_w + inner_half_w) / 2
        mid_local = self._wavy_rect_contour(mid_half_l, mid_half_w,
                                             cell_idx, level=3,
                                             amplitude_world=0.8)
        mid_pts = self._transform_contour(mid_local, cx, cy, angle)
        cv2.polylines(img, [mid_pts], True, float(wall_color + 12),
                      max(1, s // 2), cv2.LINE_AA)

        # Wall fibril lines (drawn before filling inner cell so they
        # appear only in the wall region)
        self._add_wall_fibrils(img, cell_idx, cx, cy,
                               half_l, half_w, inner_half_l, inner_half_w,
                               angle, wall_color)

        if plasm > 0.01:
            # Plasmolysis: fill inner cell with external solution color
            cv2.fillPoly(img, [inner_pts], 225)

            # Shrunken protoplast (elliptical)
            shrink = 1.0 - plasm * 0.4
            proto_half_l = inner_half_l * shrink
            proto_half_w = inner_half_w * shrink

            proto_center = (int(cx), int(cy))
            proto_axes = (max(2, int(proto_half_l)), max(2, int(proto_half_w)))
            cv2.ellipse(img, proto_center, proto_axes,
                         angle, 0, 360, cyto_color, -1, cv2.LINE_AA)
            cyto_thick_int = max(2, s)
            vac_axes = (max(1, int(proto_half_l - cyto_thick_int)),
                        max(1, int(proto_half_w - cyto_thick_int)))
            cv2.ellipse(img, proto_center, vac_axes,
                         angle, 0, 360, vacuole_color, -1, cv2.LINE_AA)
        else:
            # Normal turgid state
            cv2.fillPoly(img, [inner_pts], cyto_color)

            # Vacuole (inset further — thin cytoplasmic layer)
            cyto_thickness = self._sf(2.0)
            vac_half_l = inner_half_l - cyto_thickness
            vac_half_w = inner_half_w - cyto_thickness
            if vac_half_l > 0 and vac_half_w > 0:
                vac_local = self._wavy_rect_contour(vac_half_l, vac_half_w,
                                                     cell_idx, level=2)
                vac_pts = self._transform_contour(vac_local, cx, cy, angle)
                cv2.fillPoly(img, [vac_pts], vacuole_color)

                # Vacuole texture: low-frequency noise + inclusions
                self._add_vacuole_texture(img, vac_pts, cell_idx,
                                          vacuole_color)

            # Cytoplasmic strands crossing vacuole
            n_strands = cell.get("n_strands", 0)
            if n_strands > 0 and vac_half_l > 5 * s:
                nuc_ix = self._s(cell["nuc_x"])
                nuc_iy = self._s(cell["nuc_y"])
                for si in range(n_strands):
                    sa = cell.get(f"strand_angle_{si}", 0)
                    length = max(vac_half_l, vac_half_w) * 0.9
                    p0 = (nuc_ix, nuc_iy)
                    p1 = (int(nuc_ix + length * np.cos(sa)),
                           int(nuc_iy + length * np.sin(sa)))
                    cv2.line(img, p0, p1, cyto_color, max(1, s // 2),
                             cv2.LINE_AA)

        # Nucleus (lens-shaped ellipse at peripheral position)
        nuc_x_raw, nuc_y_raw = cell["nuc_x"], cell["nuc_y"]
        if plasm > 0.01:
            nuc_x_raw = cell["cx"] + (nuc_x_raw - cell["cx"]) * (1.0 - plasm * 0.3)
            nuc_y_raw = cell["cy"] + (nuc_y_raw - cell["cy"]) * (1.0 - plasm * 0.3)
        nuc_x = self._s(nuc_x_raw)
        nuc_y = self._s(nuc_y_raw)
        nuc_r = self._s(cell["nuc_r"])
        nuc_r2 = max(2, int(nuc_r / cell["nuc_aspect"]))
        cv2.ellipse(img, (nuc_x, nuc_y), (nuc_r, nuc_r2),
                     angle, 0, 360, nuc_color, -1, cv2.LINE_AA)
        cv2.circle(img, (nuc_x, nuc_y), max(1, nuc_r // 3),
                   max(0, nuc_color - 20), -1)

    def _add_vacuole_texture(self, img, vac_pts, cell_idx, vacuole_color):
        """Add low-frequency noise and small tonoplast inclusions to vacuole."""
        bx0, by0 = vac_pts.min(axis=0)
        bx1, by1 = vac_pts.max(axis=0)
        bw, bh = int(bx1 - bx0), int(by1 - by0)
        if bw < 8 or bh < 8:
            return
        # Unstained vacuoles are completely clear — skip texture
        if self.staining == "unstained":
            return

        bx0i, by0i = int(bx0), int(by0)
        rng = np.random.default_rng(cell_idx * 4217 + 9973)

        # Low-frequency noise (small image upscaled with bicubic)
        nw = max(2, bw // 20)
        nh = max(2, bh // 20)
        noise_small = rng.normal(0, 3.0, (nh, nw)).astype(np.float32)
        noise_up = cv2.resize(noise_small, (bw, bh),
                              interpolation=cv2.INTER_CUBIC)

        # Mask to vacuole polygon
        mask = np.zeros((bh, bw), dtype=np.uint8)
        shifted = (vac_pts - np.array([bx0i, by0i])).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 255)

        # Clip to image bounds
        iy0 = max(0, by0i)
        ix0 = max(0, bx0i)
        iy1 = min(img.shape[0], by0i + bh)
        ix1 = min(img.shape[1], bx0i + bw)
        if iy1 <= iy0 or ix1 <= ix0:
            return
        ny0, nx0 = iy0 - by0i, ix0 - bx0i
        ny1 = ny0 + (iy1 - iy0)
        nx1 = nx0 + (ix1 - ix0)

        roi = img[iy0:iy1, ix0:ix1]
        m = mask[ny0:ny1, nx0:nx1] > 0
        roi[m] += noise_up[ny0:ny1, nx0:nx1][m]

        # 1-3 sparse inclusions (tonoplast vesicles)
        n_incl = rng.integers(1, 4)
        for _ in range(n_incl):
            ix = int(rng.integers(bx0i + bw // 4, max(bx0i + bw // 4 + 1,
                                                       bx0i + 3 * bw // 4)))
            iy = int(rng.integers(by0i + bh // 4, max(by0i + bh // 4 + 1,
                                                       by0i + 3 * bh // 4)))
            r = int(rng.integers(2, 5))
            if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
                cv2.circle(img, (ix, iy), r,
                           float(vacuole_color - 12), -1, cv2.LINE_AA)

    def _add_wall_fibrils(self, img, cell_idx, cx, cy,
                          half_l, half_w, inner_half_l, inner_half_w,
                          angle, wall_color):
        """Draw short diagonal fibril segments in wall (crossed-polylamellate).

        Real onion cell walls have cellulose microfibrils at ~±45° to
        the cell long axis, creating a herringbone/cross-hatch pattern.
        """
        rng = np.random.default_rng(cell_idx * 6131 + 8887)
        n_fibrils = int(rng.integers(6, 14))
        wall_band = half_w - inner_half_w  # wall thickness in internal px

        for _ in range(n_fibrils):
            # Random position within wall region
            side = rng.choice([-1, 1])
            wall_mid = (half_w + inner_half_w) / 2 * side
            wall_range = wall_band * 0.35
            offset_y = wall_mid + rng.uniform(-wall_range, wall_range)
            offset_x = rng.uniform(-half_l * 0.85, half_l * 0.85)

            # Short diagonal segment at ±45° (crossed-polylamellate)
            fibril_angle = rng.choice([-1, 1]) * np.radians(rng.uniform(35, 55))
            seg_len = wall_band * rng.uniform(0.6, 1.2)
            dx = seg_len * np.cos(fibril_angle)
            dy = seg_len * np.sin(fibril_angle)

            endpoints = np.array([
                [offset_x - dx / 2, offset_y - dy / 2],
                [offset_x + dx / 2, offset_y + dy / 2],
            ], dtype=np.float32)
            pts = self._transform_contour(endpoints, cx, cy, angle)

            intensity = float(wall_color + int(rng.integers(-8, 9)))
            cv2.line(img, tuple(pts[0]), tuple(pts[1]),
                     intensity, 1, cv2.LINE_AA)

    def _add_junctions(self, img, wall_color):
        """Add dark spots at cell corners where 3+ walls converge."""
        r = max(2, int(self._sf(self.wall_thickness) * 0.6))
        junc_color = float(wall_color - 15)

        for ci, cell in enumerate(self._cells):
            cx, cy = self._sf(cell["cx"]), self._sf(cell["cy"])
            half_l = self._sf(cell["length"] / 2)
            half_w = self._sf(cell["width"] / 2)
            angle = cell["angle"]

            # Compute taper (same seed as contour function)
            taper_rng = np.random.default_rng(ci * 1237 + 99991)
            taper = half_w * float(taper_rng.uniform(0.4, 0.65))
            taper = min(taper, half_l * 0.4)

            # 6 hexagonal vertices
            corners = np.array([
                [-half_l + taper, -half_w],   # TL
                [half_l - taper, -half_w],    # TR
                [half_l, 0],                   # R tip
                [half_l - taper, half_w],     # BR
                [-half_l + taper, half_w],    # BL
                [-half_l, 0],                  # L tip
            ], dtype=np.float32)
            pts = self._transform_contour(corners, cx, cy, angle)

            for px, py in pts:
                if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                    cv2.circle(img, (int(px), int(py)), r, junc_color, -1)

    def _render_nuc(self) -> np.ndarray:
        """Render DAPI nuclear stain at internal resolution."""
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        for cell in self._cells:
            nuc_x = self._s(cell["nuc_x"])
            nuc_y = self._s(cell["nuc_y"])
            nuc_r = self._s(cell["nuc_r"])
            nuc_r2 = max(2, int(nuc_r / cell["nuc_aspect"]))
            angle = cell["angle"]

            intensity = self.rng.uniform(180, 240)
            cv2.ellipse(img, (nuc_x, nuc_y), (nuc_r, nuc_r2),
                         angle, 0, 360, intensity, -1, cv2.LINE_AA)
            cv2.circle(img, (nuc_x, nuc_y), max(1, nuc_r // 3),
                       min(255, intensity + 30), -1)

        img += 4.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_mem(self) -> np.ndarray:
        """Render Calcofluor White cell walls at internal resolution."""
        # Dim base for faint middle lamella fluorescence
        img = np.full((self._ih, self._iw), 20.0, dtype=np.float32)

        for ci, cell in enumerate(self._cells):
            cx, cy = self._sf(cell["cx"]), self._sf(cell["cy"])
            half_l = self._sf(cell["length"] / 2)
            half_w = self._sf(cell["width"] / 2)
            wt = self._sf(self.wall_thickness)
            angle = cell["angle"]

            # Wavy contours (same level seeds as BF for consistency)
            outer_local = self._wavy_rect_contour(half_l, half_w, ci, level=0)
            outer_pts = self._transform_contour(outer_local, cx, cy, angle)

            inner_half_l = half_l - wt
            inner_half_w = half_w - wt
            if inner_half_l <= 0 or inner_half_w <= 0:
                cv2.fillPoly(img, [outer_pts], 200)
                continue

            inner_local = self._wavy_rect_contour(inner_half_l, inner_half_w,
                                                   ci, level=1)
            inner_pts = self._transform_contour(inner_local, cx, cy, angle)

            wall_intensity = self.rng.uniform(170, 210)
            cv2.fillPoly(img, [outer_pts], wall_intensity)
            cv2.fillPoly(img, [inner_pts], 0)

        return np.clip(img, 0, 255).astype(np.uint8)

    # ── Snap frame ──

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs):
        """Capture a frame — compatible with SimulationBridge."""
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
        crop = self._apply_defocus(crop)
        crop = self._pipeline.apply(crop)
        return crop

    def _crop_fov(self, full):
        """Crop FOV from internal-resolution buffer, resize to viewport."""
        s = self.internal_scale
        ih, iw = full.shape[:2]
        out_w, out_h = self.viewport_width, self.viewport_height
        obj = self.current_objectiv

        # FOV in world units
        if obj == 100:
            fov_world = 64
        elif obj == 40:
            fov_world = 128
        elif obj == 20:
            fov_world = 256
        else:
            fov_world = min(512, self.width)

        fov_int = fov_world * s

        # Stage center in world coords → internal coords
        cx_world = int(self.camera_offset[0]) + out_w // 2
        cy_world = int(self.camera_offset[1]) + out_h // 2
        cx_int = int(cx_world * s)
        cy_int = int(cy_world * s)

        half = fov_int // 2
        x0 = max(0, min(cx_int - half, iw - fov_int))
        y0 = max(0, min(cy_int - half, ih - fov_int))

        crop = full[y0:y0 + fov_int, x0:x0 + fov_int].copy()

        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            if self.mode == 0:
                bg = getattr(self, '_tissue_bg', 155)
            elif self.mode == 2:
                bg = 20  # membrane channel base
            else:
                bg = 0
            padded = np.full((fov_int, fov_int), bg, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        if crop.shape[0] > out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        elif crop.shape[0] < out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop

    def _update_mode(self):
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            self.mode = 0
            return
        filt = self.state_devices["Filter Wheel"]
        led = self.state_devices["LED"]
        fl = filt.get("label", filt.get("Label", ""))
        ll = led.get("label", led.get("Label", "CYAN"))
        combined = (fl + " " + ll).upper()
        if "MSCARLET3" in combined or "ORANGE" in combined:
            self.mode = 1
        elif "MIRFP670" in combined or ("RED" in combined and "DAPI" not in combined):
            self.mode = 2
        else:
            for mid, ch in self._extra_channels.items():
                if fl == ch["filter"] and ll == ch["led"]:
                    self.mode = mid
                    return
            self.mode = 0

    def _update_objectif(self):
        if "Objective" not in self.state_devices:
            return
        obj = self.state_devices["Objective"]
        lbl = obj.get("label", obj.get("Label", ""))
        if lbl in self._objectif_dict:
            self.current_objectiv = self._objectif_dict[lbl]
            self._dof = self._dof_table.get(self.current_objectiv, 6.0)

    def set_focal_plane(self, z: float):
        self.focal_plane = z

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply defocus blur based on Z-drift distance from focal plane."""
        dz = abs(self.focal_plane - self.tissue_z)
        if dz < 0.5:
            return img
        sigma = min(dz * 0.5, 12.0)
        ksize = int(sigma * 4) | 1
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift (µm)."""
        return self.tissue_z

    def reset_z_drift(self):
        """Reset Z-drift to zero."""
        self.tissue_z = 0.0

    def update_state(self, dict_state: dict):
        self.state_devices = dict_state

    def apply_salt(self, concentration: float = 0.5):
        """Apply hypertonic salt solution to induce plasmolysis.

        Args:
            concentration: NaCl concentration (0.0 = water, 1.0 = saturated).
                          Plasmolysis onset at ~0.3, full at ~0.8.
        """
        self._plasmolysis_target = min(1.0, max(0.0, (concentration - 0.2) / 0.6))

    def apply_water(self):
        """Wash with distilled water to reverse plasmolysis (deplasmolysis)."""
        self._plasmolysis_target = 0.0

    def step(self, dt: float = 1.0):
        """Advance plasmolysis/deplasmolysis toward target level."""
        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        if abs(self._plasmolysis_level - self._plasmolysis_target) > 0.001:
            if self._plasmolysis_level < self._plasmolysis_target:
                self._plasmolysis_level = min(
                    self._plasmolysis_target,
                    self._plasmolysis_level + self._plasmolysis_rate * dt,
                )
            else:
                self._plasmolysis_level = max(
                    self._plasmolysis_target,
                    self._plasmolysis_level - self._plasmolysis_rate * 0.7 * dt,
                )
            # Re-render to show plasmolysis changes
            self._render_full()

    def update(self, dt: float = 0.016):
        pass

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._noise_rng = np.random.default_rng(seed + 5555)
            self._generate_cells()
            self._render_full()

    def get_ground_truth(self) -> dict:
        """Return ground truth for grading."""
        # Count cells visible in current FOV
        obj = self.current_objectiv
        if obj == 100:
            fov = 64
        elif obj == 40:
            fov = 128
        elif obj == 20:
            fov = 256
        else:
            fov = self.width

        cx = int(self.camera_offset[0]) + self.viewport_width // 2
        cy = int(self.camera_offset[1]) + self.viewport_height // 2
        half = fov // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(self.width, x0 + fov)
        y1 = min(self.height, y0 + fov)

        visible_cells = []
        for i, cell in enumerate(self._cells):
            if x0 <= cell["cx"] <= x1 and y0 <= cell["cy"] <= y1:
                visible_cells.append(i)

        # Cell dimensions
        lengths = [self._cells[i]["length"] for i in visible_cells]
        widths = [self._cells[i]["width"] for i in visible_cells]

        return {
            "n_cells_total": self.n_cells,
            "n_cells_visible": len(visible_cells),
            "mean_cell_length": float(np.mean(lengths)) if lengths else 0,
            "mean_cell_width": float(np.mean(widths)) if widths else 0,
            "wall_thickness": self.wall_thickness,
            "staining": self.staining,
            "tissue_angle": self._tissue_angle,
            "plasmolysis_level": round(self._plasmolysis_level, 3),
            "cell_positions": [
                {"x": self._cells[i]["cx"], "y": self._cells[i]["cy"],
                 "nuc_x": self._cells[i]["nuc_x"], "nuc_y": self._cells[i]["nuc_y"],
                 "length": self._cells[i]["length"], "width": self._cells[i]["width"]}
                for i in visible_cells
            ],
        }

    def add_channel(self, channel_id, name, filter_label, led_label, image=None):
        self._extra_channels[channel_id] = {
            "name": name, "filter": filter_label,
            "led": led_label, "image": image,
        }
