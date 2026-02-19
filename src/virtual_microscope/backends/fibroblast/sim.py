"""Fibroblast simulator with subcellular structure.

Large elongated cells with stress fibers, focal adhesions, and
lamellipodia. The key feature is multi-scale detail:
  - 10x: cell shapes and distribution (elongated blobs)
  - 20x: stress fibers become visible (bright lines)
  - 40x: individual stress fibers and focal adhesions (bright dots)

World scale: ~1 pixel = 1 µm. Typical fibroblast ~90 × 15 µm.

Channels:
  - BF (mode 0): cell body as dark shape on bright agar
  - Nucleus (mode 1): DAPI-stained nuclei (bright oval at cell center)
  - Actin (mode 2): phalloidin staining — stress fibers + lamellipodia
"""

import math
import numpy as np
import cv2
from virtual_microscope.optical_pipeline import OpticalPipeline


class FibroblastSim:
    """Virtual fibroblast culture with subcellular detail."""

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_cells: int = 8,
        seed: int = 42,
        fixed_dt: float = 0.0,
        internal_scale: int = 4,
    ):
        self.width = world_size
        self.height = world_size
        self.internal_scale = internal_scale
        self._iw = world_size * internal_scale
        self._ih = world_size * internal_scale
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self.rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 5555)

        # Optical pipelines per channel
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.8, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.10, rng_seed=seed + 300),
            1: OpticalPipeline(
                psf_sigma=1.5, noise={"photon_scale": 3.5, "read_std": 3.0},
                vignette=0.12, rng_seed=seed + 301),
            2: OpticalPipeline(
                psf_sigma=1.2, noise={"photon_scale": 3.5, "read_std": 3.0},
                vignette=0.12, rng_seed=seed + 302),
        }

        # SimulationBridge interface
        self.state_devices = {}
        self.mode = 0  # 0=BF, 1=nucleus, 2=actin
        self.camera_offset = [0, 0]
        self.focal_plane = 0.0
        self.tissue_z = 0.0
        self.current_objectiv = 0  # 0=10x, 1=20x, 2=40x
        self._snap_count = 0
        self.auto_step = False
        self.snaps_per_step = 1
        self.fixed_dt = fixed_dt

        # Z-drift (thermal/mechanical drift during timelapse)
        self.z_drift_rate = 0.0
        self.z_drift_noise = 0.0

        # Extra channels (e.g. vinculin for focal adhesions)
        self._extra_channels = {}

        # Drug response state
        self._drug_active = False
        self._drug_name = None
        self._drug_effect = 0.0  # 0=no effect, 1=full
        self._drug_washing_out = False
        self._drug_profiles = {
            "cytochalasin_d": {
                "target_effect": 1.0,
                "onset_rate": 0.08,
                "washout_rate": 0.04,
                "fiber_dimming": 0.9,
                "fa_dimming": 0.85,
                "cell_swelling": 0.3,
                "lamellipodium_loss": True,
            },
            "latrunculin_a": {
                "target_effect": 1.0,
                "onset_rate": 0.10,
                "washout_rate": 0.03,
                "fiber_dimming": 0.95,
                "fa_dimming": 0.9,
                "cell_swelling": 0.2,
                "lamellipodium_loss": True,
            },
            "y27632": {
                "target_effect": 0.8,
                "onset_rate": 0.06,
                "washout_rate": 0.05,
                "fiber_dimming": 0.6,
                "fa_dimming": 0.5,
                "cell_swelling": -0.15,
                "lamellipodium_loss": False,
            },
        }
        self._time = 0.0
        self._step_count = 0
        self._actin_dirty = True

        # Temperature-induced actin effects (cold shock)
        self._cold_actin_loss = 0.0  # 0=normal, 1=fully depolymerized

        # Mechanical stretch state
        self._stretch_strain = 0.0     # current strain (0-0.20)
        self._stretch_direction = 0.0  # stretch axis angle (radians), 0=horizontal
        self._bf_dirty = False  # BF needs re-render when cells reorient

        # Generate cells
        self._cells = []
        self._generate_cells(n_cells)
        self._original_angles = [c["angle"] for c in self._cells]

        # Dead cells: small bright refractile spheres (always present in culture)
        n_dead = max(1, self.rng.poisson(max(1, n_cells // 5)))
        self._dead_cells = []
        for _ in range(n_dead):
            self._dead_cells.append({
                "cx": self.rng.uniform(10, self.width - 10),
                "cy": self.rng.uniform(10, self.height - 10),
                "radius": self.rng.uniform(3.0, 6.0),
            })

        # Pre-render full-world images
        self._bf_full = None
        self._nuc_full = None
        self._actin_full = None
        self._render_all()
        self._actin_dirty = False

    def _generate_cells(self, n_cells: int):
        """Generate fibroblast positions, shapes, and subcellular features."""
        margin = 60
        for _ in range(n_cells):
            # Place cells with non-overlapping check
            for _attempt in range(100):
                cx = self.rng.uniform(margin, self.width - margin)
                cy = self.rng.uniform(margin, self.height - margin)
                if not self._overlaps(cx, cy):
                    break

            angle = self.rng.uniform(0, math.pi)  # orientation
            length = self.rng.uniform(70, 110)  # µm (long axis)
            width = self.rng.uniform(12, 22)  # µm (short axis)

            # Nucleus: oval at center, ~15×10 µm
            nuc_length = self.rng.uniform(12, 18)
            nuc_width = self.rng.uniform(8, 12)
            nuc_offset = self.rng.uniform(-5, 5)  # slight off-center

            # Stress fibers — four biological subtypes
            fibers = []

            # 1. Ventral stress fibers: thick, span full cell,
            #    anchored at focal adhesions at both ends
            n_ventral = self.rng.integers(2, 5)
            for _ in range(n_ventral):
                lat_offset = self.rng.uniform(-width * 0.35, width * 0.35)
                start_frac = self.rng.uniform(-0.45, -0.35)
                end_frac = self.rng.uniform(0.35, 0.45)
                angle_dev = self.rng.uniform(-0.05, 0.05)
                fiber_width = self.rng.uniform(2.0, 3.5)
                brightness = self.rng.uniform(190, 240)
                fibers.append({
                    "type": "ventral",
                    "lat_offset": lat_offset,
                    "start_frac": start_frac,
                    "end_frac": end_frac,
                    "angle_dev": angle_dev,
                    "width": fiber_width,
                    "brightness": brightness,
                })

            # 2. Dorsal stress fibers: thinner, from edge to center,
            #    anchored at one focal adhesion
            n_dorsal = self.rng.integers(2, 6)
            for _ in range(n_dorsal):
                lat_offset = self.rng.uniform(-width * 0.4, width * 0.4)
                if self.rng.random() < 0.5:
                    start_frac = self.rng.uniform(-0.45, -0.3)
                    end_frac = self.rng.uniform(-0.1, 0.1)
                else:
                    start_frac = self.rng.uniform(-0.1, 0.1)
                    end_frac = self.rng.uniform(0.3, 0.45)
                angle_dev = self.rng.uniform(-0.15, 0.15)
                fiber_width = self.rng.uniform(1.0, 2.0)
                brightness = self.rng.uniform(140, 190)
                fibers.append({
                    "type": "dorsal",
                    "lat_offset": lat_offset,
                    "start_frac": start_frac,
                    "end_frac": end_frac,
                    "angle_dev": angle_dev,
                    "width": fiber_width,
                    "brightness": brightness,
                })

            # 3. Perinuclear actin cap: thick fibers over nucleus
            n_cap = self.rng.integers(0, 3)
            nuc_frac_half = nuc_length / (2 * length)
            for _ in range(n_cap):
                lat_offset = self.rng.uniform(-nuc_width * 0.3, nuc_width * 0.3)
                start_frac = self.rng.uniform(
                    -nuc_frac_half - 0.12, -nuc_frac_half)
                end_frac = self.rng.uniform(
                    nuc_frac_half, nuc_frac_half + 0.12)
                angle_dev = self.rng.uniform(-0.03, 0.03)
                fiber_width = self.rng.uniform(2.5, 4.0)
                brightness = self.rng.uniform(200, 250)
                fibers.append({
                    "type": "perinuclear_cap",
                    "lat_offset": lat_offset,
                    "start_frac": start_frac,
                    "end_frac": end_frac,
                    "angle_dev": angle_dev,
                    "width": fiber_width,
                    "brightness": brightness,
                })

            # 4. Transverse arcs: curved fibers perpendicular to cell axis
            n_arcs = self.rng.integers(1, 4)
            arcs = []
            for _ in range(n_arcs):
                t_frac = self.rng.uniform(-0.15, 0.15)
                half_span = self.rng.uniform(width * 0.25, width * 0.4)
                curvature = self.rng.uniform(0.01, 0.04)
                arc_w = self.rng.uniform(1.0, 2.0)
                arc_b = self.rng.uniform(120, 170)
                arcs.append({
                    "type": "transverse_arc",
                    "t_frac": t_frac,
                    "half_span": half_span,
                    "curvature": curvature,
                    "width": arc_w,
                    "brightness": arc_b,
                })

            # Focal adhesions: bright dots at cell periphery
            n_fa = self.rng.integers(15, 40)
            focal_adhesions = []
            for _ in range(n_fa):
                # Position along cell edge
                t = self.rng.uniform(-0.45, 0.45)
                # Preferentially at ends and edges
                side = self.rng.choice([-1, 1])
                lat = side * width * self.rng.uniform(0.3, 0.5)
                fa_size = self.rng.uniform(1.5, 4.0)
                fa_brightness = self.rng.uniform(180, 255)
                focal_adhesions.append({
                    "t": t,
                    "lat": lat,
                    "size": fa_size,
                    "brightness": fa_brightness,
                })

            # Lamellipodia: thin sheet at leading edge
            has_lamellipodium = self.rng.random() < 0.6
            lamellipodium_extent = self.rng.uniform(8, 20) if has_lamellipodium else 0
            lamellipodium_side = self.rng.choice([-1, 1])  # which end

            cell = {
                "cx": cx, "cy": cy, "angle": angle,
                "length": length, "width": width,
                "nuc_length": nuc_length, "nuc_width": nuc_width,
                "nuc_offset": nuc_offset,
                "fibers": fibers,
                "arcs": arcs,
                "focal_adhesions": focal_adhesions,
                "has_lamellipodium": has_lamellipodium,
                "lamellipodium_extent": lamellipodium_extent,
                "lamellipodium_side": lamellipodium_side,
                "nuc_brightness": self.rng.uniform(180, 240),
            }
            self._cells.append(cell)

    def _overlaps(self, cx, cy, min_dist=80):
        """Check if new position overlaps existing cells."""
        for c in self._cells:
            dx = cx - c["cx"]
            dy = cy - c["cy"]
            if math.sqrt(dx * dx + dy * dy) < min_dist:
                return True
        return False

    def _s(self, v):
        """Scale world coordinate to internal resolution (int)."""
        return int(round(v * self.internal_scale))

    def _sf(self, v):
        """Scale world coordinate to internal resolution (float)."""
        return v * self.internal_scale

    def _cell_to_world(self, cell, t, lat):
        """Convert cell-local coords (t=along axis, lat=perpendicular) to world."""
        ca = math.cos(cell["angle"])
        sa = math.sin(cell["angle"])
        x = cell["cx"] + t * ca - lat * sa
        y = cell["cy"] + t * sa + lat * ca
        return x, y

    def _cell_to_internal(self, cell, t, lat):
        """Convert cell-local coords to internal resolution."""
        x, y = self._cell_to_world(cell, t, lat)
        return self._sf(x), self._sf(y)

    def _render_all(self):
        """Pre-render all full-world channel images."""
        self._bf_full = self._render_bf_full()
        self._nuc_full = self._render_nuc_full()
        self._actin_full = self._render_actin_full()

    def _render_bf_full(self) -> np.ndarray:
        """Render brightfield with phase-contrast-like optics.

        Features:
        - Phase halo: bright rim around cell boundary (phase contrast artifact)
        - Shade-off: interior lightens toward center (thick phase objects)
        - Cytoplasmic organelle texture: granularity visible at 40x
        - Nucleus: dense with visible nucleolus
        """
        bg_val = 140  # phase contrast background (medium gray)
        img = np.full((self._ih, self._iw, 3), bg_val, dtype=np.uint8)

        # Substrate texture (generate at world res, upscale)
        noise = self._noise_rng.normal(0, 2.5, (self.height, self.width)).astype(np.float32)
        if self.internal_scale > 1:
            noise = cv2.resize(noise, (self._iw, self._ih),
                               interpolation=cv2.INTER_LINEAR)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch].astype(float) + noise, 0, 255)

        s = self.internal_scale
        blur_k = max(3, s * 3) | 1

        for cell in self._cells:
            center = (self._s(cell["cx"]), self._s(cell["cy"]))
            half_len = self._s(cell["length"] / 2)
            half_wid = self._s(cell["width"] / 2)
            angle_deg = math.degrees(cell["angle"])

            # --- Phase halo: bright rim around cell boundary ---
            halo_axes = (half_len + self._s(3), half_wid + self._s(3))
            halo_mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
            cv2.ellipse(halo_mask, center, halo_axes, angle_deg, 0, 360, 255, -1)
            inner_mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
            cv2.ellipse(inner_mask, center, (half_len, half_wid),
                        angle_deg, 0, 360, 255, -1)
            halo_ring = cv2.subtract(halo_mask, inner_mask)
            halo_ring = cv2.GaussianBlur(halo_ring, (blur_k, blur_k), 1.2 * s)
            halo_add = (halo_ring.astype(np.float32) / 255.0) * 55  # bright halo
            for ch in range(3):
                img[:, :, ch] = np.clip(
                    img[:, :, ch].astype(np.float32) + halo_add, 0, 255
                ).astype(np.uint8)

            # --- Cell body: dark with shade-off (interior brightening) ---
            # Real fibroblasts are very thin at the periphery (<1µm) and
            # thick over the nucleus (~5-10µm). Phase contrast darkening
            # is proportional to optical path length → thin edges are
            # nearly transparent, thick center is darker.
            cell_mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
            cv2.ellipse(cell_mask, center, (half_len, half_wid),
                        angle_deg, 0, 360, 255, -1)

            # Distance-from-edge mask: 0 at boundary, 1 deep inside
            erode_k = max(3, self._s(8)) | 1
            interior = cv2.erode(cell_mask, np.ones((erode_k, erode_k), np.uint8))
            interior = cv2.GaussianBlur(interior, (blur_k, blur_k), 2.0 * s)
            depth_frac = interior.astype(np.float32) / 255.0

            # Phase contrast: cell interior is darker than background.
            # Shade-off makes the very center slightly lighter, but edges
            # of the cell body are darker (steepest phase gradient).
            body_edge = 65    # cell body edge: dark phase contrast
            body_center = 95  # shade-off: center trends back toward bg
            body_val = body_edge + depth_frac * (body_center - body_edge)

            # Thickness-dependent opacity: periphery is nearly transparent
            # (thin cytoplasm), nucleus region is opaque (thick)
            thickness = depth_frac ** 0.6  # nonlinear: fast rise from edge
            cell_alpha = cell_mask.astype(np.float32) / 255.0
            cell_alpha = cv2.GaussianBlur(cell_alpha, (blur_k, blur_k), 1.0 * s)
            # Modulate opacity by thickness: thin edges barely visible
            cell_alpha *= (0.15 + 0.85 * thickness)

            for ch in range(3):
                img[:, :, ch] = (
                    cell_alpha * body_val +
                    (1 - cell_alpha) * img[:, :, ch].astype(np.float32)
                ).clip(0, 255).astype(np.uint8)

            # --- Cytoplasmic organelle texture (visible at 40x) ---
            # Small dark spots for mitochondria/vesicles
            n_organelles = self._noise_rng.integers(30, 60)
            ca = math.cos(cell["angle"])
            sa = math.sin(cell["angle"])
            for _ in range(n_organelles):
                # Random position inside cell ellipse
                t = self._noise_rng.uniform(-0.4, 0.4) * cell["length"]
                lat = self._noise_rng.uniform(-0.35, 0.35) * cell["width"]
                ox = self._sf(cell["cx"] + t * ca - lat * sa)
                oy = self._sf(cell["cy"] + t * sa + lat * ca)
                ix, iy = int(ox), int(oy)
                if 0 <= ix < self._iw and 0 <= iy < self._ih:
                    r = max(1, self._s(self._noise_rng.uniform(0.5, 1.5)))
                    dark = int(self._noise_rng.uniform(40, 65))
                    cv2.circle(img, (ix, iy), r, (dark, dark, dark), -1,
                               lineType=cv2.LINE_AA)

            # --- Nucleus: dense phase object, darker with nucleolus ---
            nuc_center, nuc_axes = self._get_nucleus_ellipse(cell)
            # Nucleus body (dense, dark in phase contrast)
            cv2.ellipse(img, nuc_center, nuc_axes, angle_deg, 0, 360,
                        (55, 55, 55), -1)
            # Slight shade-off inside nucleus
            inner_nuc = (max(1, nuc_axes[0] - self._s(2)),
                         max(1, nuc_axes[1] - self._s(1.5)))
            cv2.ellipse(img, nuc_center, inner_nuc, angle_deg, 0, 360,
                        (65, 65, 65), -1)
            # Nucleolus: 1-2 dark round spots
            n_nucleoli = self._noise_rng.integers(1, 3)
            for _ in range(n_nucleoli):
                nx = self._noise_rng.uniform(-nuc_axes[0] * 0.4, nuc_axes[0] * 0.4)
                ny = self._noise_rng.uniform(-nuc_axes[1] * 0.4, nuc_axes[1] * 0.4)
                px = int(nuc_center[0] + nx * ca - ny * sa)
                py = int(nuc_center[1] + nx * sa + ny * ca)
                if 0 <= px < self._iw and 0 <= py < self._ih:
                    nr = max(2, self._s(1.5))
                    cv2.circle(img, (px, py), nr, (35, 35, 35), -1,
                               lineType=cv2.LINE_AA)

            # --- Stress fibers in BF (subtle dark lines, visible at 40x+) ---
            # In phase contrast, actin bundles are dense → slight dark streaks
            for fiber in cell["fibers"]:
                self._draw_stress_fiber_bf(img, cell, fiber)
            for arc in cell.get("arcs", []):
                self._draw_transverse_arc_bf(img, cell, arc)

            # Lamellipodium: very thin, nearly transparent leading edge
            if cell["has_lamellipodium"]:
                self._draw_lamellipodium_bf(img, cell)

        # Dead cells: bright refractile spheres (floating, out-of-focus)
        for dc in self._dead_cells:
            dcx = self._s(dc["cx"])
            dcy = self._s(dc["cy"])
            dcr = self._s(dc["radius"])
            # Bright sphere with dark rim (refractile)
            cv2.circle(img, (dcx, dcy), dcr + self._s(1),
                       (100, 100, 100), -1, cv2.LINE_AA)
            cv2.circle(img, (dcx, dcy), dcr,
                       (220, 220, 220), -1, cv2.LINE_AA)
            # Central highlight
            cv2.circle(img, (dcx - dcr // 3, dcy - dcr // 3),
                       max(1, dcr // 3), (240, 240, 240), -1, cv2.LINE_AA)

        return img

    def _render_nuc_full(self) -> np.ndarray:
        """Render nucleus channel — DAPI-stained nuclei with textured detail.

        Features matching real DAPI fluorescence:
        - Nuclear envelope brightening (DNA-rich periphery)
        - Nucleoli as dark voids (rRNA, not DNA-dense)
        - Heterochromatin foci (bright condensed spots)
        - Coarse chromatin texture (lumpy pattern)
        """
        img = np.full((self._ih, self._iw, 3), 5, dtype=np.uint8)

        s = self.internal_scale

        for cell in self._cells:
            nuc_center, nuc_axes = self._get_nucleus_ellipse(cell)
            angle_deg = math.degrees(cell["angle"])
            brightness = int(cell["nuc_brightness"])
            ca = math.cos(cell["angle"])
            sa = math.sin(cell["angle"])

            # --- Base nucleus fill (uniform) ---
            base_b = int(brightness * 0.7)
            cv2.ellipse(img, nuc_center, nuc_axes, angle_deg, 0, 360,
                        (base_b, base_b, base_b), -1)

            # --- Nuclear envelope brightening (DNA-rich lamina) ---
            envelope_thick = max(2, self._s(1.5))
            envelope_b = min(255, brightness + 20)
            cv2.ellipse(img, nuc_center, nuc_axes, angle_deg, 0, 360,
                        (envelope_b, envelope_b, envelope_b), envelope_thick,
                        lineType=cv2.LINE_AA)

            # --- Coarse chromatin texture (lumpy DAPI pattern) ---
            # Generate random blobs inside the nucleus
            n_blobs = self._noise_rng.integers(6, 12)
            for _ in range(n_blobs):
                # Random position in nucleus-aligned coords
                bx = self._noise_rng.uniform(-0.7, 0.7) * nuc_axes[0]
                by = self._noise_rng.uniform(-0.7, 0.7) * nuc_axes[1]
                # Check if inside ellipse
                if (bx / nuc_axes[0]) ** 2 + (by / nuc_axes[1]) ** 2 > 0.85:
                    continue
                px = int(nuc_center[0] + bx * ca - by * sa)
                py = int(nuc_center[1] + bx * sa + by * ca)
                if 0 <= px < self._iw and 0 <= py < self._ih:
                    blob_r = max(2, self._s(self._noise_rng.uniform(1.0, 3.0)))
                    blob_b = int(brightness * self._noise_rng.uniform(0.8, 1.1))
                    blob_b = min(255, blob_b)
                    cv2.circle(img, (px, py), blob_r,
                               (blob_b, blob_b, blob_b), -1,
                               lineType=cv2.LINE_AA)

            # --- Heterochromatin foci (bright condensed DNA spots) ---
            n_foci = self._noise_rng.integers(3, 7)
            for _ in range(n_foci):
                fx = self._noise_rng.uniform(-0.6, 0.6) * nuc_axes[0]
                fy = self._noise_rng.uniform(-0.6, 0.6) * nuc_axes[1]
                if (fx / nuc_axes[0]) ** 2 + (fy / nuc_axes[1]) ** 2 > 0.7:
                    continue
                px = int(nuc_center[0] + fx * ca - fy * sa)
                py = int(nuc_center[1] + fx * sa + fy * ca)
                if 0 <= px < self._iw and 0 <= py < self._ih:
                    foci_r = max(1, self._s(self._noise_rng.uniform(0.5, 1.5)))
                    foci_b = min(255, int(brightness * 1.2))
                    cv2.circle(img, (px, py), foci_r,
                               (foci_b, foci_b, foci_b), -1,
                               lineType=cv2.LINE_AA)

            # --- Nucleoli (dark voids — rRNA-rich, DAPI-dim) ---
            n_nucleoli = self._noise_rng.integers(1, 3)
            for _ in range(n_nucleoli):
                nx = self._noise_rng.uniform(-0.35, 0.35) * nuc_axes[0]
                ny = self._noise_rng.uniform(-0.35, 0.35) * nuc_axes[1]
                px = int(nuc_center[0] + nx * ca - ny * sa)
                py = int(nuc_center[1] + nx * sa + ny * ca)
                if 0 <= px < self._iw and 0 <= py < self._ih:
                    nuc_r = max(2, self._s(self._noise_rng.uniform(1.5, 2.5)))
                    # Dark void with slight rim
                    dark_b = int(base_b * 0.3)
                    cv2.circle(img, (px, py), nuc_r,
                               (dark_b, dark_b, dark_b), -1,
                               lineType=cv2.LINE_AA)

        # Faint background noise (generate at world res, upscale)
        noise = self._noise_rng.normal(0, 2, (self.height, self.width, 3)).astype(np.float32)
        if s > 1:
            noise = cv2.resize(noise, (self._iw, self._ih),
                               interpolation=cv2.INTER_LINEAR)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

        return img

    def _render_actin_full(self) -> np.ndarray:
        """Render actin channel — stress fibers, lamellipodia, focal adhesions.

        Drug effect modulates brightness of stress fibers, focal adhesions,
        and cytoplasmic actin. Lamellipodia retract with some drugs.
        """
        img = np.full((self._ih, self._iw, 3), 3, dtype=np.uint8)

        s = self.internal_scale
        de = self._drug_effect
        profile = self._drug_profiles.get(self._drug_name, {}) if self._drug_name else {}
        fiber_mult = 1.0 - de * profile.get("fiber_dimming", 0)
        fa_mult = 1.0 - de * profile.get("fa_dimming", 0)
        lam_lost = de > 0.5 and profile.get("lamellipodium_loss", False)

        # Cold-induced actin depolymerization stacks with drug effects
        cold = self._cold_actin_loss
        if cold > 0:
            fiber_mult *= (1.0 - cold * 0.85)
            fa_mult *= (1.0 - cold * 0.7)
            if cold > 0.4:
                lam_lost = True  # lamellipodia retract in cold

        for cell in self._cells:
            # Diffuse cytoplasmic actin (dims slightly with drug)
            center = (self._s(cell["cx"]), self._s(cell["cy"]))
            axes = (self._s(cell["length"] / 2), self._s(cell["width"] / 2))
            angle_deg = math.degrees(cell["angle"])
            cyto_b = int(30 * max(0.2, 1 - de * 0.5))
            cv2.ellipse(img, center, axes, angle_deg, 0, 360,
                        (cyto_b, cyto_b, cyto_b), -1)

            # Stress fibers (dimmed by drug)
            for fiber in cell["fibers"]:
                self._draw_stress_fiber(img, cell, fiber,
                                        brightness_mult=fiber_mult)

            # Transverse arcs (dimmed by drug)
            for arc in cell.get("arcs", []):
                self._draw_transverse_arc(img, cell, arc,
                                          brightness_mult=fiber_mult)

            # Focal adhesions (dimmed by drug)
            for fa in cell["focal_adhesions"]:
                self._draw_focal_adhesion(img, cell, fa,
                                          brightness_mult=fa_mult)

            # Lamellipodium: retracted with some drugs
            if cell["has_lamellipodium"] and not lam_lost:
                self._draw_lamellipodium_actin(img, cell)

        # Add noise (generate at world res, upscale)
        noise = self._noise_rng.normal(0, 2, (self.height, self.width, 3)).astype(np.float32)
        if s > 1:
            noise = cv2.resize(noise, (self._iw, self._ih),
                               interpolation=cv2.INTER_LINEAR)
        img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

        return img

    def _get_nucleus_ellipse(self, cell):
        """Get nucleus center and axes in internal coordinates."""
        ca = math.cos(cell["angle"])
        sa = math.sin(cell["angle"])
        nuc_off = cell["nuc_offset"]
        nuc_cx = self._s(cell["cx"] + nuc_off * ca)
        nuc_cy = self._s(cell["cy"] + nuc_off * sa)
        nuc_axes = (self._s(cell["nuc_length"] / 2), self._s(cell["nuc_width"] / 2))
        return (nuc_cx, nuc_cy), nuc_axes

    def _draw_stress_fiber(self, img, cell, fiber, brightness_mult=1.0):
        """Draw a single stress fiber as a bright line."""
        total_angle = cell["angle"] + fiber["angle_dev"]
        ca = math.cos(total_angle)
        sa = math.sin(total_angle)
        cell_ca = math.cos(cell["angle"])
        cell_sa = math.sin(cell["angle"])

        # Start and end in world coords, then scale to internal
        t_start = fiber["start_frac"] * cell["length"]
        t_end = fiber["end_frac"] * cell["length"]
        lat = fiber["lat_offset"]

        x1 = self._sf(cell["cx"] + t_start * ca - lat * cell_sa)
        y1 = self._sf(cell["cy"] + t_start * sa + lat * cell_ca)
        x2 = self._sf(cell["cx"] + t_end * ca - lat * cell_sa)
        y2 = self._sf(cell["cy"] + t_end * sa + lat * cell_ca)

        brightness = int(fiber["brightness"] * brightness_mult)
        if brightness < 5:
            return  # fiber effectively invisible
        thickness = max(1, self._s(fiber["width"]))

        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 (brightness, brightness, brightness), thickness,
                 lineType=cv2.LINE_AA)

    def _draw_stress_fiber_bf(self, img, cell, fiber):
        """Draw a stress fiber in BF as a subtle dark line.

        In phase contrast, actin bundles have higher refractive index than
        surrounding cytoplasm, so they appear as faint dark streaks.
        Only really visible at 40x+ magnification.
        """
        total_angle = cell["angle"] + fiber["angle_dev"]
        ca = math.cos(total_angle)
        sa = math.sin(total_angle)
        cell_ca = math.cos(cell["angle"])
        cell_sa = math.sin(cell["angle"])

        t_start = fiber["start_frac"] * cell["length"]
        t_end = fiber["end_frac"] * cell["length"]
        lat = fiber["lat_offset"]

        x1 = self._sf(cell["cx"] + t_start * ca - lat * cell_sa)
        y1 = self._sf(cell["cy"] + t_start * sa + lat * cell_ca)
        x2 = self._sf(cell["cx"] + t_end * ca - lat * cell_sa)
        y2 = self._sf(cell["cy"] + t_end * sa + lat * cell_ca)

        # Faint dark line (subtle contrast against ~80-95 shade-off interior)
        dark_val = int(55 + (fiber["brightness"] / 240.0) * 15)  # 55-70 range
        thickness = max(1, self._s(fiber["width"] * 0.6))  # thinner than actin

        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 (dark_val, dark_val, dark_val), thickness,
                 lineType=cv2.LINE_AA)

    def _draw_transverse_arc(self, img, cell, arc, brightness_mult=1.0):
        """Draw a transverse arc — curved fiber perpendicular to cell axis.

        Transverse arcs are contractile actin bundles that run across the
        cell width in the lamella region.  They curve gently (convex toward
        the cell edge) and sweep centripetally.
        """
        brightness = int(arc["brightness"] * brightness_mult)
        if brightness < 5:
            return
        thickness = max(1, self._s(arc["width"]))

        # Sample the arc as a polyline
        n_pts = 12
        pts = []
        for i in range(n_pts):
            frac = (i / (n_pts - 1)) * 2 - 1  # -1 to 1
            lat = frac * arc["half_span"]
            # Quadratic curvature: arc bows along the long axis
            t = (arc["t_frac"] + arc["curvature"] * frac * frac) * cell["length"]
            x, y = self._cell_to_internal(cell, t, lat)
            pts.append([int(x), int(y)])

        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], False,
                      (brightness, brightness, brightness), thickness,
                      lineType=cv2.LINE_AA)

    def _draw_transverse_arc_bf(self, img, cell, arc):
        """Draw a transverse arc in BF as a faint dark curve."""
        n_pts = 12
        pts = []
        for i in range(n_pts):
            frac = (i / (n_pts - 1)) * 2 - 1
            lat = frac * arc["half_span"]
            t = (arc["t_frac"] + arc["curvature"] * frac * frac) * cell["length"]
            x, y = self._cell_to_internal(cell, t, lat)
            pts.append([int(x), int(y)])

        dark_val = int(60 + (arc["brightness"] / 170.0) * 10)
        thickness = max(1, self._s(arc["width"] * 0.6))
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts_arr], False,
                      (dark_val, dark_val, dark_val), thickness,
                      lineType=cv2.LINE_AA)

    def _draw_focal_adhesion(self, img, cell, fa, brightness_mult=1.0):
        """Draw a focal adhesion as a bright dot."""
        t = fa["t"] * cell["length"]
        lat = fa["lat"]
        x, y = self._cell_to_internal(cell, t, lat)
        ix, iy = int(x), int(y)
        if 0 <= ix < self._iw and 0 <= iy < self._ih:
            brightness = int(fa["brightness"] * brightness_mult)
            if brightness < 5:
                return  # FA effectively invisible
            radius = max(1, self._s(fa["size"] / 2))
            cv2.circle(img, (ix, iy), radius,
                       (brightness, brightness, brightness), -1,
                       lineType=cv2.LINE_AA)

    def _draw_lamellipodium_bf(self, img, cell):
        """Draw lamellipodium in BF — very thin, nearly transparent fan."""
        side = cell["lamellipodium_side"]
        extent = cell["lamellipodium_extent"]

        # Fan-shaped region at one end of the cell
        base_t = side * cell["length"] / 2
        tip_t = base_t + side * extent
        n_pts = 8
        pts = []
        half_w = cell["width"] * 0.6
        for i in range(n_pts):
            frac = i / (n_pts - 1)
            t = base_t + (tip_t - base_t) * frac
            w = half_w * (1 - 0.5 * frac)  # tapers outward
            for sv in [-1, 1]:
                x, y = self._cell_to_internal(cell, t, sv * w)
                pts.append([int(x), int(y)])

        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            overlay = img.copy()
            # Lamellipodium: slightly darker than background (thin phase object)
            cv2.fillConvexPoly(overlay, hull, (115, 115, 115))
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _draw_lamellipodium_actin(self, img, cell):
        """Draw lamellipodium in actin channel (bright diffuse region)."""
        side = cell["lamellipodium_side"]
        extent = cell["lamellipodium_extent"]
        s = self.internal_scale

        base_t = side * cell["length"] / 2
        tip_t = base_t + side * extent
        n_pts = 8
        pts = []
        half_w = cell["width"] * 0.7
        for i in range(n_pts):
            frac = i / (n_pts - 1)
            t = base_t + (tip_t - base_t) * frac
            w = half_w * (1 - 0.3 * frac)
            for sv in [-1, 1]:
                x, y = self._cell_to_internal(cell, t, sv * w)
                pts.append([int(x), int(y)])

        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            blur_k = max(3, s * 9) | 1
            mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 3 * s)
            brightness = 80
            for c in range(3):
                img[:, :, c] = np.clip(
                    img[:, :, c].astype(float) + mask * (brightness / 255.0),
                    0, 255
                ).astype(np.uint8)

    # ----------------------------------------------------------------
    # SimulationBridge interface
    # ----------------------------------------------------------------

    def _update_mode(self):
        """Update rendering mode from device state."""
        led_state = self.state_devices.get("LED", {})
        fw_state = self.state_devices.get("Filter Wheel", {})
        led = led_state.get("Label", led_state.get("label", "CYAN"))
        fw = fw_state.get("Label", fw_state.get("label", ""))

        if led == "CYAN" or "brightfield" in fw.lower():
            self.mode = 0
        elif led == "ORANGE" or "mScarlet3" in fw:
            self.mode = 1
        elif led == "RED" or "miRFP670" in fw:
            self.mode = 2
        else:
            self.mode = self.state_devices.get("_mode_override", 0)

    def _update_objectif(self):
        """Update objective from device state."""
        obj_state = self.state_devices.get("Objective", {})
        label = obj_state.get("Label", obj_state.get("label", "10x"))
        if "40" in str(label):
            self.current_objectiv = 2
        elif "20" in str(label):
            self.current_objectiv = 1
        else:
            self.current_objectiv = 0

    def _crop_fov(self, full_img: np.ndarray) -> np.ndarray:
        """Crop FOV from internal-resolution image based on objective."""
        s = self.internal_scale
        # FOV in world pixels: 10x=512, 20x=256, 40x=128
        fov_map = {0: 512, 1: 256, 2: 128}
        fov_world = fov_map.get(self.current_objectiv, 512)
        fov_int = fov_world * s

        # Center FOV on stage position at any magnification
        cx_world = int(self.camera_offset[0]) + self.viewport_width // 2
        cy_world = int(self.camera_offset[1]) + self.viewport_height // 2
        ox = cx_world * s - fov_int // 2
        oy = cy_world * s - fov_int // 2

        h, w = full_img.shape[:2]
        bg_val = 140 if self.mode == 0 else 0

        ndim = len(full_img.shape)
        if ndim == 3:
            crop = np.full((fov_int, fov_int, full_img.shape[2]), bg_val, dtype=np.uint8)
        else:
            crop = np.full((fov_int, fov_int), bg_val, dtype=np.uint8)

        # Source region in internal image
        x1 = max(0, ox)
        y1 = max(0, oy)
        x2 = min(w, ox + fov_int)
        y2 = min(h, oy + fov_int)

        dst_x1 = max(0, -ox)
        dst_y1 = max(0, -oy)
        src_w = x2 - x1
        src_h = y2 - y1
        if src_w > 0 and src_h > 0:
            crop[dst_y1:dst_y1 + src_h, dst_x1:dst_x1 + src_w] = (
                full_img[y1:y2, x1:x2]
            )

        # Resize to viewport
        out_w, out_h = self.viewport_width, self.viewport_height
        interp = cv2.INTER_AREA if fov_int > out_w else cv2.INTER_LINEAR
        return cv2.resize(crop, (out_w, out_h), interpolation=interp)

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply defocus blur based on focal plane distance from tissue."""
        dz = abs(self.focal_plane - self.tissue_z)
        if dz < 1.0:
            return img
        sigma = min(dz * 0.5, 30.0)
        return cv2.GaussianBlur(img, (0, 0), sigma)

    def update_state(self, state_dict: dict):
        """Update device state from bridge."""
        for dev, props in state_dict.items():
            if dev not in self.state_devices:
                self.state_devices[dev] = {}
            self.state_devices[dev].update(props)

    def set_focal_plane(self, z: float):
        """Set focal plane position."""
        self.focal_plane = z

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift (µm)."""
        return self.tissue_z

    def reset_z_drift(self):
        """Reset Z-drift to zero."""
        self.tissue_z = 0.0

    def enable_photobleaching(self, rate: float = 0.001):
        """Enable photobleaching on fluorescence channels (1=nucleus, 2=actin)."""
        for mode in [1, 2]:
            if mode in self._pipeline:
                self._pipeline[mode].photobleach_rate = rate

    def reset_photobleaching(self):
        """Reset accumulated photobleaching."""
        for mode in [1, 2]:
            if mode in self._pipeline:
                self._pipeline[mode].reset_bleach()

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()

        # Step at start of each snap cycle
        if (self.auto_step and self._snap_count > 0
                and self._snap_count % self.snaps_per_step == 0):
            self.step()

        self._snap_count += 1

        # Re-render if cell geometry changed (stretch, drug)
        if self._bf_dirty:
            self._bf_full = self._render_bf_full()
            self._nuc_full = self._render_nuc_full()
            self._bf_dirty = False
        if self._actin_dirty:
            self._actin_full = self._render_actin_full()
            self._actin_dirty = False

        # Select pre-rendered image
        if self.mode == 0:
            full_img = self._bf_full
        elif self.mode == 1:
            full_img = self._nuc_full
        elif self.mode == 2:
            full_img = self._actin_full
        elif self.mode in self._extra_channels:
            full_img = self._extra_channels[self.mode]["image"]
        else:
            full_img = self._bf_full

        # Crop FOV from internal-resolution image
        viewport = self._crop_fov(full_img)

        # Defocus
        viewport = self._apply_defocus(viewport)

        # Optical pipeline (PSF, noise, vignetting, optional bleaching)
        if self.mode in self._pipeline:
            pipe = self._pipeline[self.mode]
            if self.mode > 0 and pipe.photobleach_rate > 0:
                viewport = pipe.apply_with_bleach(viewport, exposure_ms=exposure)
            else:
                viewport = pipe.apply(viewport, exposure_ms=exposure)

        # Exposure scaling (BF uses 2× base to keep transmitted light at full contrast)
        if self.mode == 0:
            scale = min(intensity * 0.02 * exposure, 2.0)
        else:
            scale = intensity * 0.01 * exposure
        viewport = (viewport.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)

        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def apply_drug(self, drug_name: str, onset_rate: float = None):
        """Apply an actin-disrupting drug.

        Supported drugs:
          - cytochalasin_d: F-actin capping, severe fiber/FA loss, cell rounding
          - latrunculin_a: G-actin sequestering, severe fiber/FA loss
          - y27632: ROCK inhibitor, moderate fiber reduction, slight shrinkage
        """
        drug_name = drug_name.lower().replace("-", "_").replace(" ", "_")
        if drug_name not in self._drug_profiles:
            raise ValueError(f"Unknown drug: {drug_name}")
        profile = self._drug_profiles[drug_name]
        self._drug_active = True
        self._drug_name = drug_name
        self._drug_washing_out = False
        if onset_rate is not None:
            # Use custom rate but keep rest of profile
            self._drug_onset_rate = onset_rate
        else:
            self._drug_onset_rate = profile["onset_rate"]

    def remove_drug(self, washout_rate: float = None):
        """Remove the current drug (begin washout)."""
        if not self._drug_active:
            return
        self._drug_washing_out = True
        self._drug_active = False
        profile = self._drug_profiles.get(self._drug_name, {})
        self._drug_washout_rate = washout_rate or profile.get("washout_rate", 0.04)

    def _update_drug_effect(self, effective_dt: float = 1.0):
        """Update drug effect level (exponential approach to target or zero).

        effective_dt incorporates temperature scaling — drug kinetics
        slow down at low temperatures.
        """
        if self._drug_active:
            profile = self._drug_profiles[self._drug_name]
            target = profile["target_effect"]
            rate = self._drug_onset_rate * effective_dt
            self._drug_effect += (target - self._drug_effect) * rate
        elif self._drug_washing_out:
            rate = self._drug_washout_rate * effective_dt
            self._drug_effect -= self._drug_effect * rate
            if self._drug_effect < 0.01:
                self._drug_effect = 0.0
                self._drug_washing_out = False
                self._drug_name = None

    # ── Temperature response ──

    def _get_temperature(self) -> float:
        """Read current temperature (°C) from the controller device."""
        if "Temperature" not in self.state_devices:
            return 37.0  # mammalian default
        return float(self.state_devices["Temperature"].get("label", "37"))

    def _temperature_factor(self) -> float:
        """Kinetics multiplier — Q10=2.0, reference 37°C.

        Fibroblasts are mammalian cells: 37°C optimal.
        Below 10°C: near-arrest.  Above 42°C: heat shock.
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.05  # cold arrest
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.1, 1.0 - (temp - 42) * 0.3)
        return max(0.01, min(factor, 4.0))

    def _update_cold_actin(self, dt: float):
        """Cold-induced actin depolymerization.

        Below ~15°C, actin filaments gradually depolymerize — this is a
        real biological phenomenon used in cold-shock experiments.
        Recovery occurs when warmed back.
        """
        temp = self._get_temperature()
        old_loss = self._cold_actin_loss
        if temp < 15:
            # Depolymerization rate increases with coldness
            severity = (15 - temp) / 15.0  # 0 at 15°C, 1 at 0°C
            max_loss = min(0.95, severity)  # 4°C → 0.73, 10°C → 0.33
            rate = 0.03 * severity * dt
            self._cold_actin_loss = min(max_loss,
                                        self._cold_actin_loss + rate)
        elif self._cold_actin_loss > 0:
            # Recovery at warm temperature (slower than loss)
            rate = 0.015 * dt
            self._cold_actin_loss = max(0, self._cold_actin_loss - rate)

        if abs(self._cold_actin_loss - old_loss) > 0.001:
            self._actin_dirty = True

    # ── Mechanical stretch response ──

    def _read_stretch(self):
        """Read stretch strain from the Stretch device."""
        stretch_state = self.state_devices.get("Stretch", {})
        label = stretch_state.get("label", stretch_state.get("Label", "Off"))
        if label == "Off":
            self._stretch_strain = 0.0
        else:
            try:
                self._stretch_strain = float(label.replace("%", "")) / 100.0
            except (ValueError, AttributeError):
                self._stretch_strain = 0.0

    def _update_stretch_reorientation(self, dt: float):
        """Reorient cells perpendicular to stretch axis.

        Real fibroblasts align perpendicular to uniaxial cyclic stretch
        over 2-6 hours. Rate scales with strain magnitude.
        The target angle is 90 deg from the stretch direction (horizontal).
        """
        if self._stretch_strain <= 0:
            return

        target_perp = self._stretch_direction + math.pi / 2
        # Reorientation rate: ~0.02 rad/step at 10% strain
        rate = 0.02 * (self._stretch_strain / 0.10) * dt
        any_changed = False

        for cell in self._cells:
            angle = cell["angle"]
            # Angle difference to perpendicular (cells have 180 deg symmetry)
            diff = angle - target_perp
            diff = math.atan2(math.sin(2 * diff), math.cos(2 * diff)) / 2
            if abs(diff) < 0.02:
                continue

            step_angle = -diff * rate + self.rng.normal(0, 0.005)
            cell["angle"] += step_angle
            cell["angle"] = cell["angle"] % math.pi
            any_changed = True

            # Stress fibers also realign (reduce deviation from cell axis)
            for fiber in cell["fibers"]:
                fiber["angle_dev"] *= (1 - rate * 0.3)

        if any_changed:
            self._actin_dirty = True
            self._bf_dirty = True

    def step(self, dt=None):
        """Advance simulation by one time step."""
        dt = dt or (self.fixed_dt if self.fixed_dt else 1.0)

        # Read devices
        self._read_stretch()

        # Z-drift accumulation
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        # Temperature-scaled effective dt for kinetics
        tfactor = self._temperature_factor()
        effective_dt = dt * tfactor

        self._time += dt
        self._step_count += 1

        # Cold-induced actin depolymerization
        self._update_cold_actin(dt)

        # Mechanical stretch reorientation
        self._update_stretch_reorientation(effective_dt)

        if self._drug_active or self._drug_washing_out:
            old_effect = self._drug_effect
            self._update_drug_effect(effective_dt)
            if abs(self._drug_effect - old_effect) > 0.001:
                self._actin_dirty = True

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (no SLM effects)."""
        self.step(dt)

    # ----------------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------------

    def get_ground_truth(self) -> dict:
        """Return ground truth for all cells."""
        cells = []
        for i, cell in enumerate(self._cells):
            # Count fiber subtypes
            fiber_counts = {t: 0 for t in
                            ("ventral", "dorsal", "perinuclear_cap")}
            arc_count = 0
            for f in cell["fibers"]:
                ft = f.get("type", "ventral")
                if ft in fiber_counts:
                    fiber_counts[ft] += 1
            arc_count = len(cell.get("arcs", []))
            cells.append({
                "id": i,
                "cx": round(cell["cx"], 1),
                "cy": round(cell["cy"], 1),
                "angle_deg": round(math.degrees(cell["angle"]), 1),
                "length": round(cell["length"], 1),
                "width": round(cell["width"], 1),
                "n_fibers": len(cell["fibers"]),
                "n_ventral": fiber_counts["ventral"],
                "n_dorsal": fiber_counts["dorsal"],
                "n_perinuclear_cap": fiber_counts["perinuclear_cap"],
                "n_transverse_arcs": arc_count,
                "n_focal_adhesions": len(cell["focal_adhesions"]),
                "has_lamellipodium": cell["has_lamellipodium"],
            })
        gt = {
            "n_cells": len(self._cells),
            "cells": cells,
        }
        if self._drug_name:
            profile = self._drug_profiles[self._drug_name]
            gt["drug"] = {
                "name": self._drug_name,
                "effect": round(self._drug_effect, 3),
                "active": self._drug_active,
                "washing_out": self._drug_washing_out,
                "fiber_brightness_mult": round(
                    1.0 - self._drug_effect * profile.get("fiber_dimming", 0), 3),
                "fa_brightness_mult": round(
                    1.0 - self._drug_effect * profile.get("fa_dimming", 0), 3),
            }
        gt["temperature"] = {
            "celsius": self._get_temperature(),
            "kinetics_factor": round(self._temperature_factor(), 3),
            "cold_actin_loss": round(self._cold_actin_loss, 3),
        }
        if self._stretch_strain > 0:
            angles = [math.degrees(c["angle"]) for c in self._cells]
            mean_angle = float(np.mean(angles))
            # Circular std dev (for orientation data with 180° symmetry)
            doubled = [2 * math.radians(a) for a in angles]
            R = abs(sum(complex(math.cos(d), math.sin(d)) for d in doubled)) / len(doubled)
            circ_std = math.degrees(math.sqrt(-2 * math.log(max(R, 1e-6)))) / 2
            gt["stretch"] = {
                "strain_pct": round(self._stretch_strain * 100, 1),
                "direction_deg": round(math.degrees(self._stretch_direction), 1),
                "mean_cell_angle_deg": round(mean_angle, 1),
                "cell_angle_std_deg": round(circ_std, 1),
            }
        return gt

    def get_fiber_type_stats(self) -> dict:
        """Return aggregate fiber type statistics across all cells.

        Returns counts and per-cell means for ventral, dorsal,
        perinuclear_cap, and transverse_arc subtypes.
        """
        totals = {t: 0 for t in
                  ("ventral", "dorsal", "perinuclear_cap", "transverse_arc")}
        for cell in self._cells:
            for f in cell.get("fibers", []):
                ft = f.get("type", "ventral")
                totals[ft] = totals.get(ft, 0) + 1
            for a in cell.get("arcs", []):
                totals["transverse_arc"] += 1
        n = max(len(self._cells), 1)
        return {
            "n_cells": len(self._cells),
            "total_ventral": totals["ventral"],
            "total_dorsal": totals["dorsal"],
            "total_perinuclear_cap": totals["perinuclear_cap"],
            "total_transverse_arc": totals["transverse_arc"],
            "mean_ventral_per_cell": round(totals["ventral"] / n, 2),
            "mean_dorsal_per_cell": round(totals["dorsal"] / n, 2),
            "mean_perinuclear_cap_per_cell": round(totals["perinuclear_cap"] / n, 2),
            "mean_transverse_arc_per_cell": round(totals["transverse_arc"] / n, 2),
            "dominant_type": max(totals, key=totals.get),
        }

    def get_cell_positions(self) -> list:
        """Return cell center positions as [[x,y], ...]."""
        return [[c["cx"], c["cy"]] for c in self._cells]

    def get_cell_orientations(self) -> list:
        """Return cell orientations in degrees."""
        return [math.degrees(c["angle"]) for c in self._cells]

    def add_channel(self, name: str, image: np.ndarray, mode_id: int):
        """Add an extra rendering channel."""
        self._extra_channels[mode_id] = {"name": name, "image": image}

    def reset(self, seed: int = None):
        """Reset simulation state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._noise_rng = np.random.default_rng(seed + 5555)
        self._snap_count = 0
        self.camera_offset = [0, 0]
        self.focal_plane = 0.0
        self.tissue_z = 0.0
