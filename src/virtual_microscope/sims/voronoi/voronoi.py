"""
VoronoiSim — drop-in replacement for MicroscopeSimOptmized.

Implements the same interface so it works with SimulationBridge and
all existing pymmcore devices without any changes.

Usage via SimulationBridge:
    sim = VoronoiSim(width=1024, height=1024, n_cells=200, seed=42)
    bridge = SimulationBridge(sim)  # VoronoiSim quacks like MicroscopeSimOptmized
"""

import time
import numpy as np
import cv2
from scipy.spatial import Voronoi
from virtual_microscope.optical_pipeline import OpticalPipeline
from virtual_microscope.nuclear_texture import render_textured_nuclei


class VoronoiSim:
    """Confluent tissue simulation compatible with SimulationBridge.

    Generates a static Voronoi-tessellated tissue that supports:
    - Stage position (camera_offset) for viewport panning
    - Objective switching (10x/20x/40x with crop and rescale)
    - Channel switching (brightfield / nucleus / membrane)
    - Exposure and intensity scaling
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        nb_cells: int = 200,
        cell_type: str = "voronoi",
        viewport_width: int = 512,
        viewport_height: int = 512,
        base_radius: float = 20.0,
        rng_seed: int = 42,
        jitter: float = 0.7,
        nucleus_fraction: float = 0.3,
        textured_nuclei: bool = False,
        membrane_ruffle: float = 0.0,
        internal_scale: int = 4,
        **kwargs,
    ):
        self.width = width
        self.height = height
        self.internal_scale = internal_scale
        self._iw = width * internal_scale
        self._ih = height * internal_scale
        self.nb_cells = nb_cells
        self.cell_type = cell_type
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.base_radius = base_radius
        self.rng_seed = rng_seed
        self.textured_nuclei = textured_nuclei
        self.membrane_ruffle = membrane_ruffle

        # Camera settings (same interface as MicroscopeSimOptmized)
        self.camera_offset = np.array([0.0, 0.0])
        self.focal_plane = 0.0

        # State tracking (populated by setup_microscope via bridge.update_state)
        self.state_devices = {}
        self.mode = 0  # 0=BF, 1=nucleus, 2=membrane
        self._last_time = time.perf_counter()
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40, "100x": 100}
        self.current_objectiv = 10

        # Generate tissue
        self.rng = np.random.default_rng(rng_seed)
        self.nucleus_fraction = nucleus_fraction

        self.centers = self._generate_centers(jitter)
        self._compute_voronoi()
        self._compute_cell_properties()
        self._compute_ruffled_polygons()

        # Per-cell marker flags
        self.has_nucleus_marker = np.ones(self.nb_cells, dtype=bool)
        self.has_membrane_marker = np.ones(self.nb_cells, dtype=bool)
        self.has_cytoplasm_marker = np.zeros(self.nb_cells, dtype=bool)
        self.nucleus_intensity = self.rng.uniform(0.7, 1.0, self.nb_cells)
        self.membrane_intensity = self.rng.uniform(0.6, 0.9, self.nb_cells)
        self.cytoplasm_intensity = np.zeros(self.nb_cells, dtype=float)
        self.cell_gray = self.rng.uniform(100, 155, self.nb_cells).astype(np.uint8)

        # Per-cell nucleus morphology (stable across recomputes)
        # Offset: fraction of nucleus_radii in random direction
        self._nuc_offset_frac = self.rng.uniform(0.0, 0.3, self.nb_cells)
        self._nuc_offset_angle = self.rng.uniform(0, 2 * np.pi, self.nb_cells)
        # Ellipticity: axis ratio and orientation
        self._nuc_aspect = self.rng.uniform(1.0, 1.3, self.nb_cells)
        self._nuc_orient = self.rng.uniform(0, np.pi, self.nb_cells)

        # Per-cell render visibility (False = ghost cell, skipped in all channels)
        self._renderable = np.ones(self.nb_cells, dtype=bool)

        # Optical pipelines per channel (standalone degradation)
        self._bf_pipeline = OpticalPipeline(
            psf_sigma=0, noise={"photon_scale": 5.0, "read_std": 3.0},
            vignette=0.10, rng_seed=rng_seed + 1000,
        )
        self._nuc_pipeline = OpticalPipeline(
            psf_sigma=1.0, noise={"photon_scale": 3.0, "read_std": 2.5},
            rng_seed=rng_seed + 2000,
        )
        self._mem_pipeline = OpticalPipeline(
            psf_sigma=0.8, noise={"photon_scale": 3.0, "read_std": 2.5},
            rng_seed=rng_seed + 3000,
        )

        # Live pipeline mode: re-apply pipeline on each snap (for photobleaching)
        self.live_pipeline = False
        self._skip_pipeline = False
        self._bf_clean = None
        self._nuc_clean = None
        self._mem_clean = None
        self._snap_count = 0  # counts snap_frame calls (for live pipeline)

        # Spectral crosstalk between fluorescence channels
        # e.g., {"nuc_to_mem": 0.05, "mem_to_nuc": 0.08}
        self.crosstalk = None

        # Channel wavelengths (nm) for chromatic aberration
        # Set these to enable wavelength-dependent rendering
        self.nuc_wavelength = 0   # e.g., 461 for DAPI
        self.mem_wavelength = 0   # e.g., 580 for RFP

        # Extra channels (beyond BF/nucleus/membrane)
        self._extra_channels = {}
        self._next_mode = 3  # modes 0,1,2 are BF/nuc/mem

        # Data-driven channel→mode mapping (filter_label, led_label) → mode_id
        # Subclasses / enable_*() can override this dict.
        self._mode_map = {
            ("SCFP2(434/474)", "UV"): 1,           # DAPI
            ("mScarlet3(569/582)", "ORANGE"): 2,   # membrane
        }

        # Pre-render full tissue images for each mode (cache for performance)
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None
        self._render_full_tissue()

        # Z-stack / defocus support
        self.tissue_z = 0.0  # Z position of the monolayer (µm)
        self._dof = 6.0  # depth of field (µm), updated per objective
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.6}
        # Blur scale: pixels of sigma per µm of defocus beyond DOF/2
        self._blur_scale_table = {10: 0.5, 20: 1.0, 40: 2.5, 100: 5.0}
        # Per-cell Z-stack support (None = all at tissue_z, no Z-stack)
        self.nucleus_z = None  # per-cell Z positions (µm)
        self._nuc_z_layers = None  # dict: layer_idx -> pre-rendered image
        self._z_layer_positions = None  # array of Z values per layer

        # Stage drift simulation (slow XY drift over time)
        self.stage_drift_rate = 0.0     # px per snap_frame call
        self.stage_drift_angle = 0.0    # drift direction in radians
        self.stage_drift_noise = 0.0    # random walk component (px per snap)
        self._drift_accumulator = np.array([0.0, 0.0])
        self._drift_rng = np.random.default_rng(rng_seed + 7777)

        # Dummy _cells list (empty — no particle cells)
        self._cells = []

    def _s(self, v):
        """Scale world coordinate to internal resolution (int)."""
        return int(round(v * self.internal_scale))

    def _sf(self, v):
        """Scale world coordinate to internal resolution (float)."""
        return v * self.internal_scale

    def _nuc_center(self, i: int) -> tuple:
        """Nucleus center in world coords (offset from cell centroid)."""
        cx, cy = self.cell_centroids[i]
        r = self.nucleus_radii[i]
        off = self._nuc_offset_frac[i] * r
        ang = self._nuc_offset_angle[i]
        return cx + off * np.cos(ang), cy + off * np.sin(ang)

    def _draw_nucleus(self, img, i: int, color, thickness=-1):
        """Draw nucleus as ellipse with per-cell offset, aspect, orientation."""
        s = self.internal_scale
        ncx, ncy = self._nuc_center(i)
        ix, iy = self._s(ncx), self._s(ncy)
        r = max(3 * s, self._s(self.nucleus_radii[i]))
        asp = self._nuc_aspect[i]
        angle_deg = np.degrees(self._nuc_orient[i])
        axes = (int(round(r * asp)), r)
        if isinstance(color, tuple):
            cv2.ellipse(img, (ix, iy), axes, angle_deg, 0, 360,
                        color, thickness, cv2.LINE_AA)
        else:
            cv2.ellipse(img, (ix, iy), axes, angle_deg, 0, 360,
                        (color, color, color), thickness, cv2.LINE_AA)

    # ---- Cell center generation ----

    def _generate_centers(self, jitter: float) -> np.ndarray:
        """Generate cell centers using jittered hex grid."""
        area_per_cell = (self.width * self.height) / self.nb_cells
        spacing = np.sqrt(area_per_cell / 0.866)

        centers = []
        ny = int(self.height / (spacing * 0.866)) + 2
        nx = int(self.width / spacing) + 2
        for row in range(ny):
            for col in range(nx):
                x = col * spacing + (0.5 * spacing if row % 2 else 0)
                y = row * spacing * 0.866
                x -= spacing * 0.5
                y -= spacing * 0.5
                centers.append([x, y])

        centers = np.array(centers)

        if jitter > 0:
            noise = self.rng.normal(0, spacing * 0.25 * jitter, centers.shape)
            centers += noise

        # Keep centers within bounds so Voronoi polygons aren't degenerate
        margin = max(5, spacing * 0.1)
        mask = (
            (centers[:, 0] >= margin)
            & (centers[:, 0] <= self.width - margin)
            & (centers[:, 1] >= margin)
            & (centers[:, 1] <= self.height - margin)
        )
        centers = centers[mask]

        if len(centers) > self.nb_cells:
            idx = self.rng.choice(len(centers), self.nb_cells, replace=False)
            centers = centers[idx]
        elif len(centers) < self.nb_cells:
            extra = self.nb_cells - len(centers)
            new_pts = self.rng.uniform(
                [10, 10], [self.width - 10, self.height - 10], (extra, 2)
            )
            centers = np.vstack([centers, new_pts])

        return centers[: self.nb_cells]

    def _compute_voronoi(self):
        """Compute Voronoi with mirrored boundary for clean edges."""
        pts = self.centers.copy()
        n = len(pts)

        mirror = []
        mirror.append(np.column_stack([-pts[:, 0], pts[:, 1]]))
        mirror.append(np.column_stack([2 * self.width - pts[:, 0], pts[:, 1]]))
        mirror.append(np.column_stack([pts[:, 0], -pts[:, 1]]))
        mirror.append(np.column_stack([pts[:, 0], 2 * self.height - pts[:, 1]]))

        all_pts = np.vstack([pts] + mirror)
        self.vor = Voronoi(all_pts)

        self.cell_regions = []
        self.cell_polygons = []
        for i in range(n):
            region_idx = self.vor.point_region[i]
            region = self.vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                verts = self.vor.vertices[region].copy()
                verts[:, 0] = np.clip(verts[:, 0], 0, self.width - 1)
                verts[:, 1] = np.clip(verts[:, 1], 0, self.height - 1)
                self.cell_polygons.append(verts)
            else:
                # Fallback: small hexagon at center
                cx, cy = self.centers[i]
                r = 10
                angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
                poly = np.column_stack(
                    [cx + r * np.cos(angles), cy + r * np.sin(angles)]
                )
                self.cell_polygons.append(poly)

        # Pad if fewer valid regions than cells
        while len(self.cell_polygons) < self.nb_cells:
            idx = len(self.cell_polygons)
            cx, cy = self.centers[idx]
            r = 10
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            poly = np.column_stack(
                [cx + r * np.cos(angles), cy + r * np.sin(angles)]
            )
            self.cell_polygons.append(poly)

    def get_neighbor_graph(self):
        """Return cell adjacency from Voronoi ridges.

        Returns:
            dict mapping cell index to set of neighbor indices.
            Only includes original cells (not mirror copies).
        """
        n = self.nb_cells
        neighbors = {i: set() for i in range(n)}
        for p1, p2 in self.vor.ridge_points:
            if p1 < n and p2 < n:
                neighbors[p1].add(p2)
                neighbors[p2].add(p1)
        return neighbors

    def get_contact_lengths(self):
        """Return shared boundary lengths between adjacent cells.

        Returns:
            list of (cell_i, cell_j, length_px) tuples for each Voronoi ridge
            between original (non-mirror) cells.
        """
        n = self.nb_cells
        contacts = []
        for ridge_idx, (p1, p2) in enumerate(self.vor.ridge_points):
            if p1 >= n or p2 >= n:
                continue
            verts = self.vor.ridge_vertices[ridge_idx]
            if -1 in verts:
                continue
            v1 = self.vor.vertices[verts[0]]
            v2 = self.vor.vertices[verts[1]]
            length = float(np.hypot(v2[0] - v1[0], v2[1] - v1[1]))
            contacts.append((int(p1), int(p2), round(length, 1)))
        return contacts

    def _compute_cell_properties(self):
        """Compute areas, centroids, nucleus radii."""
        self.cell_areas = np.zeros(self.nb_cells)
        self.cell_centroids = np.zeros((self.nb_cells, 2))
        self.nucleus_radii = np.zeros(self.nb_cells)

        for i, poly in enumerate(self.cell_polygons[: self.nb_cells]):
            # Area (shoelace)
            n = len(poly)
            if n < 3:
                self.cell_areas[i] = 100.0
                self.cell_centroids[i] = self.centers[i]
                self.nucleus_radii[i] = 5.0
                continue
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * abs(
                np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                + x[-1] * y[0]
                - x[0] * y[-1]
            )
            self.cell_areas[i] = max(area, 1.0)
            self.cell_centroids[i] = poly.mean(axis=0)
            self.nucleus_radii[i] = np.sqrt(area / np.pi) * self.nucleus_fraction

    def _compute_ruffled_polygons(self):
        """Create ruffled versions of cell polygons for realistic rendering.

        Subdivides each edge and adds perpendicular noise to create wavy,
        organic-looking cell boundaries. The original cell_polygons are
        kept for GT calculations; render_polygons are used for drawing.
        """
        if self.membrane_ruffle <= 0:
            self.render_polygons = self.cell_polygons
            return

        self.render_polygons = []
        for i, poly in enumerate(self.cell_polygons[:self.nb_cells]):
            if len(poly) < 3:
                self.render_polygons.append(poly)
                continue
            rng = np.random.default_rng(self.rng_seed + i * 1000 + 7777)
            ruffled = []
            n_verts = len(poly)
            subdivisions = 4  # each edge gets 4 sub-segments
            for j in range(n_verts):
                v1 = poly[j]
                v2 = poly[(j + 1) % n_verts]
                edge = v2 - v1
                edge_len = np.linalg.norm(edge)
                if edge_len < 2:
                    ruffled.append(v1)
                    continue
                perp = np.array([-edge[1], edge[0]])
                perp /= edge_len  # unit perpendicular
                ruffled.append(v1)
                for k in range(1, subdivisions):
                    t = k / subdivisions
                    mid = v1 + t * edge
                    offset = rng.normal(0, self.membrane_ruffle) * perp
                    mid = mid + offset
                    # Clip to image bounds
                    mid[0] = np.clip(mid[0], 0, self.width - 1)
                    mid[1] = np.clip(mid[1], 0, self.height - 1)
                    ruffled.append(mid)
            self.render_polygons.append(np.array(ruffled))

        # Pad if needed
        while len(self.render_polygons) < self.nb_cells:
            idx = len(self.render_polygons)
            if idx < len(self.cell_polygons):
                self.render_polygons.append(self.cell_polygons[idx])
            else:
                self.render_polygons.append(self.cell_polygons[-1])

    def _make_scaled_proxy(self):
        """Create a proxy object with scaled coordinates for external renderers."""
        s = self.internal_scale

        class Proxy:
            pass

        p = Proxy()
        p.nb_cells = self.nb_cells
        p.width = self._iw
        p.height = self._ih
        p.cell_polygons = [(poly * s) for poly in self.cell_polygons[:self.nb_cells]]
        p.cell_centroids = self.cell_centroids * s
        p.nucleus_radii = self.nucleus_radii * s
        p.has_nucleus_marker = self.has_nucleus_marker
        p.nucleus_intensity = self.nucleus_intensity
        # Pass nucleus morphology for textured rendering
        p._nuc_offset_frac = self._nuc_offset_frac
        p._nuc_offset_angle = self._nuc_offset_angle
        p._nuc_aspect = self._nuc_aspect
        p._nuc_orient = self._nuc_orient
        return p

    # ---- Full tissue rendering ----

    def _render_full_tissue(self):
        """Pre-render full tissue images for all 3 modes.

        In live_pipeline mode, stores clean (pre-pipeline) versions and applies
        pipeline dynamically during snap_frame() for photobleaching support.
        """
        if self.live_pipeline:
            # Render without pipeline, store clean versions
            self._skip_pipeline = True
            self._bf_clean = self._render_bf_full()
            self._nuc_clean = self._render_nuc_full()
            self._mem_clean = self._render_mem_full()
            self._skip_pipeline = False
            # Apply spectral crosstalk to clean images
            self._apply_crosstalk_clean()
            # Apply static pipeline for initial display
            self._bf_full = self._bf_pipeline.apply(self._bf_clean.copy())
            self._nuc_full = self._nuc_pipeline.apply(
                self._nuc_clean.copy(), wavelength_nm=self.nuc_wavelength)
            self._mem_full = self._mem_pipeline.apply(
                self._mem_clean.copy(), wavelength_nm=self.mem_wavelength)
        else:
            # BF renders with its pipeline (no crosstalk applies to BF)
            self._bf_full = self._render_bf_full()
            if self.crosstalk:
                # Render clean, apply crosstalk, then pipeline
                self._skip_pipeline = True
                nuc_clean = self._render_nuc_full()
                mem_clean = self._render_mem_full()
                self._skip_pipeline = False
                nuc_clean, mem_clean = self._mix_crosstalk(
                    nuc_clean, mem_clean, self.crosstalk)
                self._nuc_full = self._nuc_pipeline.apply(
                    nuc_clean, wavelength_nm=self.nuc_wavelength)
                self._mem_full = self._mem_pipeline.apply(
                    mem_clean, wavelength_nm=self.mem_wavelength)
            else:
                # Standard path: render with pipeline baked in
                self._nuc_full = self._render_nuc_full()
                self._mem_full = self._render_mem_full()

    def _apply_crosstalk_clean(self):
        """Apply spectral crosstalk to clean (pre-pipeline) images."""
        if not self.crosstalk or self._nuc_clean is None or self._mem_clean is None:
            return
        self._nuc_clean, self._mem_clean = self._mix_crosstalk(
            self._nuc_clean, self._mem_clean, self.crosstalk)

    @staticmethod
    def _mix_crosstalk(nuc_img: np.ndarray, mem_img: np.ndarray,
                       crosstalk: dict = None) -> tuple:
        """Mix spectral crosstalk between nucleus and membrane channels.

        Args:
            nuc_img: Clean nucleus channel image
            mem_img: Clean membrane channel image
            crosstalk: Dict with keys nuc_to_mem, mem_to_nuc (fractions 0-1)

        Returns:
            (nuc_with_crosstalk, mem_with_crosstalk)
        """
        if crosstalk is None:
            return nuc_img, mem_img

        nuc_f = nuc_img.astype(np.float32)
        mem_f = mem_img.astype(np.float32)

        nuc_leak = crosstalk.get("nuc_to_mem", 0)
        mem_leak = crosstalk.get("mem_to_nuc", 0)

        # Apply crosstalk: each channel gets a fraction of the other
        new_nuc = nuc_f + mem_f * mem_leak
        new_mem = mem_f + nuc_f * nuc_leak

        return (np.clip(new_nuc, 0, 255).astype(np.uint8),
                np.clip(new_mem, 0, 255).astype(np.uint8))

    def _get_apoptosis_stage(self, i: int) -> int:
        """Get apoptosis stage for cell i (0 if no apoptosis tracking)."""
        apo = getattr(self, 'apoptosis_stage', None)
        if apo is not None and i < len(apo):
            return int(apo[i])
        return 0

    def _render_bf_full(self) -> np.ndarray:
        """Render full brightfield/phase-contrast image at internal resolution."""
        s = self.internal_scale
        lt = max(1, s // 2)
        img = np.full((self._ih, self._iw, 3), 128, dtype=np.uint8)
        renderable = getattr(self, '_renderable', None)

        for i, poly in enumerate(self.render_polygons[: self.nb_cells]):
            if len(poly) < 3 or (renderable is not None and not renderable[i]):
                continue
            stage = self._get_apoptosis_stage(i)
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            gray = int(self.cell_gray[i])
            if stage >= 2:
                gray = min(128, gray + 15 * stage)
            cv2.fillPoly(img, [pts], (gray, gray, gray))

            if stage >= 3:
                cx, cy = self._s(self.cell_centroids[i][0]), self._s(self.cell_centroids[i][1])
                cell_r = max(8 * s, self._s(self.nucleus_radii[i] * 2.5))
                cv2.fillPoly(img, [pts], (128, 128, 128))
                cv2.circle(img, (cx, cy), cell_r,
                           (gray, gray, gray), -1, cv2.LINE_AA)

        # Phase-contrast halo
        for i, poly in enumerate(self.render_polygons[: self.nb_cells]):
            if len(poly) < 3 or (renderable is not None and not renderable[i]):
                continue
            stage = self._get_apoptosis_stage(i)
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            halo_val = 155 + min(40, 15 * stage)
            if stage >= 3:
                cx, cy = self._s(self.cell_centroids[i][0]), self._s(self.cell_centroids[i][1])
                cell_r = max(8 * s, self._s(self.nucleus_radii[i] * 2.5))
                cv2.circle(img, (cx, cy), cell_r + 2 * s,
                           (halo_val, halo_val, halo_val), max(4, 4 * lt), cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], True, (halo_val, halo_val, halo_val),
                              thickness=max(4, 4 * lt), lineType=cv2.LINE_AA)

        # Junctions
        for i, poly in enumerate(self.render_polygons[: self.nb_cells]):
            if len(poly) < 3 or (renderable is not None and not renderable[i]):
                continue
            stage = self._get_apoptosis_stage(i)
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            if stage >= 3:
                cx, cy = self._s(self.cell_centroids[i][0]), self._s(self.cell_centroids[i][1])
                cell_r = max(8 * s, self._s(self.nucleus_radii[i] * 2.5))
                cv2.circle(img, (cx, cy), cell_r,
                           (40, 40, 40), max(2, lt), cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], True, (40, 40, 40), thickness=max(2, lt),
                              lineType=cv2.LINE_AA)

        # Nuclei (ellipsoidal, offset from centroid)
        for i in range(self.nb_cells):
            if renderable is not None and not renderable[i]:
                continue
            ncx, ncy = self._nuc_center(i)
            if 0 <= ncx < self.width and 0 <= ncy < self.height:
                self._draw_nucleus(img, i, (60, 60, 60), -1)
                # Brighter inner region
                r = max(3 * s, self._s(self.nucleus_radii[i]))
                ix, iy = self._s(ncx), self._s(ncy)
                inner_r = max(1, r // 2)
                cv2.circle(img, (ix, iy), inner_r, (50, 50, 50), -1, cv2.LINE_AA)

        if not self._skip_pipeline:
            img = self._bf_pipeline.apply(img)
        return img

    def _render_nuc_full(self) -> np.ndarray:
        """Render full nuclear fluorescence image at internal resolution."""
        s = self.internal_scale
        img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)

        renderable = getattr(self, '_renderable', None)

        if self.textured_nuclei:
            # Textured nuclei renders into img using sim coordinates
            # Pass scaled coordinates via a proxy
            render_textured_nuclei(
                img, self._make_scaled_proxy(),
                positive=self.has_nucleus_marker,
                intensities=self.nucleus_intensity,
                rng=np.random.default_rng(self.rng_seed + 5000),
                renderable=renderable,
            )
        else:
            for i, poly in enumerate(self.render_polygons[: self.nb_cells]):
                if len(poly) < 3 or (renderable is not None and not renderable[i]):
                    continue
                pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
                auto_level = int(self.rng.uniform(5, 12))
                cv2.fillPoly(img, [pts], (auto_level, auto_level, auto_level))

            for i in range(self.nb_cells):
                if renderable is not None and not renderable[i]:
                    continue
                if not self.has_nucleus_marker[i]:
                    continue
                ncx, ncy = self._nuc_center(i)
                if 0 <= ncx < self.width and 0 <= ncy < self.height:
                    intensity = int(self.nucleus_intensity[i] * 200)
                    self._draw_nucleus(img, i, (intensity, intensity, intensity), -1)
                    # Brighter inner region
                    r = max(3 * s, self._s(self.nucleus_radii[i]))
                    ix, iy = self._s(ncx), self._s(ncy)
                    bright = min(255, int(intensity * 1.2))
                    cv2.circle(img, (ix, iy), max(1, r // 2),
                               (bright, bright, bright), -1, cv2.LINE_AA)

        if not self._skip_pipeline:
            img = self._nuc_pipeline.apply(img, wavelength_nm=self.nuc_wavelength)
        return img

    def _render_mem_full(self) -> np.ndarray:
        """Render full membrane fluorescence image at internal resolution."""
        s = self.internal_scale
        lt = max(1, s // 2)
        img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)
        renderable = getattr(self, '_renderable', None)

        # Autofluorescence + cytoplasmic E-cadherin (ER/Golgi pool)
        for i, poly in enumerate(self.render_polygons[: self.nb_cells]):
            if len(poly) < 3 or (renderable is not None and not renderable[i]):
                continue
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            # Cells with membrane marker have faint cytoplasmic E-cadherin
            if self.has_membrane_marker[i]:
                auto_level = int(self.rng.uniform(10, 20))
            else:
                auto_level = int(self.rng.uniform(3, 8))
            cv2.fillPoly(img, [pts], (auto_level, auto_level, auto_level))

        # Base membrane signal
        for i, poly in enumerate(self.render_polygons[: self.nb_cells]):
            if len(poly) < 3 or not self.has_membrane_marker[i]:
                continue
            if renderable is not None and not renderable[i]:
                continue
            stage = self._get_apoptosis_stage(i)
            intensity = int(self.membrane_intensity[i] * 140)
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)

            if stage >= 3:
                cx, cy = self.cell_centroids[i]
                bleb_r = max(8 * s, self._s(self.nucleus_radii[i] * 2.0))
                n_blebs = 5 + stage * 2
                bleb_intensity = max(30, intensity // 2)
                rng = np.random.default_rng(self.rng_seed + i * 1000 + stage)
                for _ in range(n_blebs):
                    angle = rng.uniform(0, 2 * np.pi)
                    dist = rng.uniform(bleb_r * 0.5, bleb_r * 1.3)
                    bx = int(self._sf(cx) + np.cos(angle) * dist)
                    by = int(self._sf(cy) + np.sin(angle) * dist)
                    br = rng.integers(2, 5) * s
                    if 0 <= bx < self._iw and 0 <= by < self._ih:
                        cv2.circle(img, (bx, by), br,
                                   (bleb_intensity, bleb_intensity, bleb_intensity),
                                   lt, cv2.LINE_AA)
            elif stage >= 2:
                dim_intensity = max(20, intensity // 2)
                cv2.polylines(img, [pts], True,
                              (dim_intensity, dim_intensity, dim_intensity),
                              thickness=max(2, lt), lineType=cv2.LINE_AA)
            else:
                cv2.polylines(img, [pts], True,
                              (intensity, intensity, intensity),
                              thickness=max(2, lt), lineType=cv2.LINE_AA)

        # Tricellular junction vertices
        tricell_vertices = set()
        junction_gap = getattr(self, 'junction_gap_radius', 0)
        if junction_gap > 0:
            vert_degree = {}
            n = self.nb_cells
            for ridge_idx, (p1, p2) in enumerate(self.vor.ridge_points):
                if p1 >= n or p2 >= n:
                    continue
                vert_indices = self.vor.ridge_vertices[ridge_idx]
                if -1 in vert_indices:
                    continue
                for vi in vert_indices:
                    vert_degree[vi] = vert_degree.get(vi, 0) + 1
            tricell_vertices = {vi for vi, deg in vert_degree.items() if deg >= 3}

        # Shared-boundary brightness with per-ridge width variability
        ridge_rng = np.random.default_rng(self.rng_seed + 7777)
        n = self.nb_cells
        for ridge_idx, (p1, p2) in enumerate(self.vor.ridge_points):
            if p1 >= n or p2 >= n:
                continue
            if not (self.has_membrane_marker[p1] and self.has_membrane_marker[p2]):
                continue
            p1_ok = renderable is None or renderable[p1]
            p2_ok = renderable is None or renderable[p2]
            if not p1_ok and not p2_ok:
                continue  # Both ghost: skip
            # wound_edge = exactly one side is a ghost (alive↔ghost boundary)
            wound_edge = p1_ok != p2_ok
            alive_idx = p1 if p1_ok else p2
            if self._get_apoptosis_stage(alive_idx) >= 2:
                continue
            vert_indices = self.vor.ridge_vertices[ridge_idx]
            if -1 in vert_indices:
                continue
            v1 = self.vor.vertices[vert_indices[0]].copy()
            v2 = self.vor.vertices[vert_indices[1]].copy()
            v1 = np.clip(v1, [0, 0], [self.width - 1, self.height - 1])
            v2 = np.clip(v2, [0, 0], [self.width - 1, self.height - 1])

            if not wound_edge and junction_gap > 0:
                direction = v2 - v1
                length = np.linalg.norm(direction)
                if length > 0:
                    unit = direction / length
                    if vert_indices[0] in tricell_vertices:
                        v1 = v1 + unit * min(junction_gap, length * 0.4)
                    if vert_indices[1] in tricell_vertices:
                        v2 = v2 - unit * min(junction_gap, length * 0.4)

            avg_int = self.membrane_intensity[alive_idx] if wound_edge else \
                (self.membrane_intensity[p1] + self.membrane_intensity[p2]) / 2
            # Per-ridge intensity variation (mechanical heterogeneity)
            int_var = ridge_rng.uniform(0.8, 1.2)
            bright = min(255, int(avg_int * 220 * int_var))
            if wound_edge:
                # Wound edge: bright lamellipodia-like leading edge (60% intensity,
                # thicker than normal junctions to be visible in wound zone)
                bright = int(bright * 0.60)
                ridge_thick = max(3, int(lt * 1.5))
            else:
                # Per-ridge thickness variation
                ridge_thick = max(2, int(lt * ridge_rng.uniform(0.7, 1.5)))
            pt1 = (self._s(v1[0]), self._s(v1[1]))
            pt2 = (self._s(v2[0]), self._s(v2[1]))
            cv2.line(img, pt1, pt2, (bright, bright, bright),
                     thickness=ridge_thick, lineType=cv2.LINE_AA)

        # Erase at tricellular junctions (gap mode)
        if junction_gap > 0 and tricell_vertices:
            gap_s = junction_gap * s
            for vi in tricell_vertices:
                vx, vy = self.vor.vertices[vi]
                vx_i = self._s(np.clip(vx, 0, self.width - 1))
                vy_i = self._s(np.clip(vy, 0, self.height - 1))
                cv2.circle(img, (vx_i, vy_i), int(gap_s),
                           (5, 5, 5), -1, cv2.LINE_AA)

        # Tricellular vertex brightening — cadherin clusters are larger
        # at 3-way junctions, creating bright spots (real E-cadherin feature)
        if junction_gap <= 0:
            # Identify tricellular vertices if not already done
            if not tricell_vertices:
                vert_degree = {}
                nn = self.nb_cells
                for ridge_idx, (p1, p2) in enumerate(self.vor.ridge_points):
                    if p1 >= nn or p2 >= nn:
                        continue
                    vert_indices = self.vor.ridge_vertices[ridge_idx]
                    if -1 in vert_indices:
                        continue
                    for vi in vert_indices:
                        vert_degree[vi] = vert_degree.get(vi, 0) + 1
                tricell_vertices = {vi for vi, deg in vert_degree.items()
                                    if deg >= 3}
            # Draw bright spots at vertices
            vertex_r = max(2, int(lt * 1.5))
            for vi in tricell_vertices:
                vx, vy = self.vor.vertices[vi]
                if 0 <= vx < self.width and 0 <= vy < self.height:
                    vx_i = self._s(vx)
                    vy_i = self._s(vy)
                    cv2.circle(img, (vx_i, vy_i), vertex_r,
                               (220, 220, 220), -1, cv2.LINE_AA)

        if not self._skip_pipeline:
            img = self._mem_pipeline.apply(img, wavelength_nm=self.mem_wavelength)
        return img

    def _render_cyto_full(self) -> np.ndarray:
        """Render cytoplasm fluorescence channel at internal resolution."""
        s = self.internal_scale
        img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)

        renderable = getattr(self, '_renderable', None)
        for i, poly in enumerate(self.render_polygons[:self.nb_cells]):
            if len(poly) < 3 or not self.has_cytoplasm_marker[i]:
                continue
            if renderable is not None and not renderable[i]:
                continue
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            intensity = int(self.cytoplasm_intensity[i] * 180)
            cv2.fillPoly(img, [pts], (intensity, intensity, intensity))
            ncx, ncy = self._nuc_center(i)
            if 0 <= ncx < self.width and 0 <= ncy < self.height:
                nuc_val = max(0, int(intensity * 0.15))
                self._draw_nucleus(img, i, (nuc_val, nuc_val, nuc_val), -1)

        if not self._skip_pipeline:
            img = self._nuc_pipeline.apply(img)
        return img

    def _render_dic_full(self, shear_angle: float = 0.785) -> np.ndarray:
        """Render DIC image at internal resolution."""
        s = self.internal_scale
        opl = np.full((self._ih, self._iw), 0.0, dtype=np.float32)
        renderable = getattr(self, '_renderable', None)
        for i, poly in enumerate(self.render_polygons[:self.nb_cells]):
            if len(poly) < 3 or (renderable is not None and not renderable[i]):
                continue
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            cell_opl = float(self.cell_gray[i]) / 255.0 * 0.5 + 0.3
            mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            opl[mask > 0] = cell_opl

        for i in range(self.nb_cells):
            if renderable is not None and not renderable[i]:
                continue
            ncx, ncy = self._nuc_center(i)
            r = max(3 * s, self._s(self.nucleus_radii[i]))
            ix, iy = self._s(ncx), self._s(ncy)
            if 0 <= ncx < self.width and 0 <= ncy < self.height:
                mask = np.zeros((self._ih, self._iw), dtype=np.uint8)
                asp = self._nuc_aspect[i]
                angle_deg = np.degrees(self._nuc_orient[i])
                axes = (int(round(r * asp)), r)
                cv2.ellipse(mask, (ix, iy), axes, angle_deg, 0, 360, 255, -1)
                opl[mask > 0] = 0.8

        k = max(3, 5 * s) | 1  # ensure odd kernel
        opl = cv2.GaussianBlur(opl, (k, k), 1.0 * s)

        dx = cv2.Sobel(opl, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(opl, cv2.CV_32F, 0, 1, ksize=3)
        dic = dx * np.cos(shear_angle) + dy * np.sin(shear_angle)

        dic_img = (128 + dic * 350).clip(0, 255).astype(np.uint8)
        dic_bgr = cv2.merge([dic_img, dic_img, dic_img])

        if not self._skip_pipeline:
            dic_bgr = self._bf_pipeline.apply(dic_bgr)
        return dic_bgr

    def enable_dic_channel(self, core=None, shear_angle: float = 0.785):
        """Render and register DIC as an extra channel.

        Returns the mode_id assigned to this channel.
        """
        dic_img = self._render_dic_full(shear_angle)
        mode_id = self.add_channel(
            "dic", dic_img,
            filter_label="Electra1(402/454)", led_label="BLUE",
        )
        if core is not None:
            core.defineConfig("Fake", "dic-channel",
                              "LED", "Label", "BLUE")
            core.defineConfig("Fake", "dic-channel",
                              "Filter Wheel", "Label", "Electra1(402/454)")
        return mode_id

    def enable_cytoplasm_channel(self, core=None):
        """Render and register cytoplasm as an extra channel.

        Call after setting has_cytoplasm_marker and cytoplasm_intensity.
        Optionally pass a core object to register the Fake config.

        Returns the mode_id assigned to this channel.
        """
        cyto_img = self._render_cyto_full()
        mode_id = self.add_channel(
            "cytoplasm", cyto_img,
            filter_label="TagGFP2(483/506)", led_label="GREEN",
        )
        if core is not None:
            core.defineConfig("Fake", "cytoplasm-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "cytoplasm-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")
        return mode_id

    def _render_translocation_full(self) -> np.ndarray:
        """Render GFP translocation reporter at internal resolution.

        Shows a protein (e.g. NF-kB-GFP) distributed between nucleus and
        cytoplasm. Nuclear fraction is controlled by _transloc_nuc_fraction.
        """
        s = self.internal_scale
        img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)
        renderable = getattr(self, '_renderable', None)
        nuc_frac = getattr(self, '_transloc_nuc_fraction', None)
        total = getattr(self, '_transloc_total_reporter', None)
        if nuc_frac is None or total is None:
            return img

        # Cytoplasmic GFP: fill cell polygon with (1 - nuc_fraction) * total
        for i, poly in enumerate(self.render_polygons[:self.nb_cells]):
            if len(poly) < 3:
                continue
            if renderable is not None and not renderable[i]:
                continue
            cyto_signal = (1.0 - nuc_frac[i]) * total[i]
            intensity = int(cyto_signal * 120)
            pts = (poly * s).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(img, [pts], (intensity, intensity, intensity))

        # Nuclear GFP: bright nuclei where nuc_fraction is high
        for i in range(self.nb_cells):
            if renderable is not None and not renderable[i]:
                continue
            ncx, ncy = self._nuc_center(i)
            if not (0 <= ncx < self.width and 0 <= ncy < self.height):
                continue
            nuc_signal = nuc_frac[i] * total[i]
            intensity = int(nuc_signal * 200)
            self._draw_nucleus(img, i, (intensity, intensity, intensity), -1)

        if not self._skip_pipeline:
            img = self._nuc_pipeline.apply(img, wavelength_nm=488)
        return img

    # ---- SimulationBridge-compatible interface ----

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs) -> np.ndarray:
        """Capture a frame — compatible with MicroscopeSimOptmized.snap_frame()."""
        self._update_mode()
        self._update_objectif()
        self._snap_count += 1

        # In live pipeline mode, re-apply pipeline with bleaching
        if self.live_pipeline and self._bf_clean is not None:
            pipelines = {0: self._bf_pipeline, 1: self._nuc_pipeline, 2: self._mem_pipeline}
            cleans = {0: self._bf_clean, 1: self._nuc_clean, 2: self._mem_clean}
            wavelengths = {0: 0, 1: self.nuc_wavelength, 2: self.mem_wavelength}
            if self.mode in pipelines and self.mode in cleans:
                full_img = pipelines[self.mode].apply_with_bleach(
                    cleans[self.mode].copy(), exposure_ms=exposure,
                    wavelength_nm=wavelengths.get(self.mode, 0))
            elif self.mode in self._extra_channels:
                full_img = self._extra_channels[self.mode]["image"]
            else:
                full_img = self._bf_pipeline.apply_with_bleach(
                    self._bf_clean.copy(), exposure_ms=exposure)
        else:
            # Static mode (default): use pre-rendered images
            if self.mode == 0:
                full_img = self._bf_full
            elif self.mode == 1:
                if self._nuc_z_layers is not None:
                    full_img = self._composite_z_nuc()
                else:
                    full_img = self._nuc_full
            elif self.mode == 2:
                full_img = self._mem_full
            elif self.mode in self._extra_channels:
                full_img = self._extra_channels[self.mode]["image"]
            else:
                full_img = self._bf_full

        # Apply stage drift (cumulative XY shift)
        if self.stage_drift_rate > 0 or self.stage_drift_noise > 0:
            self._apply_stage_drift()

        # Crop FOV from internal-resolution buffer
        viewport = self._crop_fov(full_img)

        # Apply Z-defocus blur if focal plane != tissue plane
        viewport = self._apply_defocus(viewport)

        # Apply exposure and intensity.
        # BF (transmitted light) uses 2× base so default exposure=50 gives
        # full contrast. Fluorescence uses standard 1× photon-collection model.
        if self.mode == 0:
            scale = min(intensity * 0.02 * exposure, 2.0)
        else:
            scale = intensity * 0.01 * exposure
        viewport = (
            viewport.astype(np.float32) * scale
        ).clip(0, 255).astype(np.uint8)

        # Return grayscale
        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def _apply_stage_drift(self):
        """Accumulate stage drift — called once per snap_frame().

        Combines linear drift (constant direction) with random walk noise.
        The drift is applied to the viewport extraction, not to camera_offset
        itself, so the agent can still read the "commanded" position correctly.
        """
        # Linear drift component
        if self.stage_drift_rate > 0:
            dx = self.stage_drift_rate * np.cos(self.stage_drift_angle)
            dy = self.stage_drift_rate * np.sin(self.stage_drift_angle)
            self._drift_accumulator[0] += dx
            self._drift_accumulator[1] += dy

        # Random walk component
        if self.stage_drift_noise > 0:
            self._drift_accumulator += self._drift_rng.normal(
                0, self.stage_drift_noise, 2)

    def get_stage_drift(self) -> tuple:
        """Return current accumulated stage drift (dx, dy) in pixels."""
        return (round(float(self._drift_accumulator[0]), 2),
                round(float(self._drift_accumulator[1]), 2))

    def reset_stage_drift(self):
        """Reset accumulated drift to zero."""
        self._drift_accumulator[:] = 0.0

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply defocus blur based on distance from focal plane to tissue."""
        # Skip for nucleus channel when Z-stack layers handle defocus
        if self.mode == 1 and self._nuc_z_layers is not None:
            return img
        dz = abs(self.focal_plane - self.tissue_z)
        half_dof = self._dof / 2.0
        if dz <= half_dof:
            return img  # within depth of field — sharp
        # Defocus amount in µm beyond DOF boundary
        defocus_um = dz - half_dof
        blur_scale = self._blur_scale_table.get(self.current_objectiv, 0.5)
        sigma = defocus_um * blur_scale
        if sigma < 0.3:
            return img
        # Cap sigma to avoid extreme blur (max ~30px)
        sigma = min(sigma, 30.0)
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        # Fade toward background as defocus increases (opacity drop)
        # At 3x DOF defocus, opacity → ~0.3
        opacity = max(0.2, 1.0 / (1.0 + 0.3 * (defocus_um / max(0.5, self._dof))))
        if opacity < 0.99:
            bg_val = 20 if self.mode == 0 else 0
            bg = np.full_like(blurred, bg_val)
            blurred = cv2.addWeighted(blurred, opacity, bg, 1.0 - opacity, 0)
        return blurred

    # ---- Z-stack rendering ----

    def enable_z_stack(self, z_range: float = 6.0, n_layers: int = 7):
        """Enable per-cell Z-depth for nucleus channel.

        Assigns each cell nucleus a random Z position spanning ±z_range/2
        around tissue_z.  Nuclei are binned into *n_layers* discrete Z-layers
        and pre-rendered separately.  At snap time the layers are composited
        with per-layer defocus, so scanning through Z shows different nuclei
        coming in and out of focus.

        Args:
            z_range: Total depth range in µm (e.g. 6.0 means ±3 µm).
            n_layers: Number of discrete Z-layers (odd recommended).
        """
        z_lo = self.tissue_z - z_range / 2.0
        z_hi = self.tissue_z + z_range / 2.0
        self.nucleus_z = self.rng.uniform(z_lo, z_hi, self.nb_cells)
        self._z_layer_positions = np.linspace(z_lo, z_hi, n_layers)
        # Assign each cell to its nearest layer
        self._cell_z_layer = np.argmin(
            np.abs(self.nucleus_z[:, None] - self._z_layer_positions[None, :]),
            axis=1,
        )
        self._render_z_layers()

    def _render_z_layers(self):
        """Pre-render nucleus images per Z-layer at internal resolution."""
        s = self.internal_scale
        self._nuc_z_layers = {}
        renderable = getattr(self, "_renderable", None)
        for layer_idx in range(len(self._z_layer_positions)):
            cells = np.where(self._cell_z_layer == layer_idx)[0]
            img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)
            for i in cells:
                if renderable is not None and not renderable[i]:
                    continue
                if not self.has_nucleus_marker[i]:
                    continue
                ncx, ncy = self._nuc_center(i)
                if 0 <= ncx < self.width and 0 <= ncy < self.height:
                    val = int(self.nucleus_intensity[i] * 200)
                    self._draw_nucleus(img, i, (val, val, val), -1)
                    r = max(3 * s, self._s(self.nucleus_radii[i]))
                    ix, iy = self._s(ncx), self._s(ncy)
                    bright = min(255, int(val * 1.2))
                    cv2.circle(img, (ix, iy), max(1, r // 2),
                               (bright, bright, bright), -1, cv2.LINE_AA)
            self._nuc_z_layers[layer_idx] = img

    def _composite_z_nuc(self) -> np.ndarray:
        """Composite pre-rendered nucleus Z-layers with per-layer defocus."""
        h, w = self._ih, self._iw
        composite = np.zeros((h, w, 3), dtype=np.float32)
        half_dof = self._dof / 2.0
        blur_scale = self._blur_scale_table.get(self.current_objectiv, 0.5)

        for layer_idx, z in enumerate(self._z_layer_positions):
            layer = self._nuc_z_layers[layer_idx]
            if layer.max() == 0:
                continue  # empty layer
            dz = abs(self.focal_plane - z)
            if dz <= half_dof:
                # In focus
                composite += layer.astype(np.float32)
            else:
                defocus_um = dz - half_dof
                sigma = min(defocus_um * blur_scale, 30.0)
                opacity = max(0.15, 1.0 / (1.0 + 0.3 * (defocus_um / max(0.5, self._dof))))
                if sigma >= 0.3:
                    blurred = cv2.GaussianBlur(layer, (0, 0), sigma)
                else:
                    blurred = layer
                composite += blurred.astype(np.float32) * opacity

        return np.clip(composite, 0, 255).astype(np.uint8)

    def _crop_fov(self, full):
        """Crop FOV from internal-res buffer, resize to viewport."""
        s = self.internal_scale
        ih, iw = full.shape[:2]
        out_w, out_h = self.viewport_width, self.viewport_height
        obj = self.current_objectiv

        fov_map = {100: 64, 40: 128, 20: 256}
        fov_world = fov_map.get(obj, min(512, self.width))

        fov_int = fov_world * s

        # Stage center in world coords → internal coords (includes drift)
        cx_world = int(self.camera_offset[0] + self._drift_accumulator[0]) + out_w // 2
        cy_world = int(self.camera_offset[1] + self._drift_accumulator[1]) + out_h // 2
        cx_int = int(cx_world * s)
        cy_int = int(cy_world * s)

        half = fov_int // 2
        x0 = max(0, min(cx_int - half, iw - fov_int))
        y0 = max(0, min(cy_int - half, ih - fov_int))

        crop = full[y0:y0 + fov_int, x0:x0 + fov_int].copy()

        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            bg_val = 20 if self.mode == 0 else 0
            if len(crop.shape) == 3:
                padded = np.full((fov_int, fov_int, crop.shape[2]), bg_val, dtype=crop.dtype)
            else:
                padded = np.full((fov_int, fov_int), bg_val, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        if crop.shape[0] > out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        elif crop.shape[0] < out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop

    def _update_mode(self):
        """Update rendering mode from state devices via _mode_map lookup."""
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            self.mode = 0
            return
        filter_label = self.state_devices["Filter Wheel"]["label"]
        led_label = self.state_devices["LED"]["label"]

        key = (filter_label, led_label)
        if key in self._mode_map:
            self.mode = self._mode_map[key]
            return
        for mode_id, ch_info in self._extra_channels.items():
            if filter_label == ch_info["filter"] and led_label == ch_info["led"]:
                self.mode = mode_id
                return
        self.mode = 0

    def _update_objectif(self):
        """Update objective from state devices and set DOF."""
        if "Objective" not in self.state_devices:
            return
        obj_label = self.state_devices["Objective"]["label"]
        if obj_label in self._objectif_dict:
            self.current_objectiv = self._objectif_dict[obj_label]
            self._dof = self._dof_table.get(self.current_objectiv, 6.0)

    def set_focal_plane(self, z: float):
        """Set focal plane (µm). Defocus blur applied when off tissue_z."""
        self.focal_plane = z

    def update(self, dt: float = 0.016):
        """Update simulation (no-op for static tissue)."""
        pass

    def _init_numpy_arrays(self):
        """Initialize numpy arrays (compatibility stub)."""
        pass

    # ---- Renderer stub for set_objective calls ----

    class _RendererStub:
        def set_objective(self, obj, dof):
            pass

    @property
    def renderer(self):
        return self._RendererStub()

    # ---- Extra channel registration ----

    def add_channel(self, name: str, image: np.ndarray,
                    filter_label: str, led_label: str) -> int:
        """Register an extra fluorescence channel beyond BF/nucleus/membrane.

        Args:
            name: Channel name (e.g., "Ki67", "phalloidin")
            image: Pre-rendered full-tissue image (HxWx3, uint8)
            filter_label: Filter Wheel label for this channel
            led_label: LED label for this channel

        Returns:
            Mode ID assigned to this channel
        """
        mode_id = self._next_mode
        self._extra_channels[mode_id] = {
            "name": name,
            "image": image,
            "filter": filter_label,
            "led": led_label,
        }
        self._next_mode += 1
        return mode_id

    # ---- Marker configuration ----

    def set_marker_pattern(
        self,
        nucleus_positive_indices: list | None = None,
        membrane_positive_indices: list | None = None,
    ):
        """Configure which cells express each marker, then re-render."""
        if nucleus_positive_indices is not None:
            self.has_nucleus_marker[:] = False
            self.has_nucleus_marker[nucleus_positive_indices] = True
        if membrane_positive_indices is not None:
            self.has_membrane_marker[:] = False
            self.has_membrane_marker[membrane_positive_indices] = True
        self._render_full_tissue()

    def apply_marker_config(self, config: dict, rng=None):
        """Apply data-driven marker configuration.

        Config format:
            {
                "nucleus": {
                    "fraction": 0.5,       # fraction of cells that are positive
                    "pattern": "random",   # "random", "left", "right", "gradient_x"
                    "intensity": {
                        "mode": "uniform", # "uniform", "bimodal", "gradient_x"
                        "range": [0.7, 1.0],          # for uniform
                        "bright_range": [0.85, 1.0],  # for bimodal
                        "dim_range": [0.4, 0.6],      # for bimodal
                        "bright_fraction": 0.5,        # for bimodal
                    }
                },
                "membrane": {
                    "fraction": 1.0,
                    "pattern": "all",
                    "intensity": {"mode": "uniform", "range": [0.6, 0.9]}
                }
            }
        """
        if rng is None:
            rng = self.rng

        for channel, key, marker_arr, intensity_arr in [
            ("nucleus", "nucleus", self.has_nucleus_marker, self.nucleus_intensity),
            ("membrane", "membrane", self.has_membrane_marker, self.membrane_intensity),
            ("cytoplasm", "cytoplasm", self.has_cytoplasm_marker, self.cytoplasm_intensity),
        ]:
            cfg = config.get(key)
            if cfg is None:
                continue

            frac = cfg.get("fraction", 1.0)
            pattern = cfg.get("pattern", "random")
            n_pos = int(self.nb_cells * frac)

            # Handle zero-fraction case
            if n_pos <= 0:
                marker_arr[:] = False
                continue

            # Select positive cells based on pattern
            if pattern == "all" or frac >= 1.0:
                indices = list(range(self.nb_cells))
            elif pattern == "random":
                indices = sorted(rng.choice(self.nb_cells, n_pos, replace=False).tolist())
            elif pattern == "left":
                left = [i for i in range(self.nb_cells)
                        if self.cell_centroids[i][0] < self.width / 2]
                indices = sorted(rng.choice(left, min(n_pos, len(left)), replace=False).tolist())
            elif pattern == "right":
                right = [i for i in range(self.nb_cells)
                         if self.cell_centroids[i][0] >= self.width / 2]
                indices = sorted(rng.choice(right, min(n_pos, len(right)), replace=False).tolist())
            elif pattern == "gradient_x":
                # Probability increases left to right
                probs = np.array([self.cell_centroids[i][0] / self.width
                                  for i in range(self.nb_cells)])
                probs /= probs.sum()
                indices = sorted(rng.choice(self.nb_cells, n_pos, replace=False, p=probs).tolist())
            else:
                indices = sorted(rng.choice(self.nb_cells, n_pos, replace=False).tolist())

            marker_arr[:] = False
            marker_arr[indices] = True

            # Set intensities
            icfg = cfg.get("intensity")
            if icfg is not None:
                mode = icfg.get("mode", "uniform")
                if mode == "uniform":
                    lo, hi = icfg.get("range", [0.6, 1.0])
                    intensity_arr[:] = rng.uniform(lo, hi, self.nb_cells)
                elif mode == "bimodal":
                    bright_r = icfg.get("bright_range", [0.85, 1.0])
                    dim_r = icfg.get("dim_range", [0.4, 0.6])
                    bf = icfg.get("bright_fraction", 0.5)
                    n_bright = int(len(indices) * bf)
                    for j, idx in enumerate(indices):
                        if j < n_bright:
                            intensity_arr[idx] = rng.uniform(*bright_r)
                        else:
                            intensity_arr[idx] = rng.uniform(*dim_r)
                elif mode == "gradient_x":
                    lo, hi = icfg.get("range", [0.4, 1.0])
                    for idx in indices:
                        frac_x = self.cell_centroids[idx][0] / self.width
                        intensity_arr[idx] = lo + (hi - lo) * frac_x

        self._render_full_tissue()

    # ---- Metrics ----

    def get_confluency(self, region=None):
        """Compute fraction of area covered by renderable cells.

        Args:
            region: Optional (x1, y1, x2, y2) to measure within a sub-region.
                    Default: full world.

        Returns:
            float in [0, 1] — fraction of region covered by cells.
        """
        if region is None:
            x1, y1, x2, y2 = 0, 0, self.width, self.height
        else:
            x1, y1, x2, y2 = region

        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        renderable = getattr(self, '_renderable', None)
        for i, poly in enumerate(self.cell_polygons[:self.nb_cells]):
            if len(poly) < 3:
                continue
            if renderable is not None and not renderable[i]:
                continue
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)

        roi = mask[y1:y2, x1:x2]
        total_pixels = roi.size
        covered_pixels = (roi > 0).sum()
        return round(float(covered_pixels / max(1, total_pixels)), 4)

    # ---- Ground truth ----

    def get_ground_truth(self):
        """Return ground truth data including shape descriptors."""
        cells = []
        for i in range(self.nb_cells):
            cx, cy = self.cell_centroids[i]
            poly = self.cell_polygons[i] if i < len(self.cell_polygons) else np.array([])

            # Shape descriptors
            perimeter = 0.0
            circularity = 0.0
            elongation = 1.0
            if len(poly) >= 3:
                # Perimeter
                diffs = np.diff(np.vstack([poly, poly[:1]]), axis=0)
                perimeter = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
                # Circularity = 4π·area / perimeter²
                area = self.cell_areas[i]
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                # Elongation from fitted ellipse (if enough points)
                if len(poly) >= 5:
                    pts_i32 = poly.astype(np.float32).reshape(-1, 1, 2)
                    (_, _), (w, h), _ = cv2.fitEllipse(pts_i32)
                    if min(w, h) > 0:
                        elongation = max(w, h) / min(w, h)

            cells.append({
                "idx": i,
                "centroid_x": round(float(cx), 1),
                "centroid_y": round(float(cy), 1),
                "area": round(float(self.cell_areas[i]), 1),
                "perimeter": round(perimeter, 1),
                "circularity": round(circularity, 3),
                "elongation": round(elongation, 2),
                "nucleus_radius": round(float(self.nucleus_radii[i]), 1),
                "has_nucleus_marker": bool(self.has_nucleus_marker[i]),
                "has_membrane_marker": bool(self.has_membrane_marker[i]),
            })

        return {
            "n_cells": self.nb_cells,
            "cells": cells,
            "mean_area": round(float(self.cell_areas.mean()), 1),
            "n_nucleus_positive": int(self.has_nucleus_marker.sum()),
            "n_membrane_positive": int(self.has_membrane_marker.sum()),
        }
