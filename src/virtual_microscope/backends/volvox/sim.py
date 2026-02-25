"""
VolvoxSim — 3D colonial green alga simulation.

A Volvox colony consists of ~200-500 somatic cells arranged on the surface
of a hollow sphere, plus a few large reproductive cells (gonidia) inside.
The colony rotates around its anterior-posterior axis and swims through
the medium. It exhibits phototaxis — the colony turns toward light.

Compatible with SimulationBridge via snap_frame() interface.

Physics:
  - Cells distributed on sphere surface (Fibonacci lattice)
  - Colony rotates around its axis (~0.5-2 rev/sec)
  - Swimming: constant forward velocity along axis + Brownian noise
  - Phototaxis: SLM light mask → colony steers toward illuminated side

Rendering:
  - Project 3D cell positions to 2D at the current focal_plane
  - Cells near focal plane: bright dots; far cells: dim/blurred
  - Gonidia: larger, brighter dots inside the sphere
  - Brightfield: colony outline + internal structure
  - Chlorophyll: green autofluorescence from all cells

Usage:
    sim = VolvoxSim(n_somatic=300, colony_radius=60, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


def _fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    """Generate n points uniformly distributed on a sphere surface."""
    points = np.zeros((n, 3))
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
        phi = 2.0 * np.pi * i / golden
        points[i] = [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ]
    return points


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for axis-angle rotation."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


class VolvoxSim(SimBase):
    """Volvox colony simulation compatible with SimulationBridge."""

    continuous = True

    def __init__(
        self,
        n_somatic: int = 300,
        n_gonidia: int = 4,
        colony_radius: float = 60.0,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        seed: int = 42,
        swim_speed: float = 3.0,
        rotation_speed: float = 0.15,
        fixed_dt: float = 1.0,
        internal_scale: int = 1,
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=fixed_dt,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
                ("mScarlet3(569/582)", "ORANGE"): 2,
            },
        )

        self.n_somatic = n_somatic
        self.n_gonidia = n_gonidia
        self.colony_radius = colony_radius
        self._ivw = viewport_width * internal_scale   # internal viewport width
        self._ivh = viewport_height * internal_scale   # internal viewport height
        self.seed = seed

        # Colony state
        self.colony_pos = np.array([
            world_size / 2.0,
            world_size / 2.0,
            0.0,  # Z position (µm)
        ])
        # Colony orientation: axis of rotation (initially pointing up in Z)
        self.colony_axis = np.array([0.0, 0.0, 1.0])
        # Forward direction (initially along +Y in world)
        self.swim_dir = np.array([0.0, 1.0, 0.0])
        self.rotation_angle = 0.0  # current rotation phase

        # Dynamics
        self.swim_speed = swim_speed       # px per step
        self.rotation_speed = rotation_speed  # radians per step
        self.brownian_noise = 0.5          # px per step

        # Generate cell positions on sphere (local coordinates)
        self._somatic_local = _fibonacci_sphere(n_somatic, colony_radius)
        # Gonidia: inside the sphere (randomly placed at ~60% radius)
        self._gonidia_local = self.rng.normal(
            0, colony_radius * 0.3, (n_gonidia, 3)
        )
        # Clamp gonidia to be inside sphere
        for i in range(n_gonidia):
            r = np.linalg.norm(self._gonidia_local[i])
            if r > colony_radius * 0.6:
                self._gonidia_local[i] *= colony_radius * 0.5 / r

        # Per-cell properties
        self.somatic_brightness = self.rng.uniform(0.6, 1.0, n_somatic)
        self.gonidia_brightness = self.rng.uniform(0.8, 1.0, n_gonidia)
        self.somatic_radius_px = 2.5  # rendered dot size at 40x
        self.gonidia_radius_px = 6.0  # larger reproductive cells

        # SLM phototaxis
        self._current_slm_mask = None
        self.phototaxis_strength = 0.15  # radians per step toward light

        # Optical pipeline — same pipeline for all channels
        pipe = OpticalPipeline(
            psf_sigma=0.8,
            noise={"photon_scale": 3.0, "read_std": 2.0},
            rng_seed=seed + 1000,
        )
        self._pipeline = {0: pipe, 1: pipe, 2: pipe}

        # For compatibility
        self._cells = []

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 25.0  # algae default
        return float(self.state_devices["Temperature"].get("label", "25"))

    def _temp_speed_factor(self) -> float:
        """Temperature-dependent speed factor for Volvox (green alga).

        Optimal at ~25°C. Q10 ~ 1.5 for flagellar beating.
        Cold slows, heat above 35°C causes stress.
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.05
        if temp <= 25:
            return 1.5 ** ((temp - 25) / 10.0)
        if temp <= 35:
            return 1.0 - 0.04 * (temp - 25)  # 35°C → 0.6
        return max(0.05, 0.6 - 0.15 * (temp - 35))

    def step(self, dt: float = 1.0):
        """Advance colony dynamics by one timestep."""
        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        temp_factor = self._temp_speed_factor()

        # Rotate around colony axis
        self.rotation_angle += self.rotation_speed * dt * temp_factor
        if self.rotation_angle > 2 * np.pi:
            self.rotation_angle -= 2 * np.pi

        # Swimming: move along swim direction
        self.colony_pos[:2] += self.swim_dir[:2] * self.swim_speed * dt * temp_factor
        self.colony_pos[:2] += self.rng.normal(0, self.brownian_noise, 2) * dt

        # Soft reflection at world boundaries
        r = self.colony_radius
        for axis in [0, 1]:
            if self.colony_pos[axis] < r:
                self.colony_pos[axis] = r
                # Reverse swim direction component + angular jitter
                self.swim_dir[axis] = abs(self.swim_dir[axis])
                jitter = self.rng.uniform(-0.26, 0.26)  # ±15°
                R = _rotation_matrix(np.array([0, 0, 1.0]), jitter)
                self.swim_dir = R @ self.swim_dir
                self.swim_dir /= np.linalg.norm(self.swim_dir) + 1e-12
            elif self.colony_pos[axis] > self.width - r:
                self.colony_pos[axis] = self.width - r
                self.swim_dir[axis] = -abs(self.swim_dir[axis])
                jitter = self.rng.uniform(-0.26, 0.26)
                R = _rotation_matrix(np.array([0, 0, 1.0]), jitter)
                self.swim_dir = R @ self.swim_dir
                self.swim_dir /= np.linalg.norm(self.swim_dir) + 1e-12

        self._time += dt

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — applies cached SLM phototaxis then steps."""
        if self._current_slm_mask is not None and np.any(self._current_slm_mask):
            self._apply_phototaxis(scale=1.0)
        self.step(dt)

    def _apply_phototaxis(self, scale=1.0):
        """Steer colony toward the centroid of SLM illumination."""
        mask = self._current_slm_mask
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return

        # Light centroid in viewport coords
        light_cx = np.mean(xs) + self.camera_offset[0]
        light_cy = np.mean(ys) + self.camera_offset[1]

        # Direction from colony to light (XY plane)
        dx = light_cx - self.colony_pos[0]
        dy = light_cy - self.colony_pos[1]
        dist = np.hypot(dx, dy)
        if dist < 1.0:
            return

        # Target direction
        target_dir = np.array([dx / dist, dy / dist, 0.0])

        # Rotate swim_dir toward target (limited by phototaxis_strength)
        cross = np.cross(self.swim_dir, target_dir)
        sin_angle = np.linalg.norm(cross)
        if sin_angle > 0.01:
            max_steer = self.phototaxis_strength * scale
            steer_angle = min(max_steer, np.arcsin(min(1.0, sin_angle)))
            steer_axis = cross / sin_angle
            R = _rotation_matrix(steer_axis, steer_angle)
            self.swim_dir = R @ self.swim_dir
            self.swim_dir /= np.linalg.norm(self.swim_dir) + 1e-12

    def _get_world_positions(self):
        """Get current 3D world positions of all cells.

        Returns:
            somatic_world: (n_somatic, 3) positions in world coords
            gonidia_world: (n_gonidia, 3) positions in world coords
        """
        # Apply rotation around colony axis
        R = _rotation_matrix(self.colony_axis, self.rotation_angle)
        somatic_rotated = (R @ self._somatic_local.T).T
        gonidia_rotated = (R @ self._gonidia_local.T).T

        # Translate to colony position
        somatic_world = somatic_rotated + self.colony_pos
        gonidia_world = gonidia_rotated + self.colony_pos

        return somatic_world, gonidia_world

    def _render_frame(self) -> np.ndarray:
        """Render the current scene at the current focal plane.

        Returns a (viewport_h, viewport_w, 3) uint8 image.
        """
        somatic_world, gonidia_world = self._get_world_positions()

        if self.mode == 0:
            internal = self._render_brightfield(somatic_world, gonidia_world)
        elif self.mode == 2:
            internal = self._render_gonidia(somatic_world, gonidia_world)
        else:
            internal = self._render_chlorophyll(somatic_world, gonidia_world)

        # Downscale from internal resolution to viewport
        if self.internal_scale > 1:
            vw, vh = self.viewport_width, self.viewport_height
            internal = cv2.resize(internal, (vw, vh), interpolation=cv2.INTER_AREA)
        return internal

    def _cell_visibility(self, cell_z: float) -> tuple:
        """Compute visibility (opacity, blur_sigma) for a cell at given Z.

        Returns:
            opacity: float [0, 1] — how bright the cell appears
            sigma: float — gaussian blur sigma in pixels
        """
        dz = abs(cell_z - (self.focal_plane - self.tissue_z))
        half_dof = self._dof / 2.0

        if dz <= half_dof:
            return 1.0, 0.0  # In focus

        defocus = dz - half_dof
        # Blur increases with defocus and magnification
        blur_scale = {10: 0.3, 20: 0.6, 40: 1.5}.get(self.current_objectiv, 1.0)
        sigma = defocus * blur_scale

        # Opacity falls off with defocus
        opacity = max(0.05, 1.0 / (1.0 + 0.5 * defocus / max(0.5, self._dof)))

        return opacity, min(sigma, 15.0)

    def _world_to_internal(self, world_xy: np.ndarray) -> np.ndarray:
        """Convert world XY coordinates to internal-resolution pixel coords.

        Accounts for camera_offset, objective magnification, and internal_scale.
        """
        ox, oy = self.camera_offset
        vw, vh = self.viewport_width, self.viewport_height
        s = self.internal_scale
        mag = self.current_objectiv / 10.0

        center_x = ox + vw / 2.0  # stage_x (center of FOV in world)
        center_y = oy + vh / 2.0

        # Convert to internal-resolution coords
        internal_xy = np.zeros_like(world_xy[:, :2])
        internal_xy[:, 0] = (world_xy[:, 0] - center_x) * mag * s + self._ivw / 2.0
        internal_xy[:, 1] = (world_xy[:, 1] - center_y) * mag * s + self._ivh / 2.0

        return internal_xy

    def _render_chlorophyll(self, somatic_world, gonidia_world) -> np.ndarray:
        """Render chlorophyll autofluorescence channel (additive blending)."""
        ih, iw = self._ivh, self._ivw
        s = self.internal_scale
        acc = np.zeros((ih, iw), dtype=np.float32)
        mag = self.current_objectiv / 10.0

        vp_pos = self._world_to_internal(somatic_world)
        for i in range(self.n_somatic):
            x, y = int(vp_pos[i, 0]), int(vp_pos[i, 1])
            if x < -20 * s or x > iw + 20 * s or y < -20 * s or y > ih + 20 * s:
                continue

            opacity, sigma = self._cell_visibility(somatic_world[i, 2])
            if opacity < 0.05:
                continue

            brightness = self.somatic_brightness[i] * 180.0 * opacity
            r = max(1, int(self.somatic_radius_px * mag * s))

            if sigma < 0.5:
                self._add_circle(acc, x, y, r, brightness)
            else:
                blur_r = max(r + 1, int(r + sigma * mag * s * 0.5))
                self._add_circle(acc, x, y, blur_r, brightness * 0.5)

        vp_gon = self._world_to_internal(gonidia_world)
        for i in range(self.n_gonidia):
            x, y = int(vp_gon[i, 0]), int(vp_gon[i, 1])
            if x < -20 * s or x > iw + 20 * s or y < -20 * s or y > ih + 20 * s:
                continue

            opacity, sigma = self._cell_visibility(gonidia_world[i, 2])
            if opacity < 0.05:
                continue

            brightness = self.gonidia_brightness[i] * 220.0 * opacity
            r = max(2, int(self.gonidia_radius_px * mag * s))

            if sigma < 0.5:
                self._add_circle(acc, x, y, r, brightness)
            else:
                blur_r = max(r + 1, int(r + sigma * mag * s * 0.5))
                self._add_circle(acc, x, y, blur_r, brightness * 0.4)

        g = np.clip(acc, 0, 255).astype(np.uint8)
        img = np.zeros((ih, iw, 3), dtype=np.uint8)
        img[:, :, 1] = g
        return img

    def _render_gonidia(self, somatic_world, gonidia_world) -> np.ndarray:
        """Render ECM compartment boundaries (membrane-channel).

        Volvox somatic cells are embedded in a multi-layered extracellular
        matrix (ECM).  With pherophorin-YFP or lectin staining, each cell's
        ECM compartment boundary is visible — creating a honeycomb-like
        pattern on the colony surface.

        Uses cv2 direct drawing (like BF) to preserve brightness through
        PSF blur and exposure scaling.
        """
        ih, iw = self._ivh, self._ivw
        s = self.internal_scale
        img = np.zeros((ih, iw, 3), dtype=np.uint8)
        mag = self.current_objectiv / 10.0

        # ECM compartment boundary radius (~half NN distance on Fibonacci lattice)
        surface_area = 4.0 * np.pi * self.colony_radius ** 2
        cell_area = surface_area / max(1, self.n_somatic)
        compartment_r_world = np.sqrt(cell_area / np.pi)  # ~6.9 px

        vp_pos = self._world_to_internal(somatic_world)
        for i in range(self.n_somatic):
            x, y = int(vp_pos[i, 0]), int(vp_pos[i, 1])
            if x < -20 * s or x > iw + 20 * s or y < -20 * s or y > ih + 20 * s:
                continue
            opacity, sigma = self._cell_visibility(somatic_world[i, 2])
            if opacity < 0.05:
                continue

            # Ring radius in internal pixels
            r_outer = max(2, int(compartment_r_world * mag * s))
            ring_t = max(1, int(1.5 * mag * s))  # ring line thickness

            val = int(min(255, 200 * opacity * self.somatic_brightness[i]))

            if sigma > 1.0:
                # Out of focus: faint filled circle (haze)
                blur_r = max(r_outer + 1, int(r_outer + sigma * mag * s * 0.3))
                haze = max(5, int(val * 0.15))
                cv2.circle(img, (x, y), blur_r, (0, haze, 0), -1, cv2.LINE_AA)
            else:
                # Bright annular ring — ECM compartment boundary
                cv2.circle(img, (x, y), r_outer, (0, val, 0),
                           ring_t, cv2.LINE_AA)
                # Small bright dot at cell center (flagellar base)
                dot_r = max(1, int(1.0 * mag * s))
                dot_val = max(5, int(val * 0.5))
                cv2.circle(img, (x, y), dot_r, (0, dot_val, 0),
                           -1, cv2.LINE_AA)

        # Gonidia: bright filled circles (large reproductive cells)
        vp_gon = self._world_to_internal(gonidia_world)
        for i in range(self.n_gonidia):
            x, y = int(vp_gon[i, 0]), int(vp_gon[i, 1])
            if x < -20 * s or x > iw + 20 * s or y < -20 * s or y > ih + 20 * s:
                continue
            opacity, sigma = self._cell_visibility(gonidia_world[i, 2])
            if opacity < 0.05:
                continue
            val = int(min(255, self.gonidia_brightness[i] * 240 * opacity))
            r = max(2, int(self.gonidia_radius_px * mag * s))
            if sigma < 0.5:
                cv2.circle(img, (x, y), r, (0, val, 0), -1, cv2.LINE_AA)
            else:
                blur_r = max(r + 1, int(r + sigma * mag * s * 0.5))
                haze = max(5, int(val * 0.4))
                cv2.circle(img, (x, y), blur_r, (0, haze, 0), -1, cv2.LINE_AA)

        return img

    @staticmethod
    def _add_circle(acc: np.ndarray, cx: int, cy: int, r: int, val: float):
        """Add a filled circle to the float accumulator (additive)."""
        h, w = acc.shape
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        acc[y0:y1, x0:x1][mask] += val

    def _render_brightfield(self, somatic_world, gonidia_world) -> np.ndarray:
        """Render brightfield with hollow-sphere optics.

        Volvox is a hollow sphere — optical thickness varies radially:
          - Rim (d ≈ R): tangential path through shell → DARKEST
          - Center (d ≈ 0): straight through front+back caps → lighter
        This creates the characteristic dark ring appearance.
        """
        ih, iw = self._ivh, self._ivw
        s = self.internal_scale
        bg_val = 200
        img = np.full((ih, iw, 3), bg_val, dtype=np.uint8)
        mag = self.current_objectiv / 10.0

        colony_vp = self._world_to_internal(self.colony_pos.reshape(1, 3))
        cx, cy = int(colony_vp[0, 0]), int(colony_vp[0, 1])
        r_colony = max(4, int(self.colony_radius * mag * s))

        # Shell thickness: ~8 world px (1 cell diameter + ECM)
        shell_t = max(3, int(8 * mag * s))
        r_inner = max(0, r_colony - shell_t)

        # Compute radial distance from colony center (clipped ROI for speed)
        x0 = max(0, cx - r_colony - 6 * s)
        x1 = min(iw, cx + r_colony + 6 * s)
        y0 = max(0, cy - r_colony - 6 * s)
        y1 = min(ih, cy + r_colony + 6 * s)
        if x1 <= x0 or y1 <= y0:
            return img

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)

        # Optical path through hollow sphere shell
        R2 = float(r_colony ** 2)
        r2 = float(r_inner ** 2)
        thickness = np.zeros_like(dist)

        # Region: r_inner <= d < r_colony → shell intersection
        mask_shell = (dist < r_colony) & (dist >= r_inner)
        d2 = dist[mask_shell] ** 2
        thickness[mask_shell] = 2.0 * np.sqrt(np.maximum(0.0, R2 - d2))

        # Region: d < r_inner → two thin caps (front + back)
        mask_inner = dist < r_inner
        d2i = dist[mask_inner] ** 2
        thickness[mask_inner] = 2.0 * (
            np.sqrt(np.maximum(0.0, R2 - d2i))
            - np.sqrt(np.maximum(0.0, r2 - d2i))
        )

        # Normalize → 0-1 darkness
        t_max = thickness.max() if thickness.max() > 0 else 1.0
        darkness = thickness / t_max

        # Colony body: bg minus optical thickness, slight green tint
        colony_mask = dist < r_colony
        gray_base = bg_val - 50.0 * darkness  # 150-200 range
        roi = img[y0:y1, x0:x1]
        for c, tint in enumerate([-2, 3, -2]):  # slight green tint (R↓ G↑ B↓)
            ch = roi[:, :, c].astype(np.float32)
            ch[colony_mask] = gray_base[colony_mask] + tint
            roi[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # ECM gel texture: faint noise inside colony
        tex_rng = np.random.default_rng(self.seed + 7777)
        tex = tex_rng.normal(0, 1.5, (y1 - y0, x1 - x0)).astype(np.float32)
        for c in range(3):
            ch = roi[:, :, c].astype(np.float32)
            ch[colony_mask] += tex[colony_mask]
            roi[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

        # Phase contrast bright halo outside colony boundary
        halo_w = max(2, 3 * s)
        halo_mask = (dist >= r_colony) & (dist < r_colony + halo_w)
        for c in range(3):
            roi[:, :, c][halo_mask] = np.minimum(
                255, roi[:, :, c][halo_mask].astype(np.int16) + 40
            ).astype(np.uint8)

        # Dark boundary edge (matrix/medium interface)
        edge_mask = (dist >= r_colony - max(1, 2 * s)) & (dist < r_colony)
        for c in range(3):
            roi[:, :, c][edge_mask] = np.maximum(
                0, roi[:, :, c][edge_mask].astype(np.int16) - 25
            ).astype(np.uint8)

        # --- Individual somatic cells ---
        vp_pos = self._world_to_internal(somatic_world)
        for i in range(self.n_somatic):
            x, y = int(vp_pos[i, 0]), int(vp_pos[i, 1])
            if x < -20 * s or x > iw + 20 * s or y < -20 * s or y > ih + 20 * s:
                continue
            opacity, sigma = self._cell_visibility(somatic_world[i, 2])
            if opacity < 0.1:
                continue

            r = max(1, int(self.somatic_radius_px * mag * s * 0.8))
            gray = max(70, int(150 - 80 * opacity))
            # Cell body (dark phase object)
            cv2.circle(img, (x, y), r, (gray, gray + 5, gray), -1, cv2.LINE_AA)
            # Phase halo around in-focus cells
            if opacity > 0.4 and r >= 2:
                halo_gray = min(235, gray + 55)
                cv2.circle(img, (x, y), r + max(1, s),
                           (halo_gray, halo_gray + 3, halo_gray),
                           max(1, s), cv2.LINE_AA)

        # --- Gonidia (large reproductive cells) ---
        vp_gon = self._world_to_internal(gonidia_world)
        for i in range(self.n_gonidia):
            x, y = int(vp_gon[i, 0]), int(vp_gon[i, 1])
            if x < -20 * s or x > iw + 20 * s or y < -20 * s or y > ih + 20 * s:
                continue
            opacity, sigma = self._cell_visibility(gonidia_world[i, 2])
            if opacity < 0.1:
                continue

            r = max(2, int(self.gonidia_radius_px * mag * s * 0.8))
            gray = max(55, int(130 - 75 * opacity))
            # Gonidia body (larger, darker than somatic)
            cv2.circle(img, (x, y), r, (gray, gray + 4, gray), -1, cv2.LINE_AA)
            # Dark edge ring (thick cell wall)
            edge_gray = max(40, gray - 25)
            cv2.circle(img, (x, y), r,
                       (edge_gray, edge_gray + 2, edge_gray),
                       max(1, s), cv2.LINE_AA)
            # Lighter interior (shade-off)
            inner_r = max(1, r - max(2, 2 * s))
            inner_gray = min(180, gray + 20)
            cv2.circle(img, (x, y), inner_r,
                       (inner_gray, inner_gray + 4, inner_gray),
                       -1, cv2.LINE_AA)

        return img

    def _render_for_mode(self, mode):
        return self._render_frame()

    # ── SimBase template-method overrides ─────────────────────

    def _handle_mask(self, mask):
        """Cache SLM mask for phototaxis steering."""
        self._current_slm_mask = mask

    def _auto_step_tick(self):
        """Apply phototaxis response before the physics step."""
        if self._current_slm_mask is not None and np.any(self._current_slm_mask):
            self._apply_phototaxis(scale=1.0)
        super()._auto_step_tick()

    def _crop_fov(self, full):
        """No-op crop — Volvox renders directly at viewport resolution."""
        return cv2.resize(full, (self.viewport_width, self.viewport_height))

    def _apply_defocus(self, img):
        """No-op — Volvox handles per-cell depth visibility internally."""
        return img

    def get_ground_truth(self) -> dict:
        """Return current colony state for grading."""
        somatic_world, gonidia_world = self._get_world_positions()
        return {
            "colony_x": round(float(self.colony_pos[0]), 1),
            "colony_y": round(float(self.colony_pos[1]), 1),
            "colony_z": round(float(self.colony_pos[2]), 1),
            "colony_radius": self.colony_radius,
            "n_somatic": self.n_somatic,
            "n_gonidia": self.n_gonidia,
            "rotation_angle": round(float(self.rotation_angle), 3),
            "swim_direction": [
                round(float(self.swim_dir[0]), 3),
                round(float(self.swim_dir[1]), 3),
            ],
            "time": round(float(self._time), 1),
        }
