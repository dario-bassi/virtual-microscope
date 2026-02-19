"""Single-Particle Tracking (SPT) simulator.

Simulates fluorescent membrane proteins diffusing on a 2D cell surface.
The "world" represents a single cell membrane patch imaged by TIRF/HiLo microscopy.

Three population types:
  FREE     — Brownian diffusion, D_free (µm²/s), linear MSD vs time
  CONFINED — Brownian inside circular corrals, D_confined (µm²/s), plateau MSD
  DIRECTED — Active transport + noise, directed_speed (µm/s), parabolic MSD

World scale: 1 world px = 0.1 µm.
World size:  128 × 128 px = 12.8 × 12.8 µm (one cell membrane patch)

Stage navigation:
  At 100x: FOV = 64 px = 6.4 µm (half the cell). Use XY to explore.
  At 40x:  FOV = 128 px = 12.8 µm (whole cell at once).
  Stage (0, 0) centers FOV at world center (64, 64).

Channels:
  mode 0 (brightfield): Epifluorescence overview — all particles (lower SNR)
  mode 1 (spt-channel): TIRF-like — bright individual spots, near-zero background

Workflow for agent:
  1. Snap overview at 40x (all particles visible)
  2. Navigate to densest region at 100x
  3. Acquire timelapse (30–50 frames, 100ms each)
  4. Detect particle centroids per frame (local maxima or blob detection)
  5. Link positions across frames (nearest-neighbor or Hungarian matching)
  6. Compute MSD(τ) = <|r(t+τ) - r(t)|²> for each track
  7. Fit: linear = free (D from slope/4), plateau = confined, parabolic = directed
  8. Report: D_free, D_confined, fraction_confined, fraction_directed

Pixel calibration:
  100x objective: FOV = 64 world px → camera image = 512 px
  → 1 camera pixel = 64/512 = 0.125 world px = 0.0125 µm
  (8 camera pixels span 1 world pixel at 100x)

SPT physics constraint (IMPORTANT for challenge design):
  Per-frame displacement σ = sqrt(2*D*dt) / PX_SCALE [world px]
  At D=0.1 µm²/s, dt=0.1s: σ = sqrt(0.02)/0.1 = 1.41 world px = 11 camera px → TRACKABLE
  At D=1.0 µm²/s, dt=0.1s: σ = sqrt(0.2)/0.1 = 4.47 world px = 36 camera px → TOO FAST
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from virtual_microscope.optical_pipeline import OpticalPipeline


class SPTSim:
    """Single-particle tracking simulation with multiple diffusion populations."""

    PX_SCALE = 0.1  # µm per world pixel

    def __init__(
        self,
        world_size: int = 128,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_free: int = 20,
        n_confined: int = 15,
        n_directed: int = 5,
        D_free: float = 1.0,         # µm²/s
        D_confined: float = 0.3,     # µm²/s (inside corral)
        D_directed: float = 0.1,     # µm²/s noise on directed
        confinement_radius: float = 0.5,  # µm corral radius
        directed_speed: float = 0.5,      # µm/s transport speed
        blink_rate: float = 0.05,    # per second: dark-state entry
        recovery_rate: float = 0.3,  # per second: dark-state exit
        bleach_rate: float = 0.003,  # per second: irreversible photobleach
        seed: int = 42,
        fixed_dt: float = 0.0,
    ):
        self.width = world_size
        self.height = world_size
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.world_pixel_size_um = 0.1  # 0.1 µm per world pixel (SPT nanoscale)
        self.internal_scale = 4  # internal rendering resolution
        self._iw = world_size * self.internal_scale
        self._ih = world_size * self.internal_scale

        self.rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 9999)
        self.fixed_dt = fixed_dt
        self._time = 0.0
        self._step_count = 0

        # SimulationBridge interface
        self.camera_offset = np.array([0.0, 0.0])
        self.focal_plane = 0.0
        self.tissue_z = 0.0
        self.state_devices = {}
        self.mode = 1  # default to SPT channel
        self.current_objectiv = 40   # start at 40x for overview
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40, "100x": 100}
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.6}
        self._dof = 1.5
        self._blur_scale_table = {10: 0.3, 20: 0.5, 40: 1.0, 100: 2.0}
        self._extra_channels = {}
        self._snap_count = 0
        self.auto_step = False
        self.snaps_per_step = 1
        self.z_drift_rate = 0.0
        self.z_drift_noise = 0.0

        # Physics parameters (store in µm units for diffusion calc)
        self.D_free = D_free
        self.D_confined = D_confined
        self.D_directed = D_directed
        self.confinement_radius_px = confinement_radius / self.PX_SCALE
        self.directed_speed_px = directed_speed / self.PX_SCALE  # px/s
        self.blink_rate = blink_rate
        self.recovery_rate = recovery_rate
        self.bleach_rate = bleach_rate

        # Ground-truth params
        self.gt_D_free = D_free
        self.gt_D_confined = D_confined
        self.gt_confinement_radius = confinement_radius
        self.gt_directed_speed = directed_speed

        # Particle populations
        self.n_free = n_free
        self.n_confined = n_confined
        self.n_directed = n_directed
        self.n_total = n_free + n_confined + n_directed
        self._ptype = np.array(
            [0] * n_free + [1] * n_confined + [2] * n_directed, dtype=np.int8
        )

        # Initialize positions (spread across world with margin)
        # For world_size == 64 (matching 100x FOV), initialize across full world.
        # For larger worlds, still use full range — caller can set fov_confined via
        # passing a smaller world_size equal to the FOV.
        margin = 4
        self._px = self.rng.uniform(margin, world_size - margin, self.n_total).astype(np.float64)
        self._py = self.rng.uniform(margin, world_size - margin, self.n_total).astype(np.float64)

        # Confinement corral centers (placed at initial particle position)
        self._corral_cx = self._px.copy()
        self._corral_cy = self._py.copy()

        # Directed: assign angle and oscillation phase
        self._directed_angle = self.rng.uniform(0, 2 * np.pi, self.n_total)
        self._directed_phase = self.rng.uniform(0, 2 * np.pi, self.n_total)

        # Fluorescence state
        self._bright = np.ones(self.n_total, dtype=bool)
        self._bleached = np.zeros(self.n_total, dtype=bool)
        self._intensity = self.rng.uniform(0.75, 1.0, self.n_total)

        # Optical pipeline
        self._pipeline_bf = OpticalPipeline(
            psf_sigma=0.8, noise={"photon_scale": 6.0, "read_std": 2.0},
            vignette=0.05, rng_seed=seed + 100)
        self._pipeline_spt = OpticalPipeline(
            psf_sigma=1.2, noise={"photon_scale": 1.5, "read_std": 1.5},
            vignette=0.02, rng_seed=seed + 200)

    def step(self, dt: float = 0.1) -> None:
        """Advance dynamics by dt seconds (default 100 ms for typical SPT)."""
        if self.fixed_dt > 0:
            dt = self.fixed_dt
        self._time += dt
        self._step_count += 1

        s = self.PX_SCALE  # µm/px

        # Diffusion coefficient → pixel displacement σ
        sig_free = np.sqrt(2 * self.D_free * dt) / s
        sig_conf = np.sqrt(2 * self.D_confined * dt) / s
        sig_dir = np.sqrt(2 * self.D_directed * dt) / s

        for i in range(self.n_total):
            if self._bleached[i]:
                continue

            ptype = int(self._ptype[i])
            noise_x = self.rng.normal(0, 1)
            noise_y = self.rng.normal(0, 1)

            if ptype == 0:  # free diffusion
                self._px[i] += sig_free * noise_x
                self._py[i] += sig_free * noise_y

            elif ptype == 1:  # confined in corral
                nx = self._px[i] + sig_conf * noise_x
                ny = self._py[i] + sig_conf * noise_y
                # Reflect at corral boundary
                dcx = nx - self._corral_cx[i]
                dcy = ny - self._corral_cy[i]
                dist = np.sqrt(dcx**2 + dcy**2)
                if dist > self.confinement_radius_px:
                    # Elastic reflection
                    nx = self._corral_cx[i] + dcx / dist * self.confinement_radius_px * 0.98
                    ny = self._corral_cy[i] + dcy / dist * self.confinement_radius_px * 0.98
                self._px[i] = nx
                self._py[i] = ny
                continue

            else:  # directed: oscillatory transport on cytoskeletal track
                self._directed_phase[i] += dt * 1.5  # ~1.5 rad/s → back-and-forth
                ang = self._directed_angle[i]
                v = self.directed_speed_px * np.sin(self._directed_phase[i]) * dt
                self._px[i] += np.cos(ang) * v + sig_dir * noise_x
                self._py[i] += np.sin(ang) * v + sig_dir * noise_y

            # Reflective world boundary
            self._px[i] = np.clip(self._px[i], 2.0, self.width - 2.0)
            self._py[i] = np.clip(self._py[i], 2.0, self.height - 2.0)

        # Reflect free and directed particles at boundary
        for i in range(self.n_total):
            if self._ptype[i] != 1:
                self._px[i] = np.clip(self._px[i], 2.0, self.width - 2.0)
                self._py[i] = np.clip(self._py[i], 2.0, self.height - 2.0)

        # Blinking: random transitions
        r_blink = self.rng.random(self.n_total)
        r_recov = self.rng.random(self.n_total)
        r_bleach = self.rng.random(self.n_total)
        for i in range(self.n_total):
            if self._bleached[i]:
                continue
            if self._bright[i]:
                if r_blink[i] < self.blink_rate * dt:
                    self._bright[i] = False
                elif r_bleach[i] < self.bleach_rate * dt:
                    self._bleached[i] = True
                    self._bright[i] = False
            else:
                if r_recov[i] < self.recovery_rate * dt:
                    self._bright[i] = True

        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise != 0:
            self.tissue_z += (self.z_drift_rate * dt +
                              self._noise_rng.normal(0, max(self.z_drift_noise * dt, 0)))

    def _update_mode(self):
        """Sync rendering mode from LED/Filter Wheel state devices.

        brightfield (CYAN LED / Electra1) → mode 0
        spt-channel (GREEN LED / TagGFP2) → mode 1
        """
        led = self.state_devices.get("LED", {})
        fw = self.state_devices.get("Filter Wheel", {})
        ll = led.get("label", led.get("Label", "")).upper()
        fl = fw.get("label", fw.get("Label", "")).upper()
        if "GREEN" in ll or "TAGGFP2" in fl:
            self.mode = 1
        else:
            self.mode = 0

    def _update_objectif(self):
        """Sync current_objectiv from state_devices."""
        obj_raw = self.state_devices.get("Objective", 1)
        # state_devices stores either int or dict {'state': '3', 'label': '100x'}
        if isinstance(obj_raw, dict):
            try:
                obj_state = int(obj_raw.get("state", 1))
            except (ValueError, TypeError):
                obj_state = 1
        elif isinstance(obj_raw, int):
            obj_state = obj_raw
        else:
            try:
                obj_state = int(obj_raw)
            except (ValueError, TypeError):
                obj_state = 1
        mags = [10, 20, 40, 100]
        if 0 <= obj_state < len(mags):
            self.current_objectiv = mags[obj_state]
        self._dof = self._dof_table.get(self.current_objectiv, 1.5)

    def snap_frame(self, mask=None, exposure: float = 50.0,
                   intensity: float = 1.0, **kwargs) -> np.ndarray:
        """Render current state. Returns uint8 (H, W, 3) image."""
        self._update_mode()
        self._update_objectif()
        if self.auto_step:
            for _ in range(self.snaps_per_step):
                self.step()
        self._snap_count += 1

        if self.mode == 0:
            img = self._render_bf()
        else:
            img = self._render_spt(exposure=exposure)

        return img

    def _render_bf(self) -> np.ndarray:
        """Wide-field overview: all particles visible but lower contrast."""
        s = self.internal_scale
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)
        spot_sigma = s * 1.2  # wider PSF in widefield
        r_px = max(8, int(spot_sigma * 3.5))

        # Patch-based Gaussian rendering (efficient)
        for i in range(self.n_total):
            if self._bleached[i] or not self._bright[i]:
                continue
            xi = self._px[i] * s
            yi = self._py[i] * s
            brightness = self._intensity[i] * 150.0
            xi_i, yi_i = int(round(xi)), int(round(yi))
            x_lo = max(0, xi_i - r_px)
            x_hi = min(self._iw, xi_i + r_px + 1)
            y_lo = max(0, yi_i - r_px)
            y_hi = min(self._ih, yi_i + r_px + 1)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            px_arr = np.arange(x_lo, x_hi)
            py_arr = np.arange(y_lo, y_hi)
            Xp, Yp = np.meshgrid(px_arr, py_arr)
            r2 = (Xp - xi)**2 + (Yp - yi)**2
            buf[y_lo:y_hi, x_lo:x_hi] += brightness * np.exp(-r2 / (2 * spot_sigma**2))

        # Clip and downsample
        img_hi = np.clip(buf, 0, 255).astype(np.uint8)
        cropped = self._crop_fov_internal(img_hi)
        processed = self._pipeline_bf.apply(cropped.astype(np.float32), exposure_ms=100.0)
        return cv2.merge([processed, processed, processed])

    def _render_spt(self, exposure: float = 50.0) -> np.ndarray:
        """TIRF-like rendering: tight PSF, very low background, high SNR per spot."""
        s = self.internal_scale
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)
        spot_sigma = s * 0.7  # tight PSF (diffraction-limited)

        # Use meshgrid only for particles in rough FOV
        fov_map = {100: 64, 40: 128, 20: 256, 10: self.width}
        fov_px = fov_map.get(self.current_objectiv, self.width)
        cx_world = int(self.camera_offset[0]) + self.viewport_width // 2
        cy_world = int(self.camera_offset[1]) + self.viewport_height // 2
        half = fov_px // 2
        x0_w, x1_w = cx_world - half, cx_world + half
        y0_w, y1_w = cy_world - half, cy_world + half

        for i in range(self.n_total):
            if self._bleached[i] or not self._bright[i]:
                continue
            # Only render particles near FOV (optimization)
            if not (x0_w - 5 <= self._px[i] <= x1_w + 5 and
                    y0_w - 5 <= self._py[i] <= y1_w + 5):
                continue
            xi = self._px[i] * s
            yi = self._py[i] * s
            brightness = self._intensity[i] * 200.0
            # Render in a small patch around particle center (efficient)
            r_px = max(6, int(spot_sigma * 3.5))
            xi_i, yi_i = int(round(xi)), int(round(yi))
            x_lo = max(0, xi_i - r_px)
            x_hi = min(self._iw, xi_i + r_px + 1)
            y_lo = max(0, yi_i - r_px)
            y_hi = min(self._ih, yi_i + r_px + 1)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            px_arr = np.arange(x_lo, x_hi)
            py_arr = np.arange(y_lo, y_hi)
            Xp, Yp = np.meshgrid(px_arr, py_arr)
            r2 = (Xp - xi)**2 + (Yp - yi)**2
            buf[y_lo:y_hi, x_lo:x_hi] += brightness * np.exp(-r2 / (2 * spot_sigma**2))

        img_hi = np.clip(buf, 0, 255).astype(np.uint8)
        cropped = self._crop_fov_internal(img_hi)
        processed = self._pipeline_spt.apply(cropped.astype(np.float32),
                                              exposure_ms=max(10.0, exposure))
        return cv2.merge([processed, processed, processed])

    def _crop_fov_internal(self, img_hi: np.ndarray) -> np.ndarray:
        """Crop internal-resolution buffer to FOV, return viewport-sized image."""
        s = self.internal_scale
        W, H = self.viewport_width, self.viewport_height
        fov_map = {100: 64, 40: 128, 20: 256, 10: self.width}
        fov_px = fov_map.get(self.current_objectiv, self.width)
        fov_int = fov_px * s

        # FOV center in world coords → internal coords
        cx_world = int(self.camera_offset[0]) + W // 2
        cy_world = int(self.camera_offset[1]) + H // 2
        cx_int = cx_world * s
        cy_int = cy_world * s

        half = fov_int // 2
        x0 = max(0, min(cx_int - half, self._iw - fov_int))
        y0 = max(0, min(cy_int - half, self._ih - fov_int))

        crop = img_hi[y0:y0 + fov_int, x0:x0 + fov_int]
        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            canvas = np.zeros((fov_int, fov_int), dtype=np.uint8)
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            crop = canvas

        if crop.shape[0] != H or crop.shape[1] != W:
            interp = cv2.INTER_NEAREST if fov_px < W else cv2.INTER_AREA
            crop = cv2.resize(crop, (W, H), interpolation=interp)
        return crop

    def get_visible_particles(self, at_objective: int = None) -> list:
        """Return list of currently visible particle dicts for scoring."""
        if at_objective is None:
            at_objective = self.current_objectiv
        fov_map = {100: 64, 40: 128, 20: 256, 10: self.width}
        fov_px = fov_map.get(at_objective, self.width)
        W, H = self.viewport_width, self.viewport_height
        cx_world = int(self.camera_offset[0]) + W // 2
        cy_world = int(self.camera_offset[1]) + H // 2
        half = fov_px // 2

        type_names = {0: 'free', 1: 'confined', 2: 'directed'}
        result = []
        for i in range(self.n_total):
            in_fov = (abs(self._px[i] - cx_world) <= half and
                      abs(self._py[i] - cy_world) <= half)
            p = {
                'idx': int(i),
                'x_world': float(self._px[i]),
                'y_world': float(self._py[i]),
                'type': type_names[int(self._ptype[i])],
                'bright': bool(self._bright[i]),
                'bleached': bool(self._bleached[i]),
                'in_fov': in_fov,
            }
            if self._ptype[i] == 1:
                p['corral_cx'] = float(self._corral_cx[i])
                p['corral_cy'] = float(self._corral_cy[i])
            result.append(p)
        return result

    def get_gt_summary(self) -> dict:
        """Ground-truth summary for scoring."""
        n_active = int(np.sum(self._bright & ~self._bleached))
        return {
            'n_free': self.n_free,
            'n_confined': self.n_confined,
            'n_directed': self.n_directed,
            'n_total': self.n_total,
            'n_currently_active': n_active,
            'D_free_um2s': self.gt_D_free,
            'D_confined_um2s': self.gt_D_confined,
            'confinement_radius_um': self.gt_confinement_radius,
            'directed_speed_ums': self.gt_directed_speed,
            'px_scale_um_per_px': self.PX_SCALE,
            'world_size_px': self.width,
            'world_size_um': self.width * self.PX_SCALE,
        }

    def step_autonomous(self, dt: float = 0.1) -> None:
        """Advance without SLM effects (for RealtimeEngine)."""
        self.step(dt)
