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
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class SPTSim(SimBase):
    """Single-particle tracking simulation with multiple diffusion populations."""

    continuous = True

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
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=4, fixed_dt=fixed_dt,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
            },
        )

        self.world_pixel_size_um = 0.1  # 0.1 µm per world pixel (SPT nanoscale)
        self._noise_rng = np.random.default_rng(seed + 9999)
        self._step_count = 0

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

        # Optical pipeline (keyed by mode: 0=BF, 1=SPT/fluorescence)
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.8, noise={"photon_scale": 6.0, "read_std": 2.0},
                vignette=0.05, rng_seed=seed + 100),
            1: OpticalPipeline(
                psf_sigma=1.2, noise={"photon_scale": 1.5, "read_std": 1.5},
                vignette=0.02, rng_seed=seed + 200),
        }

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
        self._accumulate_z_drift(dt)

    def _render_for_mode(self, mode):
        """Return full-resolution single-channel uint8 at internal resolution."""
        if mode == 0:
            return self._render_bf()
        return self._render_spt()

    def _render_bf(self) -> np.ndarray:
        """Wide-field overview: all particles visible but lower contrast.

        Returns single-channel uint8 at internal resolution (_ih x _iw).
        """
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

        return np.clip(buf, 0, 255).astype(np.uint8)

    def _render_spt(self) -> np.ndarray:
        """TIRF-like rendering: tight PSF, very low background, high SNR per spot.

        Returns single-channel uint8 at internal resolution (_ih x _iw).
        """
        s = self.internal_scale
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)
        spot_sigma = s * 0.7  # tight PSF (diffraction-limited)

        # Use meshgrid only for particles in rough FOV
        fov_px = self._FOV_MAP.get(self.current_objectiv, self.width)
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

        return np.clip(buf, 0, 255).astype(np.uint8)

    def _finalize_output(self, viewport: np.ndarray) -> np.ndarray:
        """Return single-channel grayscale output."""
        if viewport.ndim == 3:
            return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)
        return viewport

    def get_visible_particles(self, at_objective: int = None) -> list:
        """Return list of currently visible particle dicts for scoring."""
        if at_objective is None:
            at_objective = self.current_objectiv
        fov_px = self._FOV_MAP.get(at_objective, self.width)
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
