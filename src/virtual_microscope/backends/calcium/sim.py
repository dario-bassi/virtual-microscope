"""
CalciumSim — Calcium wave propagation backend.

A FitzHugh-Nagumo excitable medium simulated on a coarse 128x128 grid
with a Voronoi cell monolayer overlay for realistic GCaMP imaging.
Produces fast-propagating fluorescence waves at ~30-50 rendered px/frame.

Key features:
  - Voronoi cellular structure with per-cell dye loading heterogeneity
  - Nonlinear GCaMP Hill-function response (realistic indicator dynamics)
  - Waves propagate, collide, and annihilate (realistic calcium dynamics)
  - SLM stimulation can TRIGGER waves at any location
  - Recovery period: tissue can't be re-excited immediately after a wave
  - Phase contrast BF with visible cell outlines at high magnification

Usage via SimulationBridge:
    sim = CalciumSim(grid_size=512, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.optical_pipeline import OpticalPipeline


class CalciumSim:
    """FitzHugh-Nagumo excitable medium for calcium wave simulation.

    Simulated on a 128x128 internal grid, rendered with a dense Voronoi
    cell monolayer at 512×512 world resolution (×internal_scale for detail).

    Channels:
      - mode 0: Brightfield — phase contrast with cellular structure
      - mode 1: GCaMP (nucleus channel) — calcium signal (bright = active)
      - mode 2: E-cadherin (membrane channel) — cell boundaries
    """

    # Internal simulation grid (coarse for speed)
    SIM_SIZE = 128

    def __init__(
        self,
        grid_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        seed: int = 42,
        fixed_dt: float = 0.0,
        n_sources: int = 0,
        n_cells: int = 200,
        noise_amplitude: float = 0.005,
        internal_scale: int = 4,
        Du: float = 5.0,
    ):
        self.width = grid_size
        self.height = grid_size
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._scale = grid_size / self.SIM_SIZE  # 4x for 512
        self.nb_cells = n_cells
        self.internal_scale = internal_scale
        self._iw = grid_size * internal_scale
        self._ih = grid_size * internal_scale
        self._seed = seed

        # Camera / state (SimulationBridge interface)
        self.camera_offset = np.array([0.0, 0.0])
        self.focal_plane = 0.0
        self.tissue_z = 0.0
        self.state_devices = {}
        self.mode = 0
        self.current_objectiv = 10
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40, "100x": 100}
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.6}
        self._dof = 6.0
        self._blur_scale_table = {10: 0.3, 20: 0.5, 40: 1.0, 100: 2.0}
        self._extra_channels = {}
        self._snap_count = 0

        # FitzHugh-Nagumo parameters (on coarse grid)
        self.Du = float(Du)     # diffusion → wave speed
        self.epsilon = 0.03     # recovery rate
        self.a = 0.5            # excitability threshold
        self.b = 0.8            # recovery coupling
        self.pde_dt = 0.005     # PDE timestep
        self.steps_per_snap = 1000  # PDE steps per evolution

        self.fixed_dt = fixed_dt
        self.noise_amplitude = noise_amplitude
        self.rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 7777)

        # Inexcitable mask: must be set before _evolve() is called.
        # Set via set_inexcitable_mask(mask) after initialization.
        self._inexcitable_mask = None

        # Initialize fields (coarse grid)
        N = self.SIM_SIZE
        self.u = np.full((N, N), -1.0, dtype=np.float64)  # resting
        self.w = np.full((N, N), -0.6, dtype=np.float64)  # resting
        self._time = 0.0

        # Seed initial wave sources (on coarse grid)
        # n_sources < 0: use random count (2-4); n_sources == 0: no initial sources
        if n_sources < 0:
            n_sources = self.rng.integers(2, 5)
        for _ in range(n_sources):
            cx = self.rng.integers(20, N - 20)
            cy = self.rng.integers(20, N - 20)
            r = self.rng.integers(3, 7)
            Y, X = np.ogrid[:N, :N]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
            self.u[mask] = 1.5

        # Evolve briefly to get initial waves propagating
        self._evolve(500)

        # ── Dense cell layout (Voronoi overlay) ──
        self._generate_cells(n_cells, grid_size)
        self._cell_pde_y = np.clip(
            (self.centers[:, 1] / self._scale).astype(int), 0, N - 1)
        self._cell_pde_x = np.clip(
            (self.centers[:, 0] / self._scale).astype(int), 0, N - 1)
        self._dye_loading = 0.5 + 0.5 * self.rng.random(n_cells)
        self._cell_response_amp = 0.7 + 0.3 * self.rng.random(n_cells)
        self._cell_intensities = np.zeros(n_cells)
        self._build_voronoi()

        # Optogenetic stimulation
        self._stim_mask = None        # bool array on SIM_SIZE grid
        self._stim_strength = 2.0     # supra-threshold stimulus
        self._inhibit_strength = 2.5  # sub-threshold clamp for wave blocking
        self._slm_mode = 0            # 0 = excite, 1 = inhibit

        # Pacemaker cells: spontaneous wave emission at intervals.
        # Set add_pacemakers() to configure pacemaker regions.
        self._pacemaker_positions = []   # list of (cx, cy, radius) on coarse grid
        self._pacemaker_period = 150.0   # dt-units between firings (default)
        self._pacemaker_strength = 1.5   # stimulus intensity (supra-threshold)
        self._pacemaker_accum = 0.0      # accumulated dt for time-based firing
        # Per-pacemaker config: list of dicts with 'period', 'accum', 'strength'
        # When populated, overrides global _pacemaker_period/_pacemaker_strength
        self._pacemaker_configs = []

        # Auto-step
        self.auto_step = True
        self.snaps_per_step = 1

        # Stage drift (interface compat)
        self.stage_drift_rate = 0.0
        self.stage_drift_noise = 0.0
        self._drift_accumulator = np.array([0.0, 0.0])

        # Z-drift
        self.z_drift_rate = 0.0   # µm/s
        self.z_drift_noise = 0.0  # σ of z-jitter (µm·s⁻½, Brownian)

        # Optical pipelines per channel
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.5, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 300),
            1: OpticalPipeline(
                psf_sigma=0.8, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 301),
            2: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 6.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 302),
        }

        # Default pacemakers: ensure tissue always has activity
        self.add_pacemakers(n=3, period=150)

    def _laplacian(self, Z: np.ndarray) -> np.ndarray:
        """Discrete Laplacian with no-flux boundary conditions."""
        padded = np.pad(Z, 1, mode='edge')
        return (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:] -
            4 * Z
        )

    def _evolve(self, n_steps: int):
        """Evolve the FitzHugh-Nagumo system on the coarse grid with noise."""
        u, w = self.u, self.w
        Du, eps, a, b = self.Du, self.epsilon, self.a, self.b
        dt = self.pde_dt
        noise_amp = self.noise_amplitude
        sqrt_dt = np.sqrt(dt)
        inex = self._inexcitable_mask

        for _ in range(n_steps):
            lu = self._laplacian(u)
            u += dt * (Du * lu + u - u ** 3 / 3.0 - w)
            w += dt * eps * (u + a - b * w)

            # Stochastic current injection (spontaneous calcium release)
            if noise_amp > 0:
                u += noise_amp * self._noise_rng.normal(0, 1, u.shape) * sqrt_dt

            # Clamp inexcitable regions to rest (absorbing barrier)
            if inex is not None:
                u[inex] = -1.0
                w[inex] = -0.6

            np.clip(u, -2.5, 2.5, out=u)
            np.clip(w, -2.0, 2.0, out=w)

        self.u, self.w = u, w

    def _evolve_with_stim(self, n_steps: int):
        """Evolve with SLM stimulation: inject or suppress current where mask is active."""
        u, w = self.u, self.w
        Du, eps, a, b = self.Du, self.epsilon, self.a, self.b
        dt = self.pde_dt
        mask_bool = self._stim_mask

        if self._slm_mode == 0:
            # EXCITE: inject positive current → nucleate waves
            stim = mask_bool.astype(np.float64) * self._stim_strength
            for _ in range(n_steps):
                lu = self._laplacian(u)
                u += dt * (Du * lu + u - u ** 3 / 3.0 - w + stim)
                w += dt * eps * (u + a - b * w)
                np.clip(u, -2.5, 2.5, out=u)
                np.clip(w, -2.0, 2.0, out=w)
        else:
            # INHIBIT: clamp u to resting state in masked region → block waves
            # This creates a "wall" that absorbs incoming wavefronts
            rest_u = -1.0
            rest_w = -0.6
            inex = self._inexcitable_mask
            for _ in range(n_steps):
                lu = self._laplacian(u)
                u += dt * (Du * lu + u - u ** 3 / 3.0 - w)
                w += dt * eps * (u + a - b * w)
                # Force resting state in inhibited region
                u[mask_bool] = rest_u
                w[mask_bool] = rest_w
                # Clamp inexcitable barriers
                if inex is not None:
                    u[inex] = -1.0
                    w[inex] = -0.6
                np.clip(u, -2.5, 2.5, out=u)
                np.clip(w, -2.0, 2.0, out=w)

        self.u, self.w = u, w

    def set_inexcitable_mask(self, mask: np.ndarray):
        """Set regions that cannot propagate calcium waves (physical barriers).

        These regions are clamped to resting state after each PDE micro-step,
        creating absorbing boundaries between excitable territories.  Useful
        for modelling tissue gaps, scars, or isolated cell islands.

        Args:
            mask: Boolean array of shape (SIM_SIZE, SIM_SIZE) on the coarse
                  grid (128×128 by default).  True = inexcitable.
        """
        self._inexcitable_mask = mask.astype(bool)
        # Initialize inexcitable regions to rest immediately
        self.u[self._inexcitable_mask] = -1.0
        self.w[self._inexcitable_mask] = -0.6

    def add_pacemakers(self, n: int = 3, period: float = 150.0,
                       strength: float = 1.5):
        """Add spontaneous pacemaker cells for homeostatic wave emission.

        Args:
            n: Number of pacemaker regions.
            period: Time between firings in dt-units (default 150 = every
                150 step(dt=1.0) calls, or equivalently 1500 step(dt=0.1) calls).
            strength: Stimulus intensity per firing.
        """
        self._pacemaker_period = float(period)
        self._pacemaker_strength = strength
        self._pacemaker_positions = []
        self._pacemaker_configs = []
        N = self.SIM_SIZE
        for _ in range(n):
            cx = int(self.rng.integers(15, N - 15))
            cy = int(self.rng.integers(15, N - 15))
            r = int(self.rng.integers(3, 6))
            self._pacemaker_positions.append((cx, cy, r))

    def add_pacemaker(self, cx: int, cy: int, radius: int = 5,
                      period: float = 10.0, strength: float = 2.0):
        """Add a single pacemaker with its own period and strength.

        Use this for per-pacemaker frequency control (e.g., competing
        pacemakers at different frequencies for wave blocking challenges).
        """
        self._pacemaker_positions.append((cx, cy, radius))
        self._pacemaker_configs.append({
            "period": float(period),
            "strength": float(strength),
            "accum": 0.0,
        })

    # ── Cell layout and Voronoi tessellation ──

    def _generate_cells(self, n_cells, grid_size):
        """Generate dense packed cell layout using jittered grid."""
        side = int(np.ceil(np.sqrt(n_cells * 1.1)))
        spacing = grid_size / side
        self.cell_radius = spacing * 0.45

        candidates = []
        for row in range(side):
            for col in range(side):
                x = (col + 0.5) * spacing + self.rng.uniform(-spacing * 0.15, spacing * 0.15)
                y = (row + 0.5) * spacing + self.rng.uniform(-spacing * 0.15, spacing * 0.15)
                x = np.clip(x, 2, grid_size - 2)
                y = np.clip(y, 2, grid_size - 2)
                candidates.append([x, y])

        candidates = np.array(candidates)
        if len(candidates) > n_cells:
            idx = self.rng.choice(len(candidates), n_cells, replace=False)
            self.centers = candidates[idx]
        else:
            self.centers = candidates[:n_cells]
            self.nb_cells = len(self.centers)

    def _build_voronoi(self):
        """Precompute Voronoi cell labels, boundaries, and nucleus mask."""
        s = self.internal_scale
        h_w, w_w = self.height, self.width
        h_i, w_i = self._ih, self._iw

        # Label map at world resolution (chunked for memory)
        yy, xx = np.mgrid[:h_w, :w_w]
        cx = self.centers[:, 0].astype(np.float32)
        cy = self.centers[:, 1].astype(np.float32)

        labels_world = np.zeros((h_w, w_w), dtype=np.int32)
        chunk = 64
        for y0 in range(0, h_w, chunk):
            y1 = min(y0 + chunk, h_w)
            dy = yy[y0:y1, :, None] - cy[None, None, :]
            dx = xx[y0:y1, :, None] - cx[None, None, :]
            labels_world[y0:y1] = np.argmin(dy ** 2 + dx ** 2, axis=2)

        # Upscale to internal resolution
        if s > 1:
            self._cell_labels = cv2.resize(
                labels_world.astype(np.float32), (w_i, h_i),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        else:
            self._cell_labels = labels_world

        # Boundary mask (pixels where neighbors differ)
        shifted_r = np.roll(self._cell_labels, 1, axis=1)
        shifted_d = np.roll(self._cell_labels, 1, axis=0)
        self._boundary_mask = (
            (self._cell_labels != shifted_r) | (self._cell_labels != shifted_d)
        )

        # Nucleus mask: small oval per cell
        nuc_mask = np.zeros((h_i, w_i), dtype=np.float32)
        nuc_radius = 4.0  # world px
        for ci in range(self.nb_cells):
            ncx = int(self.centers[ci, 0] * s)
            ncy = int(self.centers[ci, 1] * s)
            r = int((nuc_radius + self.rng.normal(0, 0.5)) * s)
            r = max(2 * s, r)
            cv2.circle(nuc_mask, (ncx, ncy), r, 1.0, -1)
        self._nuc_mask = nuc_mask

        # Phase contrast halo: gradient bright fringe flanking boundaries
        boundary_u8 = self._boundary_mask.astype(np.uint8)
        dist_from_boundary = cv2.distanceTransform(
            1 - boundary_u8, cv2.DIST_L2, 3,
        )
        halo_width = 3.0 * s
        halo_falloff = np.clip(1.0 - dist_from_boundary / halo_width, 0, 1)
        halo_falloff[self._boundary_mask] = 0  # boundary is dark, not halo
        self._halo_gradient = (halo_falloff ** 1.2 * 45.0).astype(np.float32)

        # Subcellular texture (organelle granularity)
        tex_rng = np.random.default_rng(self._seed + 555)
        self._bf_texture = tex_rng.normal(0, 2.0, (h_i, w_i)).astype(np.float32)

        # Cytoplasmic gradient: distance from cell center → thicker cytoplasm near center
        # Vectorized: compute distance of each pixel to its cell center
        cy_all = (self.centers[:, 1] * s).astype(np.float32)
        cx_all = (self.centers[:, 0] * s).astype(np.float32)
        yy_i = np.arange(h_i, dtype=np.float32)
        xx_i = np.arange(w_i, dtype=np.float32)
        # Distance of each pixel to its assigned cell center
        dy = yy_i[:, None] - cy_all[self._cell_labels]
        dx = xx_i[None, :] - cx_all[self._cell_labels]
        dist_map = np.sqrt(dy ** 2 + dx ** 2)
        # Approximate max distance per cell using cell spacing
        avg_spacing = self.width * s / np.sqrt(self.nb_cells)
        norm_dist = np.clip(dist_map / (avg_spacing * 0.6), 0, 1)
        # Invert: 1.0 near center, 0.7 at boundary (thicker cytoplasm = more signal)
        self._cyto_gradient = (1.0 - 0.3 * norm_dist).astype(np.float32)

    def _update_intensities(self):
        """Sample GCaMP field at each cell's position with Hill function."""
        u_vals = self.u[self._cell_pde_y, self._cell_pde_x]
        # Map FHN u (range ~ -1 to 2) to calcium [0, 1]
        ca = np.clip((u_vals + 1.0) / 2.5, 0, 1)
        # Hill function: GCaMP6s Kd ≈ 0.35, n ≈ 3.0 (wider dynamic range)
        kd, n_hill = 0.35, 3.0
        ca_n = ca ** n_hill
        gcamp = ca_n / (kd ** n_hill + ca_n)
        self._cell_intensities = np.clip(
            gcamp * self._dye_loading * self._cell_response_amp, 0, 1)

    def _fire_pacemakers(self):
        """Inject current at pacemaker locations (all fire simultaneously)."""
        N = self.SIM_SIZE
        Y, X = np.ogrid[:N, :N]
        for cx, cy, r in self._pacemaker_positions:
            mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
            self.u[mask] = np.maximum(self.u[mask],
                                      self._pacemaker_strength)

    def _fire_pacemaker_individual(self, idx: int):
        """Fire a single pacemaker by index."""
        N = self.SIM_SIZE
        Y, X = np.ogrid[:N, :N]
        cx, cy, r = self._pacemaker_positions[idx]
        cfg = self._pacemaker_configs[idx]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
        self.u[mask] = np.maximum(self.u[mask], cfg["strength"])

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 37.0  # mammalian default
        return float(self.state_devices["Temperature"].get("label", "37"))

    def _temp_rate_factor(self) -> float:
        """Temperature-dependent rate scaling for calcium dynamics.

        Q10 ~ 2.0 for ion channel kinetics. Optimal at 37°C.
        Cold slows wave propagation, heat accelerates then denatures.
        """
        temp = self._get_temperature()
        if temp < 10:
            return 0.05
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.05, 1.0 - (temp - 42) * 0.3)
        return factor

    def step(self, dt: float = 1.0):
        """Advance simulation, scaling PDE steps proportional to dt."""
        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        temp_factor = self._temp_rate_factor()
        n_steps = max(1, round(dt * self.steps_per_snap * temp_factor))

        # Spontaneous pacemaker firing (time-based, not call-count)
        self._advance_pacemakers(dt)

        if self._stim_mask is not None and np.any(self._stim_mask):
            self._evolve_with_stim(n_steps)
        else:
            self._evolve(n_steps)
        self._time += n_steps * self.pde_dt

    def _advance_pacemakers(self, dt: float):
        """Fire pacemakers based on accumulated time.

        Supports both global period (all fire together) and per-pacemaker
        periods (each has its own config with independent accumulator).
        """
        if not self._pacemaker_positions:
            return

        if self._pacemaker_configs:
            # Per-pacemaker firing with individual periods
            for i, cfg in enumerate(self._pacemaker_configs):
                cfg["accum"] += dt
                while cfg["accum"] >= cfg["period"]:
                    cfg["accum"] -= cfg["period"]
                    self._fire_pacemaker_individual(i)
        else:
            # Global period: all fire simultaneously
            self._pacemaker_accum += dt
            while self._pacemaker_accum >= self._pacemaker_period:
                self._pacemaker_accum -= self._pacemaker_period
                self._fire_pacemakers()

    def step_autonomous(self, dt: float = 1.0):
        """Advance PDE WITH continuous SLM optogenetic effects.

        Used by RealtimeEngine for background dynamics. SLM stimulation
        (excite/inhibit) is continuous illumination and must be applied
        during background stepping, not just during snap_frame().
        """
        # Read SLM-Mode from state_devices (updated by core.setState)
        if "SLM-Mode" in self.state_devices:
            mode_dev = self.state_devices["SLM-Mode"]
            label = mode_dev.get("Label", mode_dev.get("label", "excite"))
            self._slm_mode = 1 if label == "inhibit" else 0

        temp_factor = self._temp_rate_factor()
        n_steps = max(1, round(dt * self.steps_per_snap * temp_factor))

        self._advance_pacemakers(dt)

        if self._stim_mask is not None and np.any(self._stim_mask):
            self._evolve_with_stim(n_steps)
        else:
            self._evolve(n_steps)
        self._time += n_steps * self.pde_dt

    # ── Rendering (Voronoi cellular overlay on PDE) ──

    def _upscale_pde(self, field: np.ndarray) -> np.ndarray:
        """Upscale a coarse PDE field to internal resolution."""
        return cv2.resize(
            field.astype(np.float32),
            (self._iw, self._ih),
            interpolation=cv2.INTER_LINEAR,
        )

    def _render_bf_full(self) -> np.ndarray:
        """Brightfield: phase contrast with cellular structure.

        Dense confluent monolayer.  Cells darken during calcium transients.
        Phase contrast halos flank the intercalated-disc boundaries.
        """
        h, w = self._ih, self._iw

        # Update per-cell intensities
        self._update_intensities()

        # Per-cell body: darken during calcium transient
        cell_vals = 120.0 - 25.0 * self._cell_intensities
        body = cell_vals[self._cell_labels].astype(np.float32)

        # Nuclear shade-off (lighter ovals)
        body += self._nuc_mask * 12.0

        # Phase contrast halo: gradient bright fringe
        body += self._halo_gradient

        # Contraction-coupled halo brightening
        px_calcium = self._cell_intensities[self._cell_labels]
        body += self._halo_gradient * px_calcium * 0.25

        # Intercalated discs: dark flat lines at cell-cell boundaries
        body[self._boundary_mask] = 85.0

        # Organelle texture
        body += self._bf_texture

        body = np.clip(body, 0, 255).astype(np.uint8)
        return np.stack([body, body, body], axis=2)

    def _render_nuc_full(self) -> np.ndarray:
        """GCaMP channel: calcium wave as cellular fluorescence.

        Smooth PDE wavefront modulated by per-cell dye loading.
        Hill function gives realistic GCaMP6s dynamics (Kd=0.35, n=3).
        Dark resting baseline (~3 counts), bright active (~220 counts).
        Subcellular gradient: brighter near nucleus (thicker cytoplasm).
        OOF haze: faint diffuse glow from out-of-focus planes.
        """
        h, w = self._ih, self._iw

        # Update per-cell intensities (Hill function applied)
        self._update_intensities()

        # Upscale PDE field for smooth wavefront
        u_up = self._upscale_pde(self.u)
        # Map u to calcium [0,1]
        ca_field = np.clip((u_up + 1.0) / 2.5, 0, 1)
        # Hill function on the full field (match _update_intensities params)
        kd, n_hill = 0.35, 3.0
        ca_n = ca_field ** n_hill
        gcamp_field = ca_n / (kd ** n_hill + ca_n)

        # Per-pixel dye loading and response amplitude from cell labels
        dye = self._dye_loading[self._cell_labels]
        resp = self._cell_response_amp[self._cell_labels]

        # Blend per-cell + smooth field for natural wavefront
        per_cell = self._cell_intensities[self._cell_labels]
        smooth_signal = gcamp_field * dye * resp
        blended = 0.7 * per_cell + 0.3 * smooth_signal

        # Baseline (dim — resting cells nearly dark) + calcium signal
        base = 3.0
        img_f = base + blended * 220.0

        # Subcellular gradient: brighter near cell center (thicker cytoplasm)
        img_f *= self._cyto_gradient

        # Cell boundary dimming (gap junctions attenuate signal)
        img_f[self._boundary_mask] -= 4.0

        # Nuclear exclusion: GCaMP is cytoplasmic, nuclei appear as dark voids
        nuc_dim = self._nuc_mask * 0.7
        img_f *= (1.0 - nuc_dim)
        np.clip(img_f, 0, 255, out=img_f)

        # OOF haze: diffuse glow from out-of-focus planes
        s = self.internal_scale
        haze_sigma = max(15.0 * s, 5.0)
        haze = cv2.GaussianBlur(img_f, (0, 0), haze_sigma)
        img_f = img_f * 0.88 + haze * 0.12

        np.clip(img_f, 0, 255, out=img_f)
        gcamp_img = img_f.astype(np.uint8)
        return np.stack([gcamp_img, gcamp_img, gcamp_img], axis=2)

    def _render_mem_full(self) -> np.ndarray:
        """Membrane channel: cell boundaries (E-cadherin / connexin-43)."""
        h, w = self._ih, self._iw

        # Faint cytoplasm + bright boundaries
        membrane = np.full((h, w), 10, dtype=np.uint8)
        membrane[self._boundary_mask] = 155

        return np.stack([membrane, membrane, membrane], axis=2)

    # ── Internal scale helpers ──

    def _crop_fov(self, full_img):
        """Crop field-of-view from internal-res image and resize to viewport."""
        s = self.internal_scale
        fov_map = {10: 512, 20: 256, 40: 128, 100: 64}
        fov_world = fov_map.get(self.current_objectiv, 512)
        fov_int = fov_world * s

        # Center FOV on stage position at any magnification
        cx_world = int(self.camera_offset[0]) + self.viewport_width // 2
        cy_world = int(self.camera_offset[1]) + self.viewport_height // 2
        ox = cx_world * s - fov_int // 2
        oy = cy_world * s - fov_int // 2

        ih, iw = full_img.shape[:2]
        ox = max(0, min(ox, iw - fov_int))
        oy = max(0, min(oy, ih - fov_int))

        crop = full_img[oy:oy + fov_int, ox:ox + fov_int]

        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            bg = 140 if self.mode == 0 else 0
            if len(full_img.shape) == 3:
                padded = np.full((fov_int, fov_int, 3), bg, dtype=crop.dtype)
            else:
                padded = np.full((fov_int, fov_int), bg, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        out_w, out_h = self.viewport_width, self.viewport_height
        interp = cv2.INTER_AREA if fov_int > out_w else cv2.INTER_LINEAR
        return cv2.resize(crop, (out_w, out_h), interpolation=interp)

    # ── snap_frame ──

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()

        # Read SLM-Mode device if present (0=excite, 1=inhibit)
        if "SLM-Mode" in self.state_devices:
            mode_dev = self.state_devices["SLM-Mode"]
            label = mode_dev.get("Label", mode_dev.get("label", "excite"))
            self._slm_mode = 1 if label == "inhibit" else 0

        # Handle SLM mask: downsample to coarse grid
        if mask is not None and np.any(mask):
            coarse = cv2.resize(
                mask.astype(np.uint8), (self.SIM_SIZE, self.SIM_SIZE),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            self._stim_mask = coarse
        else:
            self._stim_mask = None

        # Step at start of each snap cycle
        if (self.auto_step and self._snap_count > 0
                and self._snap_count % self.snaps_per_step == 0):
            self.step()

        self._snap_count += 1

        # Render at internal resolution
        if self.mode == 0:
            full_img = self._render_bf_full()
        elif self.mode == 1:
            full_img = self._render_nuc_full()
        elif self.mode == 2:
            full_img = self._render_mem_full()
        elif self.mode in self._extra_channels:
            full_img = self._extra_channels[self.mode]["image"]
        else:
            full_img = self._render_bf_full()

        # Crop FOV and resize to viewport
        viewport = self._crop_fov(full_img)
        viewport = self._apply_defocus(viewport)

        if self.mode in self._pipeline:
            pipe = self._pipeline[self.mode]
            if self.mode > 0 and pipe.photobleach_rate > 0:
                viewport = pipe.apply_with_bleach(viewport, exposure_ms=exposure)
            else:
                viewport = pipe.apply(viewport, exposure_ms=exposure)

        # Exposure scaling (BF uses 2× base for transmitted light)
        if self.mode == 0:
            scale = min(intensity * 0.02 * exposure, 2.0)
        else:
            scale = intensity * 0.01 * exposure
        viewport = (viewport.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)

        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        dz = abs(self.focal_plane - self.tissue_z)
        half_dof = self._dof / 2.0
        if dz <= half_dof:
            return img
        sigma = min((dz - half_dof) * self._blur_scale_table.get(
            self.current_objectiv, 0.5), 30.0)
        if sigma < 0.3:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigma)

    def _update_mode(self):
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            self.mode = 0
            return
        filt = self.state_devices["Filter Wheel"]
        led = self.state_devices["LED"]
        fl = filt.get("Label", filt.get("label", ""))
        ll = led.get("Label", led.get("label", "CYAN"))
        if fl == "mScarlet3(569/582)" and ll == "ORANGE":
            self.mode = 1
        elif fl == "miRFP670(642/670)" and ll == "RED":
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
        lbl = obj.get("Label", obj.get("label", ""))
        if lbl in self._objectif_dict:
            self.current_objectiv = self._objectif_dict[lbl]
            self._dof = self._dof_table.get(self.current_objectiv, 6.0)

    def set_focal_plane(self, z: float):
        self.focal_plane = z

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift (µm)."""
        return self.tissue_z

    def reset_z_drift(self):
        """Reset Z-drift to zero."""
        self.tissue_z = 0.0

    def enable_photobleaching(self, rate: float = 0.001):
        """Enable photobleaching on fluorescence channels."""
        for mode in [1, 2]:
            if mode in self._pipeline:
                self._pipeline[mode].photobleach_rate = rate

    def reset_photobleaching(self):
        """Reset accumulated photobleaching."""
        for pipe in self._pipeline.values():
            pipe.reset_bleach()

    def update_state(self, dict_state: dict):
        self.state_devices = dict_state

    def update(self, dt: float = 0.016):
        self.step()

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        N = self.SIM_SIZE
        self.u = np.full((N, N), -1.0, dtype=np.float64)
        self.w = np.full((N, N), -0.6, dtype=np.float64)
        self._snap_count = 0
        self._time = 0.0

    # ── Ground truth helpers ──

    def get_wave_coverage(self, threshold: float = 0.0) -> float:
        """Fraction of field with active calcium signal (u > threshold)."""
        return round(float((self.u > threshold).mean()), 4)

    def get_wave_front(self, threshold: float = 0.5) -> list:
        """Get positions of wavefront pixels (leading edge of active region).

        Returns list of (x, y) in world coordinates where the wave is actively
        rising (u > threshold and gradient is positive).
        """
        # Find active region on coarse grid
        active = self.u > threshold
        # Erode to find interior, then subtract to get boundary
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(active)
        front = active & ~interior
        fy, fx = np.nonzero(front)
        # Scale to world coordinates
        scale = self._scale
        return [(round(float(x * scale), 1), round(float(y * scale), 1))
                for x, y in zip(fx, fy)]

    def get_ground_truth(self) -> dict:
        front = self.get_wave_front()
        return {
            "wave_coverage": self.get_wave_coverage(),
            "mean_u": round(float(self.u.mean()), 4),
            "max_u": round(float(self.u.max()), 4),
            "grid_size": self.width,
            "sim_size": self.SIM_SIZE,
            "time": round(self._time, 2),
            "slm_mode": "inhibit" if self._slm_mode == 1 else "excite",
            "n_wavefront_pixels": len(front),
        }
