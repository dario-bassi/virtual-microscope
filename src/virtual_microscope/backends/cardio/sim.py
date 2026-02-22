"""
CardioSim — Beating cardiomyocyte monolayer with wave propagation.

Dense confluent tissue with excitable-wave cellular automaton on a
128×128 grid, overlaid with ~300 packed cells for rendering.
Produces visible propagating calcium wavefronts (GCaMP-like signal).

Key features:
  - Planar calcium waves from a primary pacemaker site
  - Arrhythmic ectopic focus with faster firing rate
  - SLM optogenetic pacing: illuminate region to trigger waves
  - Drug response: frequency, coupling, amplitude modifiers
  - Dense tissue rendering: Voronoi-like at high magnification

Wave model: 3-state cellular automaton (resting → excited → refractory)
with nearest-neighbor propagation.  GCaMP fluorescence accumulates
during excitation and decays exponentially, giving wide bright bands.

Usage via SimulationBridge:
    sim = CardioSim(grid_size=512, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class CardioSim(SimBase):
    """Beating cardiomyocyte monolayer with FHN-based wave propagation.

    A confluent monolayer of ~300 cardiomyocytes coupled by gap junctions.
    Calcium dynamics on a 128×128 PDE grid produce visible propagating
    wavefronts. Each cell samples its intensity from the calcium field.

    Channels:
      - mode 0: Brightfield — dense cell outlines, contraction visible
      - mode 1: GCaMP (nucleus channel) — calcium transient (bright = systole)
      - mode 2: Membrane channel — cell boundaries (gap junctions)
    """

    continuous = True

    # PDE grid size (coarse for speed)
    SIM_SIZE = 128

    def __init__(
        self,
        grid_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_cells: int = 300,
        seed: int = 42,
        fixed_dt: float = 0.0,
        normal_freq: float = 1.0,
        arrhythmia_freq: float = 1.8,
        arrhythmia_fraction: float = 0.12,
        coupling_strength: float = 2.0,
        internal_scale: int = 4,
    ):
        super().__init__(
            width=grid_size, height=grid_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=fixed_dt,
            auto_step=True, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,      # GCaMP
                ("mScarlet3(569/582)", "ORANGE"): 2,   # cell-junctions
            },
        )

        self.n_cells = n_cells
        self._noise_rng = np.random.default_rng(seed + 7777)
        self._seed = seed
        self.normal_freq = normal_freq
        self.arrhythmia_freq = arrhythmia_freq
        self.coupling_strength = coupling_strength

        # ── Dense cell layout (Poisson-disk-like) ──
        self._generate_cells(n_cells, grid_size)

        # ── Excitable-wave cellular automaton ──
        N = self.SIM_SIZE
        self._pde_scale = grid_size / N  # world px per PDE pixel (4.0)

        # Automaton parameters
        self._APD = 4           # excited for 4 steps
        self._RRP = 8           # refractory for 8 more steps (min cycle = 12)
        self.steps_per_snap = 20  # automaton steps per step()

        # State field: 0=resting, 1.._APD=excited, _APD+1.._APD+_RRP=refractory
        self._ca_state = np.zeros((N, N), dtype=np.int32)

        # GCaMP fluorescence (accumulates during excitation, decays)
        self._gcamp = np.zeros((N, N), dtype=np.float64)
        self._tau_gcamp = 40.0    # decay time constant (steps)
        self._alpha_gcamp = 0.25  # accumulation rate per excited step
        self._gcamp_decay = np.exp(-1.0 / self._tau_gcamp)

        # Backward-compat aliases (some code reads self.u / self.Du)
        self.Du = 1.0   # nominal diffusion (for GT/drug interface)
        self.pde_dt = 0.005  # nominal PDE dt (for GT calculations)

        # Pacemaker sites (on automaton grid)
        # Primary pacemaker: cluster at bottom-left quadrant
        self._pacemaker_main = (N // 4, N // 4)
        self._pacemaker_radius = 6
        # Period in automaton steps: steps_per_snap * frames_per_beat
        # At 1 Hz with 10 frames/s (steps_per_snap=20 per step, step=1 frame):
        # pm_period = steps_per_snap / normal_freq * 10  -- but simplified:
        # We want PM to fire every N steps → frequency = steps_per_snap/pm_period Hz
        self._pacemaker_period_steps = round(self.steps_per_snap * 10.0 / normal_freq)
        self._pacemaker_accum = 0

        # Arrhythmic ectopic focus: cluster at top-right
        n_arrhythmic = max(1, int(n_cells * arrhythmia_fraction))
        self._ectopic_center = (3 * N // 4, 3 * N // 4)
        self._ectopic_radius = 4
        self._ectopic_period_steps = round(self.steps_per_snap * 10.0 / arrhythmia_freq)
        self._ectopic_accum = round(self._ectopic_period_steps * 0.3)
        self._ectopic_active = True

        # Build pacemaker/ectopic masks on grid
        Y, X = np.ogrid[:N, :N]
        self._pm_mask = (
            (X - self._pacemaker_main[0]) ** 2
            + (Y - self._pacemaker_main[1]) ** 2
            < self._pacemaker_radius ** 2
        )
        self._ec_mask = (
            (X - self._ectopic_center[0]) ** 2
            + (Y - self._ectopic_center[1]) ** 2
            < self._ectopic_radius ** 2
        )

        # Backward-compat period properties (in seconds for GT)
        self._pacemaker_period = 1.0 / normal_freq
        self._ectopic_period = 1.0 / arrhythmia_freq

        # Mark arrhythmic cells (those near ectopic focus)
        ectopic_world = (
            self._ectopic_center[0] * self._pde_scale,
            self._ectopic_center[1] * self._pde_scale,
        )
        dist_to_ectopic = np.sqrt(
            (self.centers[:, 0] - ectopic_world[0]) ** 2
            + (self.centers[:, 1] - ectopic_world[1]) ** 2
        )
        sorted_idx = np.argsort(dist_to_ectopic)
        self._is_arrhythmic = np.zeros(n_cells, dtype=bool)
        self._is_arrhythmic[sorted_idx[:n_arrhythmic]] = True

        # SLM stimulation
        self._stim_mask = None  # bool array on SIM_SIZE grid
        self._stim_strength = 2.0

        # Cell intensity cache
        self._cell_intensities = np.zeros(n_cells)
        # Per-cell dye loading variation (heterogeneous baseline)
        self._dye_loading = 0.6 + 0.4 * self.rng.random(n_cells)

        # Drug response
        self._drug_active = False
        self._drug_name = None
        self._drug_effect = 0.0
        self._drug_onset_rate = 0.05
        self._drug_washout_rate = 0.03
        self._drug_washing_out = False
        self._base_Du = self.Du
        self._base_pacemaker_period = self._pacemaker_period
        self._base_ectopic_period = self._ectopic_period
        self._base_pm_period_steps = self._pacemaker_period_steps
        self._base_ec_period_steps = self._ectopic_period_steps
        self._drug_profiles = {
            "isoproterenol": {"freq_mult": 1.5, "Du_mult": 1.0, "amplitude_mult": 1.0},
            "verapamil": {"freq_mult": 0.7, "Du_mult": 0.6, "amplitude_mult": 0.8},
            "lidocaine": {"freq_mult": 1.0, "Du_mult": 0.4, "amplitude_mult": 1.0},
            "caffeine": {"freq_mult": 1.3, "Du_mult": 0.85, "amplitude_mult": 1.1},
        }
        self._drug_amplitude_mult = 1.0

        # Cell-to-PDE coordinate mapping (precomputed)
        self._cell_pde_y = np.clip(
            (self.centers[:, 1] / self._pde_scale).astype(int), 0, N - 1)
        self._cell_pde_x = np.clip(
            (self.centers[:, 0] / self._pde_scale).astype(int), 0, N - 1)

        # Pre-run to establish wave pattern (100 frames = 2000 steps)
        self._evolve(2000)

        # Precompute Voronoi tessellation for rendering
        self._build_voronoi()

        # Optical pipelines
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

    # ── Cell generation ──

    def _generate_cells(self, n_cells, grid_size):
        """Generate dense packed cell layout using jittered grid."""
        side = int(np.ceil(np.sqrt(n_cells * 1.1)))
        spacing = grid_size / side
        self.cell_radius = spacing * 0.45  # nearly touching

        candidates = []
        for row in range(side):
            for col in range(side):
                x = (col + 0.5) * spacing + self.rng.uniform(-spacing * 0.15, spacing * 0.15)
                y = (row + 0.5) * spacing + self.rng.uniform(-spacing * 0.15, spacing * 0.15)
                x = np.clip(x, 2, grid_size - 2)
                y = np.clip(y, 2, grid_size - 2)
                candidates.append([x, y])

        candidates = np.array(candidates)
        # Randomly select n_cells from candidates
        if len(candidates) > n_cells:
            idx = self.rng.choice(len(candidates), n_cells, replace=False)
            self.centers = candidates[idx]
        else:
            self.centers = candidates[:n_cells]
            self.n_cells = len(self.centers)

    def _build_voronoi(self):
        """Precompute Voronoi cell labels and sarcomere texture.

        Computed at world resolution (512×512) and upscaled to internal
        resolution for speed. Cell boundaries remain sharp enough for
        40x rendering. Sarcomere striations computed per cell orientation.
        """
        s = self.internal_scale
        h_w, w_w = self.height, self.width  # world resolution
        h_i, w_i = self._ih, self._iw      # internal resolution

        # Compute label map at world resolution (fast)
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

        # Upscale to internal resolution using nearest-neighbor
        if s > 1:
            self._cell_labels = cv2.resize(
                labels_world.astype(np.float32), (w_i, h_i),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        else:
            self._cell_labels = labels_world

        # Precompute boundary mask (pixels where neighbors differ)
        # Use 4-connectivity for thin boundaries (intercalated discs)
        shifted_r = np.roll(self._cell_labels, 1, axis=1)
        shifted_d = np.roll(self._cell_labels, 1, axis=0)
        self._boundary_mask = (
            (self._cell_labels != shifted_r) | (self._cell_labels != shifted_d)
        )

        # Phase contrast halo: gradient-fading bright fringe flanking
        # the dark intercalated disc lines — where OPL changes abruptly.
        # Use distance transform for a smooth falloff from boundary.
        boundary_u8 = self._boundary_mask.astype(np.uint8)
        dist_from_boundary = cv2.distanceTransform(
            1 - boundary_u8, cv2.DIST_L2, 3,
        )
        # Halo fades over ~3 world px (12 internal px)
        halo_width = 3.0 * s
        halo_falloff = np.clip(1.0 - dist_from_boundary / halo_width, 0, 1)
        halo_falloff[self._boundary_mask] = 0  # boundary itself is dark, not halo
        self._halo_gradient = (halo_falloff ** 1.2 * 55.0).astype(np.float32)

        # Precompute nucleus mask (lighter oval per cell, visible at 20x+)
        # CM nuclei: ~10 µm diameter, slightly offset from cell centroid
        nuc_mask = np.zeros((h_i, w_i), dtype=np.float32)
        nuc_radius_world = 4.0  # world px ≈ 4 µm radius
        for ci in range(self.n_cells):
            cx_i = int(self.centers[ci, 0] * s)
            cy_i = int(self.centers[ci, 1] * s)
            r = int((nuc_radius_world + self.rng.normal(0, 0.5)) * s)
            r = max(2 * s, r)
            cv2.circle(nuc_mask, (cx_i, cy_i), r, 1.0, -1)
        self._nuc_mask = nuc_mask

        # Precompute sarcomere striation texture (visible at 40x)
        # Per-cell myofibril orientation (random, 0..π)
        self._cell_orientation = self.rng.uniform(0, np.pi, self.n_cells)
        orient_map = self._cell_orientation[self._cell_labels]
        yy, xx = np.mgrid[:h_i, :w_i]
        # Sarcomere period ≈ 2 µm ≈ 2.5 world px ≈ 10 internal px
        # At 40x (no downscale): ~51 striations across FOV
        # At 10x (4x downscale): period < 3 px → averaged out by INTER_AREA
        period = 2.5 * s
        phase = xx * np.cos(orient_map) + yy * np.sin(orient_map)
        self._striation_tex = (
            np.sin(2.0 * np.pi * phase / period) * 7.0
        ).astype(np.float32)
        # Mask out boundary pixels (no striations at cell edges)
        self._striation_tex[self._boundary_mask] = 0

    # ── Backwards-compatible properties ──

    @property
    def theta(self):
        """Backwards compat: cell phases derived from calcium intensity."""
        return self._cell_intensities * 2 * np.pi

    @theta.setter
    def theta(self, val):
        """Accept theta writes for backward compat (resets automaton)."""
        self._ca_state[:] = 0
        self._gcamp[:] = 0.0
        self._pacemaker_accum = 0
        self._ectopic_accum = round(self._ectopic_period_steps * 0.3)
        self._update_intensities()

    @property
    def omega(self):
        """Backwards compat: frequency array (same as pacemaker freq)."""
        freqs = np.full(self.n_cells, 2 * np.pi * self.get_pacemaker_freq())
        freqs[self._is_arrhythmic] = 2 * np.pi * self.get_ectopic_freq()
        return freqs

    @omega.setter
    def omega(self, val):
        """Accept omega writes — maps to pacemaker period."""
        mean_omega = np.mean(val)
        if mean_omega > 0:
            self._pacemaker_period = 2 * np.pi / mean_omega

    @property
    def _base_omega(self):
        """Backwards compat."""
        return self.omega

    @_base_omega.setter
    def _base_omega(self, val):
        pass

    @property
    def _base_coupling(self):
        """Backwards compat."""
        return self.coupling_strength

    @_base_coupling.setter
    def _base_coupling(self, val):
        self.coupling_strength = val

    # ── Cellular automaton wave dynamics ──

    def _evolve(self, n_steps):
        """Advance excitable-wave automaton for n_steps.

        3-state model per grid point:
          0         = resting (can be triggered by excited neighbor)
          1.._APD   = excited (emits GCaMP, triggers resting neighbors)
          _APD+1.._APD+_RRP = refractory (unresponsive)

        Wave propagation: resting cells become excited if any
        nearest-neighbor is in early excitation (state 1..2).
        """
        state = self._ca_state
        gcamp = self._gcamp
        APD, RRP = self._APD, self._RRP
        total_cycle = APD + RRP
        gc_decay = self._gcamp_decay
        gc_alpha = self._alpha_gcamp

        # Prepare SLM mask (resize if needed)
        stim = None
        if self._stim_mask is not None and np.any(self._stim_mask):
            stim = self._stim_mask
            N = self.SIM_SIZE
            if stim.shape != (N, N):
                stim = cv2.resize(
                    stim.astype(np.uint8), (N, N),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

        for _ in range(n_steps):
            # Advance all active cells
            active = state > 0
            state[active] += 1

            # Cells that complete refractory → resting
            state[state > total_cycle] = 0

            # Trigger: resting cells excited by neighbors in early excitation
            # 8-connectivity for isotropic (circular) wavefront propagation
            resting = state == 0
            early_ex = (state >= 1) & (state <= 2)
            # Cardinal neighbors
            trigger = (
                np.roll(early_ex, 1, 0) | np.roll(early_ex, -1, 0)
                | np.roll(early_ex, 1, 1) | np.roll(early_ex, -1, 1)
            )
            # Diagonal neighbors
            trigger |= (
                np.roll(np.roll(early_ex, 1, 0), 1, 1)
                | np.roll(np.roll(early_ex, 1, 0), -1, 1)
                | np.roll(np.roll(early_ex, -1, 0), 1, 1)
                | np.roll(np.roll(early_ex, -1, 0), -1, 1)
            )
            state[resting & trigger] = 1

            # Pacemaker firing
            self._pacemaker_accum += 1
            if self._pacemaker_accum >= self._pacemaker_period_steps:
                self._pacemaker_accum = 0
                resting_pm = (state[self._pm_mask] == 0)
                if resting_pm.sum() > 0.5 * self._pm_mask.sum():
                    state[self._pm_mask & (state == 0)] = 1

            # Ectopic focus firing
            if self._ectopic_active:
                self._ectopic_accum += 1
                if self._ectopic_accum >= self._ectopic_period_steps:
                    self._ectopic_accum = 0
                    resting_ec = (state[self._ec_mask] == 0)
                    if resting_ec.sum() > 0.5 * self._ec_mask.sum():
                        state[self._ec_mask & (state == 0)] = 1

            # SLM stimulation: trigger resting cells under illumination
            if stim is not None:
                state[stim & (state == 0)] = 1

            # GCaMP fluorescence
            excited = (state >= 1) & (state <= APD)
            gcamp[excited] += gc_alpha
            gcamp *= gc_decay
            np.clip(gcamp, 0, 1.5, out=gcamp)

        self._ca_state = state
        self._gcamp = gcamp
        self._time += n_steps
        self._update_intensities()

    def _update_intensities(self):
        """Sample GCaMP field at each cell's position."""
        gc_vals = self._gcamp[self._cell_pde_y, self._cell_pde_x]
        self._cell_intensities = (
            np.clip(gc_vals, 0, 1) * self._dye_loading * self._drug_amplitude_mult
        )

    def _get_stimulated_cells(self) -> np.ndarray:
        """Check which cells are under SLM illumination."""
        stimulated = np.zeros(self.n_cells, dtype=bool)
        if self._stim_mask is None:
            return stimulated
        for i in range(self.n_cells):
            gx = self._cell_pde_x[i]
            gy = self._cell_pde_y[i]
            if self._stim_mask[gy, gx]:
                stimulated[i] = True
        return stimulated

    def _accumulate_z_drift(self, dt):
        """Accumulate Z-drift (thermal/mechanical drift during timelapse)."""
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

    # ── Temperature response ──

    def _get_temperature(self) -> float:
        """Read current temperature (°C) from the controller device."""
        if "Temperature" not in self.state_devices:
            return 37.0  # default body temperature for mammalian cells
        return float(self.state_devices["Temperature"].get("label", "37"))

    def _temp_rate_factor(self) -> float:
        """Beating-rate multiplier. Cardiomyocytes optimal at 37°C.

        Q10 ≈ 2.5 for cardiac conduction. Hypothermia slows beating;
        hyperthermia speeds it up. Below 20°C cells are essentially
        quiescent; above 42°C heat damage begins.
        """
        temp = self._get_temperature()
        if temp < 15:
            return 0.05  # near-quiescent
        factor = 2.5 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.1, 1.0 - (temp - 42) * 0.3)
        return factor

    def step(self, dt: float = 1.0):
        """Advance simulation, scaling PDE steps proportional to dt."""
        temp_factor = self._temp_rate_factor()
        n_steps = max(1, round(dt * self.steps_per_snap * temp_factor))
        self._accumulate_z_drift(dt)
        self._update_drug_effect()
        self._evolve(n_steps)

    def step_autonomous(self, dt: float = 1.0):
        """Advance PDE WITH continuous SLM optogenetic pacing.

        Used by RealtimeEngine for background dynamics. SLM stimulation
        triggers resting cells under illumination — this is continuous
        optogenetic pacing and must be applied during background stepping.
        Drug effects and pacemakers also continue firing.
        """
        temp_factor = self._temp_rate_factor()
        n_steps = max(1, round(dt * self.steps_per_snap * temp_factor))
        self._accumulate_z_drift(dt)
        self._update_drug_effect()
        self._evolve(n_steps)

    # ── Rendering ──

    def _render_bf_full(self) -> np.ndarray:
        """Brightfield: dense confluent tissue with phase contrast rendering.

        Phase contrast cardiomyocyte monolayer:
          - Cell bodies are darker than background (higher refractive index)
          - Intercalated discs (cell-cell junctions) are prominent dark lines
          - Phase halos flank boundaries (bright ring from OPL gradient)
          - Sarcomere striations visible at 40x (~2 µm period)
          - Nuclei lighter (less dense, shade-off)
          - Contraction: cells darken + boundary gaps widen during systole
        """
        h, w = self._ih, self._iw

        # Base: cell bodies with contraction-dependent darkening
        # Resting ~120 gray, contracting ~95 gray (darker in phase contrast)
        cell_vals = 120.0 - 25.0 * self._cell_intensities
        body = cell_vals[self._cell_labels].astype(np.float32)

        # Nuclei: lighter ovals (nuclear shade-off, less dense cytoplasm)
        body += self._nuc_mask * 14.0

        # Sarcomere striations (visible at 40x, averaged out at 10x)
        body += self._striation_tex

        # Phase contrast halo: gradient bright fringe flanking cell boundaries
        # Brightest near the boundary edge, fading toward cell interior
        body += self._halo_gradient

        # Intercalated discs: dark lines at cell-cell junctions
        # These are the most prominent feature in CM monolayer BF images
        body[self._boundary_mask] = 82.0

        # Contraction-dependent halo brightening: contracting cells
        # cause more phase shift → brighter halo
        px_calcium = self._cell_intensities[self._cell_labels]
        body += self._halo_gradient * px_calcium * 0.3

        # Subcellular texture: faint granular noise (mitochondria, organelles)
        if not hasattr(self, '_bf_texture'):
            tex_rng = np.random.default_rng(self._seed + 555)
            self._bf_texture = tex_rng.normal(0, 2.5, (h, w)).astype(np.float32)
        body += self._bf_texture

        body = np.clip(body, 0, 255).astype(np.uint8)
        img = np.stack([body, body, body], axis=2)
        return img

    def _render_nuc_full(self) -> np.ndarray:
        """GCaMP channel: calcium wave as smooth spatial gradient.

        Upscales the GCaMP PDE field directly to internal resolution for
        smooth wavefront rendering. Per-cell dye loading heterogeneity
        modulates brightness. Cell boundaries dimmed slightly.
        """
        h, w = self._ih, self._iw

        # Upscale GCaMP field (128×128) to internal resolution
        gcamp_field = cv2.resize(
            self._gcamp.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        np.clip(gcamp_field, 0, 1.0, out=gcamp_field)

        # Per-pixel dye loading from cell labels
        dye = (self._dye_loading * self._drug_amplitude_mult)[self._cell_labels]

        # Map to pixel intensity: baseline + calcium * dye * scale
        base = 15.0
        img_f = base + gcamp_field * dye * 210.0
        np.clip(img_f, 0, 255, out=img_f)

        # Subtle boundary dimming (cell-cell gaps attenuate signal)
        img_f[self._boundary_mask] -= 6.0
        np.clip(img_f, 0, 255, out=img_f)

        gcamp_img = img_f.astype(np.uint8)
        img = np.stack([gcamp_img, gcamp_img, gcamp_img], axis=2)
        return img

    def _render_mem_full(self) -> np.ndarray:
        """Membrane channel: cell boundaries (N-cadherin / connexin-43)."""
        h, w = self._ih, self._iw

        # Faint cytoplasm + bright boundaries
        membrane = np.full((h, w), 12, dtype=np.uint8)
        membrane[self._boundary_mask] = 160

        img = np.stack([membrane, membrane, membrane], axis=2)
        return img

    # ── snap_frame ──

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()

        # Handle SLM mask — map to PDE grid
        if mask is not None and np.any(mask):
            pde_mask = cv2.resize(
                mask.astype(np.uint8), (self.SIM_SIZE, self.SIM_SIZE),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            self._stim_mask = pde_mask
        else:
            self._stim_mask = None

        self._auto_step_tick()
        self._snap_count += 1

        # Render
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
        viewport = self._apply_pipeline(viewport, exposure)
        viewport = self._apply_exposure(viewport, exposure, intensity)

        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def reset(self, seed: int = None):
        super().reset(seed)
        self._ca_state[:] = 0
        self._gcamp[:] = 0.0
        self._pacemaker_accum = 0
        self._ectopic_accum = round(self._ectopic_period_steps * 0.3)
        self._snap_count = 0
        self._time = 0.0
        self._stim_mask = None
        self._drug_active = False
        self._drug_effect = 0.0
        self._drug_amplitude_mult = 1.0
        self._pacemaker_period_steps = self._base_pm_period_steps
        self._ectopic_period_steps = self._base_ec_period_steps
        self._evolve(2000)  # re-establish wave pattern

    # ── Drug response ──

    def apply_drug(self, drug_name: str, onset_rate: float = 0.05):
        """Apply a cardiac drug with gradual onset.

        Drugs modify pacemaker frequency and/or conduction velocity:
          - isoproterenol: +50% freq (beta agonist, positive chronotropy)
          - verapamil: -30% freq, -40% conduction (Ca channel blocker)
          - lidocaine: -60% conduction (Na channel blocker)
          - caffeine: +30% freq, -15% conduction (mild stimulant)
        """
        if drug_name not in self._drug_profiles:
            raise ValueError(f"Unknown drug: {drug_name}. "
                             f"Available: {list(self._drug_profiles.keys())}")
        self._drug_active = True
        self._drug_name = drug_name
        self._drug_effect = 0.0
        self._drug_onset_rate = onset_rate
        self._drug_washing_out = False

    def remove_drug(self, washout_rate: float = 0.03):
        """Remove drug — effect washes out gradually."""
        if not self._drug_active:
            return
        self._drug_washing_out = True
        self._drug_washout_rate = washout_rate

    def _update_drug_effect(self):
        """Update drug effect level and apply to automaton parameters."""
        if not self._drug_active:
            return

        if self._drug_washing_out:
            self._drug_effect = max(0.0, self._drug_effect - self._drug_washout_rate)
            if self._drug_effect <= 0.001:
                self._drug_active = False
                self._drug_name = None
                self._drug_effect = 0.0
                self._drug_washing_out = False
                self.Du = self._base_Du
                self._pacemaker_period = self._base_pacemaker_period
                self._ectopic_period = self._base_ectopic_period
                self._pacemaker_period_steps = self._base_pm_period_steps
                self._ectopic_period_steps = self._base_ec_period_steps
                self._drug_amplitude_mult = 1.0
                return
        else:
            self._drug_effect = min(1.0, self._drug_effect + self._drug_onset_rate)

        profile = self._drug_profiles[self._drug_name]
        e = self._drug_effect

        # Interpolate pacemaker frequency
        freq_mult = 1.0 + e * (profile["freq_mult"] - 1.0)
        self._pacemaker_period = self._base_pacemaker_period / max(0.1, freq_mult)
        self._ectopic_period = self._base_ectopic_period / max(0.1, freq_mult)

        # Update automaton period steps (what _evolve actually uses)
        self._pacemaker_period_steps = max(
            self._APD + self._RRP + 1,
            round(self._base_pm_period_steps / max(0.1, freq_mult)),
        )
        self._ectopic_period_steps = max(
            self._APD + self._RRP + 1,
            round(self._base_ec_period_steps / max(0.1, freq_mult)),
        )

        # Interpolate conduction velocity (nominal — automaton uses
        # nearest-neighbor propagation so Du only affects GT reporting)
        Du_mult = 1.0 + e * (profile["Du_mult"] - 1.0)
        self.Du = self._base_Du * Du_mult

        # Amplitude multiplier for rendering
        self._drug_amplitude_mult = 1.0 + e * (profile["amplitude_mult"] - 1.0)

    # ── Ground truth / analysis ──

    def get_mean_calcium(self, group: str = "all") -> float:
        """Mean calcium level for a group of cells."""
        if group == "normal":
            idx = ~self._is_arrhythmic
        elif group == "arrhythmic":
            idx = self._is_arrhythmic
        else:
            idx = np.ones(self.n_cells, dtype=bool)
        return round(float(self._cell_intensities[idx].mean()), 4)

    def get_active_fraction(self, threshold: float = 0.3) -> float:
        """Fraction of cells currently above calcium threshold (systole)."""
        return round(float((self._cell_intensities > threshold).mean()), 4)

    def get_pacemaker_freq(self) -> float:
        """Current pacemaker firing frequency (Hz)."""
        return round(1.0 / max(0.01, self._pacemaker_period), 3)

    def get_ectopic_freq(self) -> float:
        """Current ectopic focus frequency (Hz)."""
        return round(1.0 / max(0.01, self._ectopic_period), 3)

    def get_order_parameter(self) -> float:
        """Spatial synchrony measure: fraction of tissue in the same state.

        Analogous to the Kuramoto order parameter. Computed from the
        calcium field: 1.0 = all cells in same phase, 0.0 = desynchronised.
        """
        # Use complex-valued order parameter on calcium intensities
        # Map intensity [0,1] → phase [0, 2π]
        phases = self._cell_intensities * 2 * np.pi
        z = np.exp(1j * phases)
        return round(float(abs(z.mean())), 4)

    def get_group_order(self, group: str = "all") -> float:
        """Order parameter for a subset of cells."""
        if group == "normal":
            idx = ~self._is_arrhythmic
        elif group == "arrhythmic":
            idx = self._is_arrhythmic
        else:
            idx = np.ones(self.n_cells, dtype=bool)
        phases = self._cell_intensities[idx] * 2 * np.pi
        z = np.exp(1j * phases)
        return round(float(abs(z.mean())), 4)

    def get_mean_frequency(self, group: str = "all") -> float:
        """Pacemaker frequency for a group (Hz)."""
        if group == "arrhythmic":
            return self.get_ectopic_freq()
        return self.get_pacemaker_freq()

    def get_ground_truth(self) -> dict:
        gt = {
            "n_cells": self.n_cells,
            "order_parameter": self.get_order_parameter(),
            "normal_R": self.get_group_order("normal"),
            "arrhythmic_R": self.get_group_order("arrhythmic"),
            "normal_freq": self.get_pacemaker_freq(),
            "arrhythmic_freq": self.get_ectopic_freq(),
            "pacemaker_freq": self.get_pacemaker_freq(),
            "ectopic_freq": self.get_ectopic_freq(),
            "ectopic_active": self._ectopic_active,
            "n_arrhythmic": int(self._is_arrhythmic.sum()),
            "mean_calcium": self.get_mean_calcium(),
            "active_fraction": self.get_active_fraction(),
            "pacemaker_position_pde": self._pacemaker_main,
            "ectopic_position_pde": self._ectopic_center,
            "time": round(self._time, 2),
            "conduction_Du": round(self.Du, 2),
        }
        if self._drug_active:
            gt["drug"] = {
                "name": self._drug_name,
                "effect_level": round(self._drug_effect, 3),
                "washing_out": self._drug_washing_out,
                "effective_pacemaker_freq": self.get_pacemaker_freq(),
                "effective_Du": round(self.Du, 2),
            }
        return gt
