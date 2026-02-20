"""
YeastSim — Budding yeast (S. cerevisiae) simulation.

Simulates round yeast cells (~5µm diameter) with:
  - Budding division: bud emerges, grows, pinches off
  - Bud scars: visible rings on mother cell surface (replicative age)
  - Phase contrast and fluorescence rendering
  - Population growth dynamics

Usage via SimulationBridge:
    sim = YeastSim(world_size=512, n_cells=40, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.optical_pipeline import OpticalPipeline


class YeastSim:
    """Budding yeast simulation.

    Channels:
      - mode 0: Phase contrast (bright halo, dark interior)
      - mode 1: Calcofluor White (bud scars — bright rings on cell surface)
      - mode 2: GFP reporter (cytoplasmic fluorescence)
    """

    # Cell cycle phases
    PHASE_G1 = 0      # No bud, growing
    PHASE_S_BUD = 1   # Bud emerging (small)
    PHASE_G2_BUD = 2  # Bud growing (medium)
    PHASE_M_BUD = 3   # Bud mature (nearly full-size), about to divide

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_cells: int = 200,
        seed: int = 42,
        fixed_dt: float = 1.0,
        internal_scale: int = 4,
        division_time: float = 15.0,
    ):
        self.width = world_size
        self.height = world_size
        self.internal_scale = internal_scale
        self._iw = world_size * internal_scale
        self._ih = world_size * internal_scale
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

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
        self._mode_map = {
            ("SCFP2(434/474)", "UV"): 1,
            ("TagGFP2(483/506)", "GREEN"): 2,
        }
        self._snap_count = 0

        self.rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 7777)
        self.fixed_dt = fixed_dt
        self._time = 0.0

        # Auto-step and dynamics
        self.auto_step = False
        self.snaps_per_step = 2

        # Z-drift (thermal/mechanical drift during timelapse)
        self.z_drift_rate = 0.0   # µm/s (positive = tissue drifts up)
        self.z_drift_noise = 0.0  # σ of z-jitter (µm·s⁻½, Brownian)

        # Population parameters
        self.max_cells = 800
        self.division_time = division_time  # mean time steps between divisions
        self.gfp_expression = True  # all cells express GFP

        # Optical pipeline
        self._pipeline = OpticalPipeline()
        self._pipeline.noise = {"photon_scale": 200, "read_noise": 3.0}
        self._pipeline.vignette = 0.08

        # ── Cell state arrays ──
        margin = 30
        self.n_cells = n_cells
        self.x = self.rng.uniform(margin, world_size - margin, n_cells).astype(np.float64)
        self.y = self.rng.uniform(margin, world_size - margin, n_cells).astype(np.float64)
        self.radius = self.rng.normal(2.5, 0.3, n_cells).clip(2.0, 3.5)  # µm
        self.alive = np.ones(n_cells, dtype=bool)

        # Cell cycle state
        self.phase = np.full(n_cells, self.PHASE_G1, dtype=int)
        self.phase_timer = self.rng.uniform(0, self.division_time, n_cells)
        # Randomize some cells to be budding at start
        for i in range(n_cells):
            if self.rng.random() < 0.35:
                self.phase[i] = self.rng.integers(1, 4)
                self.phase_timer[i] = self.rng.uniform(0, 5)

        # Bud properties (angle, relative size)
        self.bud_angle = self.rng.uniform(0, 2 * np.pi, n_cells)
        self.bud_size = np.zeros(n_cells, dtype=np.float64)  # 0-1 relative to mother
        self._update_bud_sizes()

        # Bud scars (replicative age) — list of angles per cell
        self._bud_scars = [[] for _ in range(n_cells)]
        # Give some cells existing bud scars
        for i in range(n_cells):
            n_scars = self.rng.integers(0, 5)
            for _ in range(n_scars):
                self._bud_scars[i].append(float(self.rng.uniform(0, 2 * np.pi)))

        # GFP intensity per cell (variable expression)
        self.gfp_intensity = self.rng.uniform(120, 220, n_cells).astype(np.float64)

        # ── Per-cell morphology: ellipticity + vacuoles ──
        # Slight ellipticity (axis ratio 1.0-1.2) and random orientation
        self._aspect_ratio = self.rng.uniform(1.0, 1.2, n_cells)
        self._orientation = self.rng.uniform(0, np.pi, n_cells)  # radians

        # Vacuoles: 0-3 per cell; older cells (more bud scars) → larger/more
        self._vacuoles = []  # list of list of (rel_x, rel_y, rel_radius)
        for i in range(n_cells):
            self._vacuoles.append(self._generate_vacuoles(i))

        # Lipid droplets: 0-5 per cell, small bright puncta
        self._lipid_droplets = []  # list of list of (rel_x, rel_y)
        for i in range(n_cells):
            self._lipid_droplets.append(self._generate_lipid_droplets())

        # Phase contrast density: per-cell metabolic state
        # Actively growing cells are "phase-dark" (denser cytoplasm, higher RI)
        # Quiescent/newly divided daughter cells are "phase-light" (less dense)
        # 0.0 = phase-light (lighter interior), 1.0 = phase-dark (darker)
        self._phase_contrast = self.rng.uniform(0.5, 1.0, n_cells).astype(np.float64)
        # G1 cells (no bud) are more likely phase-light
        for i in range(n_cells):
            if self.phase[i] == self.PHASE_G1:
                self._phase_contrast[i] *= self.rng.uniform(0.6, 1.0)

        # ── Drug response state ──
        self._drug_active = False
        self._drug_arrest_phase = None  # which phase to arrest at
        self._drug_name = None
        self._arrest_time = np.full(n_cells, 0.0)  # time spent arrested
        self._drug_death_threshold = 0.0  # 0 = no death from arrest
        self._dead_cells = np.zeros(n_cells, dtype=bool)

        # Pre-render
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None
        self._dirty = True
        self._render_full()

    def _generate_vacuoles(self, cell_idx: int) -> list:
        """Generate vacuole positions for a cell.

        Returns list of (rel_x, rel_y, rel_radius) in units of cell radius.
        Older cells (more bud scars) tend to have larger/more vacuoles.
        Very old cells (5+ scars) often have a single dominant vacuole
        filling much of the cell interior.
        """
        n_scars = len(self._bud_scars[cell_idx]) if cell_idx < len(self._bud_scars) else 0
        # Old cells (5+ scars): single dominant vacuole ~55-65% of cell radius
        if n_scars >= 5 and self.rng.random() < 0.6:
            size = self.rng.uniform(0.50, 0.62)
            dist = self.rng.uniform(0.0, 0.15)
            angle = self.rng.uniform(0, 2 * np.pi)
            return [(dist * np.cos(angle), dist * np.sin(angle), size)]
        # 0-3 vacuoles, more likely with age
        n_vac = min(3, self.rng.poisson(0.6 + 0.3 * min(n_scars, 5)))
        vacs = []
        for _ in range(n_vac):
            # Position: random within cell, biased toward center
            angle = self.rng.uniform(0, 2 * np.pi)
            dist = self.rng.uniform(0.05, 0.5)  # relative to cell radius
            # Size: 25-47% of cell radius, larger in older cells
            base_size = self.rng.uniform(0.25, 0.47)
            size = base_size + 0.02 * min(n_scars, 5)
            vacs.append((dist * np.cos(angle), dist * np.sin(angle), size))
        return vacs

    def _generate_lipid_droplets(self) -> list:
        """Generate lipid droplet positions for a cell.

        Returns list of (rel_x, rel_y) in units of cell radius.
        Small, bright refractile puncta scattered in cytoplasm.
        """
        n_drops = self.rng.poisson(3)
        drops = []
        for _ in range(n_drops):
            angle = self.rng.uniform(0, 2 * np.pi)
            dist = self.rng.uniform(0.15, 0.65)
            drops.append((dist * np.cos(angle), dist * np.sin(angle)))
        return drops

    def _update_bud_sizes(self):
        """Set bud sizes with smooth continuous growth within each phase.

        Bud grows from 0.15 (S emergence) to 0.75 (M completion),
        smoothly interpolated by phase_timer progress within each phase.
        """
        for i in range(len(self.phase)):
            if not self.alive[i]:
                self.bud_size[i] = 0
            elif self.phase[i] == self.PHASE_G1:
                self.bud_size[i] = 0
            elif self.phase[i] == self.PHASE_S_BUD:
                # S phase: 0.15 → 0.35 (timer counts down from div_time*0.2)
                phase_dur = self.division_time * 0.2
                progress = np.clip(1.0 - self.phase_timer[i] / phase_dur, 0, 1)
                self.bud_size[i] = 0.15 + 0.20 * progress
            elif self.phase[i] == self.PHASE_G2_BUD:
                # G2 phase: 0.35 → 0.60
                phase_dur = self.division_time * 0.2
                progress = np.clip(1.0 - self.phase_timer[i] / phase_dur, 0, 1)
                self.bud_size[i] = 0.35 + 0.25 * progress
            elif self.phase[i] == self.PHASE_M_BUD:
                # M phase: 0.60 → 0.75
                phase_dur = self.division_time * 0.1
                progress = np.clip(1.0 - self.phase_timer[i] / phase_dur, 0, 1)
                self.bud_size[i] = 0.60 + 0.15 * progress

    # ── Drug response API ──

    def apply_drug(self, drug_name: str = "nocodazole",
                   arrest_phase: int = None,
                   death_threshold: float = 0.0):
        """Apply a drug that arrests cells at a specific phase.

        Parameters
        ----------
        drug_name : str
            Drug name: 'nocodazole' (M arrest), 'hydroxyurea' (S arrest),
            'alpha_factor' (G1 arrest), or custom name with arrest_phase.
        arrest_phase : int, optional
            Phase to arrest at (overrides drug_name default).
        death_threshold : float
            Time steps arrested before cell dies (0 = no death).
        """
        self._drug_active = True
        self._drug_name = drug_name

        # Default arrest phases per drug
        phase_map = {
            "nocodazole": self.PHASE_M_BUD,    # spindle poison → M arrest
            "hydroxyurea": self.PHASE_S_BUD,    # DNA synth inhibitor → S arrest
            "alpha_factor": self.PHASE_G1,      # mating pheromone → G1 arrest
        }
        if arrest_phase is not None:
            self._drug_arrest_phase = arrest_phase
        else:
            self._drug_arrest_phase = phase_map.get(drug_name, self.PHASE_M_BUD)

        self._drug_death_threshold = death_threshold
        # Reset arrest timers
        self._arrest_time[:self.n_cells] = 0.0

    def remove_drug(self):
        """Remove drug — cells resume cycling from current phase."""
        # Reset phase timers for arrested cells so they can advance
        if self._drug_arrest_phase is not None:
            for i in range(self.n_cells):
                if (self.alive[i] and not self._dead_cells[i]
                        and self.phase[i] == self._drug_arrest_phase
                        and self._arrest_time[i] > 0):
                    # Give cells a short remaining time in current phase
                    remaining = self.division_time * 0.05
                    self.phase_timer[i] = remaining + \
                        self.rng.normal(0, remaining * 0.3)

        self._drug_active = False
        self._drug_name = None
        self._drug_arrest_phase = None
        self._arrest_time[:self.n_cells] = 0.0

    # ── Temperature response ──

    def _get_temperature(self) -> float:
        """Read current temperature (°C) from the controller device."""
        if "Temperature" not in self.state_devices:
            return 20.0
        return float(self.state_devices["Temperature"].get("label", "20"))

    def _temp_growth_factor(self) -> float:
        """Cell-cycle rate multiplier. S. cerevisiae optimal at ~30°C.

        Q10 ≈ 2.0 below optimum. Above 33°C heat stress reduces growth;
        at 37°C ~60% of peak. Above 40°C lethal.
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.0  # cold arrest
        if temp <= 30:
            return 2.0 ** ((temp - 30) / 10.0)
        # Above 30°C: heat stress — decline toward zero
        if temp <= 37:
            return 1.0 - 0.06 * (temp - 30)  # 30°C=1.0, 37°C≈0.58
        return max(0.05, 0.58 - 0.15 * (temp - 37))  # 42°C≈~0

    def step(self, dt: float = 1.0):
        """Advance the simulation by one time step."""
        if self.fixed_dt > 0:
            dt = self.fixed_dt

        self._time += dt
        growth_factor = self._temp_growth_factor()

        for i in range(self.n_cells):
            if not self.alive[i]:
                continue

            # Check drug-induced death
            if self._dead_cells[i]:
                continue

            # Drug arrest: if cell is at arrest phase, don't advance
            if (self._drug_active and
                    self.phase[i] == self._drug_arrest_phase):
                self._arrest_time[i] += dt
                # Drug-induced death after prolonged arrest
                if (self._drug_death_threshold > 0 and
                        self._arrest_time[i] >= self._drug_death_threshold):
                    self._dead_cells[i] = True
                    self.gfp_intensity[i] *= 0.15  # lose GFP expression
                continue

            self.phase_timer[i] -= dt * growth_factor

            if self.phase_timer[i] <= 0:
                # Advance phase
                next_phase = self.phase[i] + 1

                # Check if drug blocks the NEXT phase
                if (self._drug_active and
                        next_phase == self._drug_arrest_phase and
                        self.phase[i] != self._drug_arrest_phase):
                    # Transition into arrest phase then stop
                    self.phase[i] = next_phase
                    if next_phase == self.PHASE_S_BUD:
                        self.bud_angle[i] = self.rng.uniform(0, 2 * np.pi)
                    self.phase_timer[i] = 999  # won't expire during arrest
                    continue

                if self.phase[i] == self.PHASE_G1:
                    # Start budding
                    self.phase[i] = self.PHASE_S_BUD
                    self.bud_angle[i] = self.rng.uniform(0, 2 * np.pi)
                    self.phase_timer[i] = self.division_time * 0.2

                elif self.phase[i] == self.PHASE_S_BUD:
                    self.phase[i] = self.PHASE_G2_BUD
                    self.phase_timer[i] = self.division_time * 0.2

                elif self.phase[i] == self.PHASE_G2_BUD:
                    self.phase[i] = self.PHASE_M_BUD
                    self.phase_timer[i] = self.division_time * 0.1

                elif self.phase[i] == self.PHASE_M_BUD:
                    # Division! Create daughter cell
                    self._divide(i)
                    # Mother returns to G1
                    self.phase[i] = self.PHASE_G1
                    self.phase_timer[i] = self.division_time * 0.5 + \
                        self.rng.normal(0, self.division_time * 0.1)
                    # Record bud scar on mother
                    self._bud_scars[i].append(float(self.bud_angle[i]))

        self._update_bud_sizes()

        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        self._dirty = True  # lazy re-render on next snap_frame()

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (no SLM effects)."""
        self.step(dt)

    def _divide(self, mother_idx: int):
        """Create a daughter cell from a budding mother."""
        if self.n_cells >= self.max_cells:
            return

        # Daughter position: at bud location
        angle = self.bud_angle[mother_idx]
        r = self.radius[mother_idx]
        daughter_x = self.x[mother_idx] + np.cos(angle) * r * 1.8
        daughter_y = self.y[mother_idx] + np.sin(angle) * r * 1.8

        # Clamp to world
        daughter_x = np.clip(daughter_x, 5, self.width - 5)
        daughter_y = np.clip(daughter_y, 5, self.height - 5)

        # Extend arrays if needed
        if self.n_cells >= len(self.x):
            self._grow_arrays()

        idx = self.n_cells
        self.x[idx] = daughter_x
        self.y[idx] = daughter_y
        self.radius[idx] = self.rng.normal(2.5, 0.3)
        self.alive[idx] = True
        self.phase[idx] = self.PHASE_G1
        self.phase_timer[idx] = self.division_time * 0.5 + \
            self.rng.normal(0, self.division_time * 0.1)
        self.bud_angle[idx] = self.rng.uniform(0, 2 * np.pi)
        self.bud_size[idx] = 0
        self.gfp_intensity[idx] = self.rng.uniform(120, 220)
        self._bud_scars.append([])  # fresh daughter — no scars
        # Morphology for new cell
        if idx >= len(self._aspect_ratio):
            self._aspect_ratio = np.concatenate([
                self._aspect_ratio,
                np.full(len(self.x) - len(self._aspect_ratio), 1.1)])
            self._orientation = np.concatenate([
                self._orientation,
                np.zeros(len(self.x) - len(self._orientation))])
        self._aspect_ratio[idx] = self.rng.uniform(1.0, 1.2)
        self._orientation[idx] = self.rng.uniform(0, np.pi)
        self._vacuoles.append(self._generate_vacuoles(idx))
        self._lipid_droplets.append(self._generate_lipid_droplets())
        # Phase contrast: daughters start phase-light (less dense)
        if idx >= len(self._phase_contrast):
            self._phase_contrast = np.concatenate([
                self._phase_contrast,
                np.full(len(self.x) - len(self._phase_contrast), 0.7)])
        self._phase_contrast[idx] = self.rng.uniform(0.3, 0.6)
        # Drug state for new cell
        if idx >= len(self._arrest_time):
            self._arrest_time = np.concatenate([
                self._arrest_time, np.zeros(len(self.x) - len(self._arrest_time))])
            self._dead_cells = np.concatenate([
                self._dead_cells, np.zeros(len(self.x) - len(self._dead_cells), dtype=bool)])
        self._arrest_time[idx] = 0.0
        self._dead_cells[idx] = False
        self.n_cells += 1

    def _grow_arrays(self):
        """Double array capacity."""
        n = len(self.x)

        self.x = np.concatenate([self.x, np.zeros(n)])
        self.y = np.concatenate([self.y, np.zeros(n)])
        self.radius = np.concatenate([self.radius, np.full(n, 2.5)])
        self.alive = np.concatenate([self.alive, np.zeros(n, dtype=bool)])
        self.phase = np.concatenate([self.phase, np.zeros(n, dtype=int)])
        self.phase_timer = np.concatenate([self.phase_timer, np.zeros(n)])
        self.bud_angle = np.concatenate([self.bud_angle, np.zeros(n)])
        self.bud_size = np.concatenate([self.bud_size, np.zeros(n)])
        self.gfp_intensity = np.concatenate([self.gfp_intensity, np.full(n, 160.0)])
        self._phase_contrast = np.concatenate([self._phase_contrast, np.full(n, 0.7)])
        self._arrest_time = np.concatenate([self._arrest_time, np.zeros(n)])
        self._dead_cells = np.concatenate([self._dead_cells, np.zeros(n, dtype=bool)])
        self._aspect_ratio = np.concatenate([self._aspect_ratio, np.full(n, 1.1)])
        self._orientation = np.concatenate([self._orientation, np.zeros(n)])

    # ── Rendering ──

    def _render_full(self):
        """Pre-render all channels at internal resolution."""
        self._bf_full = self._render_phase_contrast()
        self._nuc_full = self._render_calcofluor()
        self._mem_full = self._render_gfp()

    def _s(self, val):
        """Scale world coordinate to internal resolution (int)."""
        return int(round(val * self.internal_scale))

    def _sf(self, val):
        """Scale world coordinate to internal resolution (float)."""
        return val * self.internal_scale

    def _draw_ellipse(self, img, cx, cy, r, aspect, angle_rad, color, thickness):
        """Draw an ellipse with the cell's aspect ratio and orientation."""
        axes = (int(round(r * aspect)), r)
        angle_deg = np.degrees(angle_rad)
        cv2.ellipse(img, (cx, cy), axes, angle_deg, 0, 360, color, thickness,
                     cv2.LINE_AA)

    def _fill_ellipse(self, img, cx, cy, r, aspect, angle_rad, color):
        """Fill an ellipse with the cell's aspect ratio and orientation."""
        axes = (int(round(r * aspect)), r)
        angle_deg = np.degrees(angle_rad)
        cv2.ellipse(img, (cx, cy), axes, angle_deg, 0, 360, color, -1,
                     cv2.LINE_AA)

    def _render_phase_contrast(self) -> np.ndarray:
        """Render phase contrast at internal resolution.

        Realism features (based on real S. cerevisiae phase contrast imagery):
          - Slightly ellipsoidal cells (axis ratio 1.0-1.2)
          - Smooth phase halo: Gaussian-like falloff at cell boundary
          - Shade-off: radial gradient, center approaches background intensity
          - Phase-dark vs phase-light cells: metabolic heterogeneity
          - Vacuoles as phase-bright inclusions (low RI, watery lumen)
          - Lipid droplets as very bright refractile puncta (high RI)
          - Cytoplasm granularity (mitochondria, vesicles, ER)
          - Faint nucleus shadow
        """
        s = self.internal_scale
        bg_val = 128.0
        img = np.full((self._ih, self._iw), bg_val, dtype=np.float32)

        # Background texture: low-frequency illumination non-uniformity + fine noise
        bg_coarse = self._noise_rng.normal(0, 1, (self.height // 4,
                                                   self.width // 4)).astype(np.float32)
        bg_coarse = cv2.resize(bg_coarse, (self._iw, self._ih),
                               interpolation=cv2.INTER_LINEAR)
        bg_coarse = cv2.GaussianBlur(bg_coarse, (0, 0), 8.0 * s)
        img += bg_coarse * 3.0

        bg_fine = self._noise_rng.normal(0, 2.5, (self.height, self.width)).astype(np.float32)
        if s > 1:
            bg_fine = cv2.resize(bg_fine, (self._iw, self._ih),
                                 interpolation=cv2.INTER_LINEAR)
        img += bg_fine

        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            r = max(3, self._s(self.radius[i]))
            asp = self._aspect_ratio[i] if i < len(self._aspect_ratio) else 1.0
            ori = self._orientation[i] if i < len(self._orientation) else 0.0
            pc = self._phase_contrast[i] if i < len(self._phase_contrast) else 0.8

            is_dead = (i < len(self._dead_cells) and self._dead_cells[i])

            # Precompute bud geometry (needed for figure-8 drawing order)
            has_bud = self.bud_size[i] > 0
            if has_bud:
                bud_r = max(2, int(round(self.radius[i] * self.bud_size[i] * s)))
                bud_dist = self.radius[i] * s * asp + bud_r * 0.6
                bx = cx + int(np.cos(self.bud_angle[i]) * bud_dist)
                by = cy + int(np.sin(self.bud_angle[i]) * bud_dist)

            if is_dead:
                self._fill_ellipse(img, cx, cy, r + 1, asp, ori, 118)
                self._draw_ellipse(img, cx, cy, r + 2, asp, ori, 126, 1)
                if has_bud:
                    cv2.circle(img, (bx, by), bud_r, 118, -1)
            else:
                # ── Phase contrast shade-off parameters ──
                # Real yeast are relatively transparent in PC
                edge_val = 80 + (1.0 - pc) * 20    # 80 (dark) → 100 (light)
                center_val = 120 + (1.0 - pc) * 4  # 120 (dark) → 124 (light)

                # ── Halo profile ──
                ht = max(1, s)
                halo_profile = [
                    (5 * ht, 133),
                    (4 * ht, 140),
                    (3 * ht, 155),
                    (2 * ht, 172),
                    (1 * ht, 188),
                ]

                # Bud phase contrast params (buds slightly less dense)
                if has_bud:
                    bud_pc = pc * 0.8
                    b_edge = 80 + (1.0 - bud_pc) * 20
                    b_center = 120 + (1.0 - bud_pc) * 4

                # ── Figure-8 drawing order: bud halo FIRST ──
                # Drawing bud halo before mother body means the mother's
                # shade-off overwrites halo in the overlap region, creating
                # a smooth connected figure-8 contour.
                if has_bud:
                    for dr, hval in halo_profile:
                        cv2.circle(img, (bx, by), bud_r + dr, hval, -1,
                                   cv2.LINE_AA)

                # Mother halo
                for dr, hval in halo_profile:
                    self._fill_ellipse(img, cx, cy, r + dr, asp, ori, hval)

                # Mother shade-off
                n_bands = 7
                for k in range(n_bands):
                    frac = 1.0 - k / n_bands
                    band_r = max(2, int(r * frac))
                    t = k / (n_bands - 1)
                    val = edge_val + t * (center_val - edge_val)
                    self._fill_ellipse(img, cx, cy, band_r, asp, ori, val)

                # Bud shade-off (drawn after mother — fills over mother edge)
                if has_bud:
                    for k in range(5):
                        frac = 1.0 - k / 5
                        br = max(2, int(bud_r * frac))
                        t = k / 4
                        cv2.circle(img, (bx, by), br,
                                   b_edge + t * (b_center - b_edge), -1,
                                   cv2.LINE_AA)

                    # Bridge: fill neck to connect mother and bud smoothly
                    neck_cx = cx + int(np.cos(self.bud_angle[i]) * r * asp)
                    neck_cy = cy + int(np.sin(self.bud_angle[i]) * r)
                    neck_fill_r = max(2, int(min(r, bud_r) * 0.35))
                    neck_val = (edge_val + center_val) / 2
                    cv2.circle(img, (neck_cx, neck_cy), neck_fill_r,
                               neck_val, -1, cv2.LINE_AA)

                    # Thin perpendicular constriction at neck (chitin ring)
                    perp_x = -np.sin(self.bud_angle[i])
                    perp_y = np.cos(self.bud_angle[i])
                    constr_w = int(min(r, bud_r) * 0.5)
                    n1 = (neck_cx + int(perp_x * constr_w),
                          neck_cy + int(perp_y * constr_w))
                    n2 = (neck_cx - int(perp_x * constr_w),
                          neck_cy - int(perp_y * constr_w))
                    cv2.line(img, n1, n2, edge_val - 5, max(1, s))

                # ── Vacuoles: phase-bright (low RI → bright in PC) ──
                if i < len(self._vacuoles):
                    cos_o, sin_o = np.cos(ori), np.sin(ori)
                    for vx, vy, vr in self._vacuoles[i]:
                        wx = vx * r * cos_o - vy * r * sin_o
                        wy = vx * r * sin_o + vy * r * cos_o
                        vac_cx = cx + int(round(wx))
                        vac_cy = cy + int(round(wy))
                        vac_r = max(2, int(round(vr * r)))
                        vac_bright = 155 + min(10, int(vr * 20))
                        cv2.circle(img, (vac_cx, vac_cy), vac_r, vac_bright,
                                   -1, cv2.LINE_AA)
                        cv2.circle(img, (vac_cx, vac_cy), vac_r, 172, 1,
                                   cv2.LINE_AA)

                # ── Lipid droplets: very bright refractile puncta ──
                if i < len(self._lipid_droplets):
                    cos_o, sin_o = np.cos(ori), np.sin(ori)
                    for dx, dy in self._lipid_droplets[i]:
                        wx = dx * r * cos_o - dy * r * sin_o
                        wy = dx * r * sin_o + dy * r * cos_o
                        lx = cx + int(round(wx))
                        ly = cy + int(round(wy))
                        dr = max(1, s // 2)
                        cv2.circle(img, (lx, ly), dr, 215, -1, cv2.LINE_AA)

                # ── Nucleus (eccentric — offset from center) ──
                nuc_r = max(2, int(r * 0.28))
                nuc_val = center_val - 8 * pc
                nuc_off = r * 0.25
                if has_bud:
                    # Nucleus migrates toward bud neck during division
                    nuc_dx = np.cos(self.bud_angle[i]) * nuc_off * 0.5
                    nuc_dy = np.sin(self.bud_angle[i]) * nuc_off * 0.5
                else:
                    # Eccentric position based on orientation
                    nuc_dx = np.cos(ori) * nuc_off * 0.3
                    nuc_dy = np.sin(ori) * nuc_off * 0.3
                cv2.circle(img, (cx + int(nuc_dx), cy + int(nuc_dy)),
                           nuc_r, nuc_val, -1)

        # ── Cytoplasm granularity: fine noise inside cells ──
        cell_mask = img < 115
        granularity = self._noise_rng.normal(0, 4.0,
                                             img.shape).astype(np.float32)
        img[cell_mask] += granularity[cell_mask]

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_calcofluor(self) -> np.ndarray:
        """Render Calcofluor White staining at internal resolution."""
        s = self.internal_scale
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            r = max(3, self._s(self.radius[i]))
            asp = self._aspect_ratio[i] if i < len(self._aspect_ratio) else 1.0
            ori = self._orientation[i] if i < len(self._orientation) else 0.0

            is_dead = (i < len(self._dead_cells) and self._dead_cells[i])

            if is_dead:
                # Dead cells: permeable walls stain brighter with Calcofluor
                self._fill_ellipse(img, cx, cy, r, asp, ori, 120)
                self._draw_ellipse(img, cx, cy, r, asp, ori, 180, max(1, s))
            else:
                # Faint cell wall staining (chitin in wall)
                self._draw_ellipse(img, cx, cy, r, asp, ori, 35, max(1, s))

            # Bud scars: bright chitin rings on cell surface
            for scar_angle in self._bud_scars[i]:
                sx = cx + int(np.cos(scar_angle) * r * asp * 0.95)
                sy = cy + int(np.sin(scar_angle) * r * 0.95)
                scar_r = max(1, r // 5)
                cv2.circle(img, (sx, sy), scar_r, 220, max(1, s))

            # Active bud neck: very bright chitin ring
            if self.bud_size[i] > 0:
                nx = cx + int(np.cos(self.bud_angle[i]) * r * asp * 0.95)
                ny = cy + int(np.sin(self.bud_angle[i]) * r * 0.95)
                cv2.circle(img, (nx, ny), max(2, r // 3), 255, max(1, s))

                # Bud wall (fainter)
                bud_r = max(2, int(round(self.radius[i] * self.bud_size[i] * s)))
                bud_dist = self.radius[i] * s * asp + bud_r * 0.6
                bx = cx + int(np.cos(self.bud_angle[i]) * bud_dist)
                by = cy + int(np.sin(self.bud_angle[i]) * bud_dist)
                cv2.circle(img, (bx, by), bud_r,
                           120 if is_dead else 25, max(1, s))

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_gfp(self) -> np.ndarray:
        """Render cytoplasmic GFP fluorescence at internal resolution.

        Vacuoles and nucleus appear dark (GFP excluded).
        """
        s = self.internal_scale
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            r = max(3, self._s(self.radius[i]))
            asp = self._aspect_ratio[i] if i < len(self._aspect_ratio) else 1.0
            ori = self._orientation[i] if i < len(self._orientation) else 0.0
            intensity = self.gfp_intensity[i] if self.gfp_expression else 0

            if intensity > 0:
                # Cytoplasmic fill (ellipse)
                self._fill_ellipse(img, cx, cy, r, asp, ori, intensity)

                # Nuclear exclusion (eccentric — matches phase contrast)
                nuc_r = max(2, int(r * 0.28))
                has_bud = self.bud_size[i] > 0
                nuc_off = r * 0.25
                if has_bud:
                    nuc_dx = np.cos(self.bud_angle[i]) * nuc_off * 0.5
                    nuc_dy = np.sin(self.bud_angle[i]) * nuc_off * 0.5
                else:
                    nuc_dx = np.cos(ori) * nuc_off * 0.3
                    nuc_dy = np.sin(ori) * nuc_off * 0.3
                cv2.circle(img, (cx + int(nuc_dx), cy + int(nuc_dy)),
                           nuc_r, intensity * 0.15, -1)

                # Vacuole exclusion (dark inclusions — GFP can't enter)
                if i < len(self._vacuoles):
                    cos_o, sin_o = np.cos(ori), np.sin(ori)
                    for vx, vy, vr in self._vacuoles[i]:
                        wx = vx * r * cos_o - vy * r * sin_o
                        wy = vx * r * sin_o + vy * r * cos_o
                        vac_cx = cx + int(round(wx))
                        vac_cy = cy + int(round(wy))
                        vac_r = max(2, int(round(vr * r)))
                        cv2.circle(img, (vac_cx, vac_cy), vac_r,
                                   intensity * 0.1, -1)

                # Bud also fluoresces (slightly dimmer — smaller volume)
                if self.bud_size[i] > 0:
                    bud_r = max(2, int(round(self.radius[i] * self.bud_size[i] * s)))
                    bud_dist = self.radius[i] * s * asp + bud_r * 0.6
                    bx = cx + int(np.cos(self.bud_angle[i]) * bud_dist)
                    by = cy + int(np.sin(self.bud_angle[i]) * bud_dist)
                    cv2.circle(img, (bx, by), bud_r, intensity * 0.75, -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    # ── SimulationBridge interface ──

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()

        # Auto-step
        if (self.auto_step and self._snap_count > 0
                and self._snap_count % self.snaps_per_step == 0):
            self.step()

        # Lazy re-render if dirty (dynamics changed state)
        if self._dirty:
            self._render_full()
            self._dirty = False

        self._snap_count += 1

        # Select channel
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
        if self._pipeline.photobleach_rate > 0 and self.mode > 0:
            crop = self._pipeline.apply_with_bleach(crop, exposure_ms=exposure)
        else:
            crop = self._pipeline.apply(crop)
        return crop

    def _crop_fov(self, full):
        """Crop FOV from internal-resolution buffer, resize to viewport."""
        s = self.internal_scale
        ih, iw = full.shape[:2]
        out_w, out_h = self.viewport_width, self.viewport_height
        obj = self.current_objectiv

        fov_map = {100: 64, 40: 128, 20: 256}
        fov_world = fov_map.get(obj, min(512, self.width))

        fov_int = fov_world * s

        cx_world = int(self.camera_offset[0]) + out_w // 2
        cy_world = int(self.camera_offset[1]) + out_h // 2
        cx_int = int(cx_world * s)
        cy_int = int(cy_world * s)

        half = fov_int // 2
        x0 = max(0, min(cx_int - half, iw - fov_int))
        y0 = max(0, min(cy_int - half, ih - fov_int))

        crop = full[y0:y0 + fov_int, x0:x0 + fov_int].copy()

        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            bg = 128 if self.mode == 0 else 0
            padded = np.full((fov_int, fov_int), bg, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        if crop.shape[0] > out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        elif crop.shape[0] < out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop

    def _update_mode(self):
        """Update rendering mode via _mode_map lookup."""
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            self.mode = 0
            return
        filt = self.state_devices["Filter Wheel"]
        led = self.state_devices["LED"]
        filter_label = filt.get("label", filt.get("Label", ""))
        led_label = led.get("label", led.get("Label", ""))

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
        if "Objective" not in self.state_devices:
            return
        obj = self.state_devices["Objective"]
        lbl = obj.get("Label", obj.get("label", ""))
        if lbl in self._objectif_dict:
            self.current_objectiv = self._objectif_dict[lbl]
            self._dof = self._dof_table.get(self.current_objectiv, 6.0)

    def set_focal_plane(self, z: float):
        self.focal_plane = z

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply defocus blur based on distance from focal plane."""
        dz = abs(self.focal_plane - self.tissue_z)
        half_dof = self._dof / 2.0
        if dz <= half_dof:
            return img
        sigma = min((dz - half_dof) * self._blur_scale_table.get(
            self.current_objectiv, 0.5), 30.0)
        if sigma < 0.3:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigma)

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift in µm."""
        return self.tissue_z

    def reset_z_drift(self):
        """Reset Z-drift to initial position."""
        self.tissue_z = 0.0

    def enable_photobleaching(self, rate: float = 0.001):
        """Enable photobleaching on fluorescence channels.

        Args:
            rate: Fractional signal loss per exposure-ms (0.001 = slow, 0.01 = fast).
        """
        self._pipeline.photobleach_rate = rate

    def reset_photobleaching(self):
        """Reset accumulated photobleaching."""
        self._pipeline.reset_bleach()

    def update_state(self, dict_state: dict):
        self.state_devices = dict_state

    def update(self, dt: float = 0.016):
        self.step(dt)

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ── Ground truth ──

    def get_ground_truth(self) -> dict:
        """Return ground truth for grading."""
        alive_mask = self.alive[:self.n_cells]
        n_alive = int(alive_mask.sum())

        # Count budding cells (exclude dead)
        budding = sum(1 for i in range(self.n_cells)
                      if self.alive[i] and not self._dead_cells[i]
                      and self.phase[i] > 0)
        n_live = sum(1 for i in range(self.n_cells)
                     if self.alive[i] and not self._dead_cells[i])
        bud_index = budding / n_live if n_live > 0 else 0

        # Dead cell count
        n_dead = int(self._dead_cells[:self.n_cells].sum())

        # Phase distribution (live cells only)
        phase_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(self.n_cells):
            if self.alive[i] and not self._dead_cells[i]:
                phase_counts[self.phase[i]] += 1

        # Arrested cells count
        n_arrested = 0
        if self._drug_active and self._drug_arrest_phase is not None:
            n_arrested = sum(
                1 for i in range(self.n_cells)
                if self.alive[i] and not self._dead_cells[i]
                and self.phase[i] == self._drug_arrest_phase
                and self._arrest_time[i] > 0
            )

        # Bud scar counts
        scar_counts = [len(self._bud_scars[i])
                       for i in range(self.n_cells) if self.alive[i]]
        mean_scars = np.mean(scar_counts) if scar_counts else 0
        max_scars = max(scar_counts) if scar_counts else 0

        gt = {
            "n_cells": n_alive,
            "n_live": n_live,
            "n_dead": n_dead,
            "n_budding": budding,
            "bud_index": round(bud_index, 3),
            "phase_distribution": {
                "G1": phase_counts[0],
                "S": phase_counts[1],
                "G2": phase_counts[2],
                "M": phase_counts[3],
            },
            "mean_bud_scars": round(float(mean_scars), 1),
            "max_bud_scars": int(max_scars),
            "time": round(self._time, 2),
            "positions": [(float(self.x[i]), float(self.y[i]))
                          for i in range(self.n_cells) if self.alive[i]],
        }

        if self._drug_active:
            gt["drug"] = self._drug_name
            gt["arrest_phase"] = self._drug_arrest_phase
            gt["n_arrested"] = n_arrested

        return gt

    def add_channel(self, channel_id: int, name: str, filter_label: str,
                    led_label: str, image: np.ndarray = None):
        """Register an extra imaging channel."""
        self._extra_channels[channel_id] = {
            "name": name,
            "filter": filter_label,
            "led": led_label,
            "image": image,
        }
