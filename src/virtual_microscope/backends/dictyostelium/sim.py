"""
DictyosteliumSim — Dictyostelium discoideum collective aggregation simulator.

Simulates social amoebae that aggregate via cAMP-mediated chemotaxis:
  - Individual cells with random motility
  - Pacemaker cells emit periodic cAMP pulses that propagate as waves
  - Signal relay: cells amplify cAMP waves (excitable medium)
  - Chemotaxis: cells migrate up cAMP gradients → streaming → mound formation
  - cAMP field rendered as fluorescent reporter channel

Channels:
  - mode 0: Dark-field — cells as bright dots on dark background
  - mode 1: GFP fluorescence — GFP+ cell bodies
  - mode 2: cAMP reporter — cAMP field intensity

Usage via SimulationBridge:
    sim = DictyosteliumSim(n_cells=300, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class DictyosteliumSim(SimBase):
    """Dictyostelium discoideum aggregation simulation.

    Models the starvation response: scattered amoebae begin signaling via
    cAMP, form streams, and aggregate into mounds around pacemaker centers.
    """

    continuous = True

    STATE_VEGETATIVE = 0
    STATE_STREAMING = 1
    STATE_AGGREGATED = 2

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_cells: int = 300,
        n_pacemakers: int = 3,
        seed: int = 42,
        fixed_dt: float = 0.0,
        internal_scale: int = 4,
        # cAMP wave parameters
        relay_radius: float = 40.0,     # cell-to-cell signaling radius (world px)
        relay_refractory: int = 8,      # refractory period after relay (steps)
        pulse_period: int = 10,         # steps between pacemaker pulses
        # Cell motility
        random_speed: float = 0.8,      # random walk speed (px/step)
        chemo_speed: float = 3.0,       # max chemotaxis speed (px/step)
        # cAMP field (for visualization + gradient computation)
        camp_diffusion: float = 15.0,   # field diffusion coefficient
        camp_decay: float = 0.03,       # field decay rate per step
        camp_reporter_gain: float = 1.0,  # multiplier for cAMP reporter brightness
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=fixed_dt,
            auto_step=False, snaps_per_step=2,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,      # GFP
                ("mScarlet3(569/582)", "ORANGE"): 2,   # cAMP-reporter
            },
        )

        self._noise_rng = np.random.default_rng(seed + 7777)
        self._step_count = 0
        self.n_cells = n_cells

        # Auto-step dt override
        self.auto_step_dt = 1.0

        # Temperature
        self._temperature = 22.0

        # ── Cell state arrays ──
        margin = 30
        self._cell_x = self.rng.uniform(margin, world_size - margin, n_cells).astype(np.float64)
        self._cell_y = self.rng.uniform(margin, world_size - margin, n_cells).astype(np.float64)
        self._cell_state = np.full(n_cells, self.STATE_VEGETATIVE, dtype=np.int32)
        self._cell_gfp = self.rng.random(n_cells) < 0.7
        self._cell_size = self.rng.uniform(3.0, 5.0, n_cells)
        self._cell_heading = self.rng.uniform(0, 2 * np.pi, n_cells)

        # Excitable relay state per cell
        self._cell_excited = np.zeros(n_cells, dtype=bool)     # currently firing
        self._cell_refractory = np.zeros(n_cells, dtype=np.float64)
        self._relay_radius = relay_radius
        self._relay_refractory = relay_refractory
        self._random_speed = random_speed
        self._chemo_speed = chemo_speed
        self._cell_streaming_time = np.zeros(n_cells, dtype=np.int32)

        # Per-cell chemotactic direction: persistent gradient toward wave source
        self._chemo_dx = np.zeros(n_cells, dtype=np.float64)
        self._chemo_dy = np.zeros(n_cells, dtype=np.float64)
        self._wave_source_x = np.zeros(n_cells, dtype=np.float64)  # best-known source position
        self._wave_source_y = np.zeros(n_cells, dtype=np.float64)
        self._has_source = np.zeros(n_cells, dtype=bool)

        # ── Pacemakers ──
        self._n_pacemakers = n_pacemakers
        self._pacemaker_indices = self._select_pacemakers(n_pacemakers)
        self._pacemaker_timer = self.rng.uniform(0, pulse_period, n_pacemakers)
        self._pulse_period = pulse_period

        # ── Wave ring history (for dark-field OD visualization) ──
        # Each entry: (pacemaker_x, pacemaker_y, pulse_time)
        self._pulse_history: list[tuple[float, float, float]] = []
        self._wave_speed = relay_radius * 0.8  # apparent wave speed (world px/step)
        self._wave_ring_width = 30.0  # width of each OD ring (world px)
        self._wave_accum = 0.0  # accumulates dt; fires one hop per sim-second

        # ── cAMP visualization field ──
        self._field_scale = 4
        self._fw = world_size // self._field_scale
        self._fh = world_size // self._field_scale
        self._camp = np.zeros((self._fh, self._fw), dtype=np.float64)
        self._camp_diffusion = camp_diffusion
        self._camp_decay = camp_decay
        self._camp_reporter_gain = camp_reporter_gain

        # ── Mound tracking ──
        self._mound_centers = []

        # ── SLM optogenetics (bPAC: light → cAMP) ──
        self._slm_mask = None  # world-coordinate bool mask
        self._slm_excitation_strength = 5.0  # cAMP units per illuminated step

        # ── Optical pipeline ──
        self._pipeline = OpticalPipeline()

    def _select_pacemakers(self, n: int) -> np.ndarray:
        """Select well-spaced pacemaker cells using farthest-first."""
        if n <= 0:
            return np.array([], dtype=np.int32)

        cx, cy = self.width / 2, self.height / 2
        dists_to_center = np.hypot(self._cell_x - cx, self._cell_y - cy)
        # Pick from inner 2/3 of the field
        candidates = np.argsort(dists_to_center)[:self.n_cells * 2 // 3]
        selected = [self.rng.choice(candidates)]

        for _ in range(n - 1):
            min_dists = np.full(self.n_cells, np.inf)
            for s in selected:
                d = np.hypot(self._cell_x - self._cell_x[s],
                             self._cell_y - self._cell_y[s])
                min_dists = np.minimum(min_dists, d)
            for s in selected:
                min_dists[s] = -1
            selected.append(int(np.argmax(min_dists)))

        return np.array(selected, dtype=np.int32)

    # ── Temperature ──

    def _temp_rate_factor(self) -> float:
        temp = self._temperature
        if temp < 10:
            return 0.1
        factor = 2.0 ** ((temp - 22) / 10.0)
        if temp > 30:
            factor *= max(0.1, 1.0 - (temp - 30) * 0.15)
        return factor

    # ── SLM mapping ──

    def _map_slm_to_world(self, mask: np.ndarray) -> np.ndarray:
        """Map viewport-space SLM mask to world-coordinate bool array."""
        obj = self.current_objectiv
        if obj == 100:
            fov_world = 64
        elif obj == 40:
            fov_world = 128
        elif obj == 20:
            fov_world = 256
        else:
            fov_world = min(512, self.width)

        cx = int(self.camera_offset[0]) + self.viewport_width // 2
        cy = int(self.camera_offset[1]) + self.viewport_height // 2

        mask_fov = cv2.resize(
            mask.astype(np.uint8), (fov_world, fov_world),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        world_mask = np.zeros((self.height, self.width), dtype=bool)
        half = fov_world // 2
        x0 = max(0, min(cx - half, self.width - fov_world))
        y0 = max(0, min(cy - half, self.height - fov_world))

        wx1 = min(self.width, x0 + fov_world)
        wy1 = min(self.height, y0 + fov_world)
        mw = wx1 - x0
        mh = wy1 - y0
        world_mask[y0:y0 + mh, x0:x0 + mw] = mask_fov[:mh, :mw]
        return world_mask

    def _apply_slm_excitation(self):
        """Excite cells under SLM illumination (bPAC optogenetics).

        Light-activated adenylyl cyclase produces cAMP locally, exciting
        illuminated cells and making them act as artificial signal sources.
        The centroid of the illuminated region becomes the wave source.
        """
        if self._slm_mask is None:
            return

        # Find illuminated region centroid
        ys, xs = np.where(self._slm_mask)
        if len(xs) == 0:
            return
        src_x = float(np.mean(xs))
        src_y = float(np.mean(ys))

        # Excite non-refractory cells within illuminated region
        for i in range(self.n_cells):
            ix = int(self._cell_x[i])
            iy = int(self._cell_y[i])
            if (0 <= ix < self.width and 0 <= iy < self.height
                    and self._slm_mask[iy, ix]
                    and self._cell_refractory[i] <= 0):
                self._cell_excited[i] = True
                self._cell_refractory[i] = self._relay_refractory
                self._wave_source_x[i] = src_x
                self._wave_source_y[i] = src_y
                self._has_source[i] = True

        # Also deposit cAMP on the field for visualization
        camp_ys = ys // self._field_scale
        camp_xs = xs // self._field_scale
        camp_ys = np.clip(camp_ys, 0, self._fh - 1)
        camp_xs = np.clip(camp_xs, 0, self._fw - 1)
        np.add.at(self._camp, (camp_ys, camp_xs), self._slm_excitation_strength / len(xs))

    # ── Dynamics ──

    def step(self, dt: float = 1.0):
        """Advance simulation by one time step."""
        if self.fixed_dt > 0:
            dt = self.fixed_dt
        self._time += dt
        self._step_count += 1

        temp_factor = self._temp_rate_factor()

        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        # 1. SLM optogenetic stimulation (bPAC: light → local cAMP)
        if self._slm_mask is not None:
            self._apply_slm_excitation()

        # 2. Pacemaker pulses: excite nearby cells
        self._pacemaker_step(temp_factor, dt)

        # 3. Wave propagation: one relay hop per sim-second.
        # Accumulator ensures the same apparent wave speed regardless of tick rate.
        self._wave_accum += dt
        while self._wave_accum >= 1.0:
            self._wave_accum -= 1.0
            self._wave_propagation_step()

        # 4. Update cAMP field (for visualization + gradient)
        self._update_camp_field(dt, temp_factor)

        # 5. Chemotaxis: move cells toward cAMP sources
        self._move_cells(dt, temp_factor)

        # 6. Update cell states
        self._update_states()

        # 7. Decrement refractory counters (dt-scaled)
        mask = self._cell_refractory > 0
        self._cell_refractory[mask] -= dt

    def _pacemaker_step(self, temp_factor: float, dt: float = 1.0):
        """Pacemaker cells periodically excite nearby cells."""
        for i, cell_idx in enumerate(self._pacemaker_indices):
            self._pacemaker_timer[i] -= dt
            if self._pacemaker_timer[i] <= 0:
                self._pacemaker_timer[i] = self._pulse_period
                # Excite this cell (trigger relay cascade)
                if self._cell_refractory[cell_idx] <= 0:
                    self._cell_excited[cell_idx] = True
                    self._cell_refractory[cell_idx] = self._relay_refractory
                    # Pacemaker is its own wave source
                    self._wave_source_x[cell_idx] = self._cell_x[cell_idx]
                    self._wave_source_y[cell_idx] = self._cell_y[cell_idx]
                    self._has_source[cell_idx] = True
                    # Record pulse for dark-field wave ring visualization
                    self._pulse_history.append((
                        float(self._cell_x[cell_idx]),
                        float(self._cell_y[cell_idx]),
                        self._time,
                    ))
        # Prune old pulses (keep only those whose wavefront is still in field)
        max_radius = self.width * 1.5
        self._pulse_history = [
            (px, py, pt) for px, py, pt in self._pulse_history
            if (self._time - pt) * self._wave_speed < max_radius
        ]

    def _wave_propagation_step(self):
        """Propagate cAMP wave: excited cells excite non-refractory neighbors.

        The wavefront advances one hop per step. Old excited cells are cleared
        and replaced by newly excited neighbors, creating a propagating wave
        at ~relay_radius px per step (limited by refractory period to prevent
        backward propagation).
        """
        excited_idx = np.where(self._cell_excited)[0]
        if len(excited_idx) == 0:
            return

        new_excited = np.zeros(self.n_cells, dtype=bool)

        for ei in excited_idx:
            ex, ey = self._cell_x[ei], self._cell_y[ei]
            dx = self._cell_x - ex
            dy = self._cell_y - ey
            dists = np.sqrt(dx * dx + dy * dy)

            # Excite non-refractory, non-excited cells within radius
            can_relay = ((dists < self._relay_radius)
                         & (self._cell_refractory <= 0)
                         & ~self._cell_excited)
            new_excited |= can_relay

            # Propagate wave source info: newly excited cells inherit the
            # source position from the cell that excited them.
            # This means all cells in a wave trace back to the pacemaker.
            src_x = self._wave_source_x[ei] if self._has_source[ei] else ex
            src_y = self._wave_source_y[ei] if self._has_source[ei] else ey
            newly_relayed = can_relay & ~self._has_source
            self._wave_source_x[newly_relayed] = src_x
            self._wave_source_y[newly_relayed] = src_y
            self._has_source[newly_relayed] = True

        # Clear old wavefront (they've already propagated)
        self._cell_excited[:] = False
        # Set new wavefront
        self._cell_excited[new_excited] = True
        self._cell_refractory[new_excited] = self._relay_refractory

        # Transition vegetative → streaming on first excitation
        veg_and_excited = new_excited & (self._cell_state == self.STATE_VEGETATIVE)
        self._cell_state[veg_and_excited] = self.STATE_STREAMING

    def _update_camp_field(self, dt: float, temp_factor: float):
        """Update cAMP visualization field from excited cells."""
        # Add cAMP at excited cell positions
        excited_idx = np.where(self._cell_excited)[0]
        for i in excited_idx:
            fx = int(self._cell_x[i] / self._field_scale)
            fy = int(self._cell_y[i] / self._field_scale)
            fx = np.clip(fx, 1, self._fw - 2)
            fy = np.clip(fy, 1, self._fh - 2)
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    w = 3.0 if (ddx == 0 and ddy == 0) else 1.0
                    self._camp[fy + ddy, fx + ddx] += w

        # Diffuse
        sigma = np.sqrt(2 * self._camp_diffusion * dt * temp_factor) / self._field_scale
        if sigma > 0.3:
            self._camp = cv2.GaussianBlur(
                self._camp, (0, 0), sigma, borderType=cv2.BORDER_REFLECT
            )
        # Decay
        self._camp *= (1.0 - self._camp_decay * dt * temp_factor)
        np.clip(self._camp, 0, 30.0, out=self._camp)

    def _move_cells(self, dt: float, temp_factor: float):
        """Move cells: random walk + chemotaxis toward aggregation centers."""
        for i in range(self.n_cells):
            if self._cell_state[i] == self.STATE_AGGREGATED:
                # Slow drift toward wave source (mound coalescence)
                if self._has_source[i]:
                    dx_src = self._wave_source_x[i] - self._cell_x[i]
                    dy_src = self._wave_source_y[i] - self._cell_y[i]
                    dist_src = np.hypot(dx_src, dy_src) + 1e-8
                    drift = 0.5 * temp_factor * min(1.0, dist_src / 20.0)
                    self._cell_x[i] += drift * (dx_src / dist_src) + self.rng.normal(0, 0.1)
                    self._cell_y[i] += drift * (dy_src / dist_src) + self.rng.normal(0, 0.1)
                else:
                    self._cell_x[i] += self.rng.normal(0, 0.15)
                    self._cell_y[i] += self.rng.normal(0, 0.15)
                continue

            # Chemotaxis toward known wave source (pacemaker center)
            if self._has_source[i]:
                dx_src = self._wave_source_x[i] - self._cell_x[i]
                dy_src = self._wave_source_y[i] - self._cell_y[i]
                dist_src = np.hypot(dx_src, dy_src) + 1e-8
                chemo_vx = self._chemo_speed * (dx_src / dist_src) * temp_factor
                chemo_vy = self._chemo_speed * (dy_src / dist_src) * temp_factor
                # Slow down as cell approaches source
                chemo_strength = min(1.0, dist_src / 30.0)
                chemo_vx *= chemo_strength
                chemo_vy *= chemo_strength
            else:
                chemo_vx = 0.0
                chemo_vy = 0.0

            # Random walk with persistence
            self._cell_heading[i] += self.rng.normal(0, 0.4)
            rand_vx = self._random_speed * np.cos(self._cell_heading[i]) * temp_factor
            rand_vy = self._random_speed * np.sin(self._cell_heading[i]) * temp_factor

            # Streaming cells: strong chemotaxis, weak random
            if self._cell_state[i] == self.STATE_STREAMING:
                vx = chemo_vx * 1.0 + rand_vx * 0.2
                vy = chemo_vy * 1.0 + rand_vy * 0.2
            else:
                vx = chemo_vx * 0.3 + rand_vx * 1.0
                vy = chemo_vy * 0.3 + rand_vy * 1.0

            self._cell_x[i] += vx * dt
            self._cell_y[i] += vy * dt

            # Boundary reflection
            margin = 10
            if self._cell_x[i] < margin:
                self._cell_x[i] = margin + 1
                self._cell_heading[i] = self.rng.uniform(-np.pi / 2, np.pi / 2)
            elif self._cell_x[i] > self.width - margin:
                self._cell_x[i] = self.width - margin - 1
                self._cell_heading[i] = self.rng.uniform(np.pi / 2, 3 * np.pi / 2)
            if self._cell_y[i] < margin:
                self._cell_y[i] = margin + 1
                self._cell_heading[i] = self.rng.uniform(0, np.pi)
            elif self._cell_y[i] > self.height - margin:
                self._cell_y[i] = self.height - margin - 1
                self._cell_heading[i] = self.rng.uniform(-np.pi, 0)

    def _update_states(self):
        """Update cell states: streaming cells with enough neighbors → aggregated."""
        streaming_mask = self._cell_state == self.STATE_STREAMING
        self._cell_streaming_time[streaming_mask] += 1

        for i in range(self.n_cells):
            if self._cell_state[i] != self.STATE_STREAMING:
                continue
            if self._cell_streaming_time[i] < 20:
                continue

            dx = self._cell_x - self._cell_x[i]
            dy = self._cell_y - self._cell_y[i]
            dists = np.sqrt(dx * dx + dy * dy)
            nearby = (dists < 20) & (self._cell_state >= self.STATE_STREAMING)
            neighbors = np.sum(nearby) - 1

            if neighbors >= 6:
                self._cell_state[i] = self.STATE_AGGREGATED

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — SLM mask is cached between snaps."""
        self.step(dt)

    # ── Rendering ──

    def _cell_shape(self, i: int) -> tuple:
        """Return (major_axis, minor_axis, angle_deg) for cell i.

        Streaming cells elongate along their heading (real Dictyostelium
        stretch from ~10µm round to ~15-20µm polarized during chemotaxis).
        Excited cells momentarily round up (cAMP response).
        """
        r = max(2, self._s(self._cell_size[i]))
        angle = self._cell_heading[i] * 180 / np.pi

        if self._cell_state[i] == self.STATE_AGGREGATED:
            r = int(r * 1.3)
            aspect = 0.7 + 0.15 * abs(np.sin(i * 0.37))
        elif self._cell_state[i] == self.STATE_STREAMING:
            if self._cell_excited[i]:
                aspect = 0.8
            else:
                aspect = 0.45 + 0.1 * abs(np.sin(i * 0.37))
            r = int(r * 1.15)
        else:
            aspect = 0.6 + 0.3 * abs(np.sin(self._cell_heading[i] + i * 0.1))

        axes = (r, max(2, int(r * aspect)))
        return axes, angle

    def _amoeboid_contour(self, i: int) -> np.ndarray | None:
        """Generate irregular amoeboid contour with pseudopod bumps.

        Real Dictyostelium cells have 1-3 pseudopods extending from an
        irregular cell body — NOT smooth ellipses. This creates a polygon
        contour with bumps along the leading edge.
        Returns int32 array of shape (N, 1, 2) for cv2.fillPoly, or None if
        cell is too small at current scale.
        """
        axes, angle_deg = self._cell_shape(i)
        major, minor = axes
        if major < 3:
            return None

        angle_rad = angle_deg * np.pi / 180.0
        cx = self._s(self._cell_x[i])
        cy = self._s(self._cell_y[i])

        # Per-cell deterministic seed for consistent shape per cell
        seed_val = (i * 7919 + 3571) & 0xFFFFFF
        local_rng = np.random.default_rng(seed_val)

        # Base ellipse with angular perturbation
        n_pts = 24
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

        # Radial perturbation: smooth bumps for pseudopods
        # Streaming/vegetative cells get 1-3 pseudopod lobes
        state = self._cell_state[i]
        if state == self.STATE_AGGREGATED:
            # Compact in mound — minimal perturbation
            n_lobes = 0
            perturb_amp = 0.08
        elif state == self.STATE_STREAMING:
            # Leading pseudopod at front + smaller lateral ones
            n_lobes = 1 + int(local_rng.random() < 0.4)  # 1-2 lobes
            perturb_amp = 0.20
        else:
            # Vegetative: 1-3 pseudopods, more irregular
            n_lobes = 1 + int(local_rng.random() * 2.5)  # 1-3 lobes
            perturb_amp = 0.25

        # Smooth base perturbation (general irregularity)
        r_perturb = np.ones(n_pts)
        harmonics = local_rng.normal(0, perturb_amp, 4)
        phases = local_rng.uniform(0, 2 * np.pi, 4)
        for h_idx in range(4):
            freq = h_idx + 2
            r_perturb += harmonics[h_idx] * np.cos(freq * theta + phases[h_idx])

        # Add pseudopod lobes (bumps in specific directions)
        for lobe in range(n_lobes):
            if lobe == 0:
                # Leading pseudopod: in direction of heading (theta=0 in cell frame)
                lobe_angle = 0.0 + local_rng.normal(0, 0.3)
                lobe_width = 0.6  # radians
                lobe_height = 0.35 + local_rng.random() * 0.2
            else:
                # Lateral pseudopods
                lobe_angle = local_rng.uniform(-np.pi, np.pi)
                lobe_width = 0.5
                lobe_height = 0.15 + local_rng.random() * 0.15

            # Smooth bump using cos²
            d_angle = np.arctan2(np.sin(theta - lobe_angle),
                                 np.cos(theta - lobe_angle))
            bump = np.maximum(0, np.cos(d_angle / lobe_width * np.pi / 2)) ** 2
            bump[np.abs(d_angle) > lobe_width] = 0
            r_perturb += bump * lobe_height

        # Construct contour in cell-local frame, then rotate
        rx = major * r_perturb * np.cos(theta)
        ry = minor * r_perturb * np.sin(theta)

        # Rotate to heading
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        px = cx + (rx * cos_a - ry * sin_a)
        py = cy + (rx * sin_a + ry * cos_a)

        pts = np.stack([px, py], axis=-1).astype(np.int32)
        return pts.reshape(-1, 1, 2)

    def _compute_od_wave_overlay(self) -> np.ndarray | None:
        """Compute analytical OD wave rings for dark-field visualization.

        Each pacemaker pulse creates an expanding ring of optical density
        change.  Real dark-field waves have an asymmetric profile:
          - Leading edge (just ahead of wavefront): narrow DARK band
            from cell cringe/rounding (less scattering)
          - Trailing edge (behind wavefront): broader BRIGHT band
            from cell elongation/chemotaxis (more scattering)

        Wave collision annihilation: in real excitable media, wavefronts
        from different pacemakers annihilate when they meet.  Each pixel
        is assigned to its nearest pacemaker (Voronoi territory) and only
        renders waves from that pacemaker, creating natural collision
        boundaries between territories.

        Computed at world resolution (not internal_scale) for performance,
        then upscaled — OD waves are broad features that don't need pixel
        resolution.

        Returns internal-resolution float32 image, or None.
        """
        if not self._pulse_history:
            return None

        s = self.internal_scale
        # Compute at world resolution for speed, upscale at end
        h, w = self.height, self.width

        yy, xx = np.ogrid[0:h, 0:w]

        # --- Assign each pixel to the nearest pacemaker (Voronoi) ---
        pm_positions = []
        pm_lookup = {}
        for px, py, _pt in self._pulse_history:
            key = (round(px, 1), round(py, 1))
            if key not in pm_lookup:
                pm_lookup[key] = len(pm_positions)
                pm_positions.append((px, py))

        if len(pm_positions) == 1:
            nearest_pm = np.zeros((h, w), dtype=np.int32)
        else:
            min_dist_sq = np.full((h, w), np.inf, dtype=np.float32)
            nearest_pm = np.zeros((h, w), dtype=np.int32)
            for pm_idx, (px, py) in enumerate(pm_positions):
                d2 = ((xx - px).astype(np.float32)) ** 2 + \
                     ((yy - py).astype(np.float32)) ** 2
                mask = d2 < min_dist_sq
                min_dist_sq = np.where(mask, d2, min_dist_sq)
                nearest_pm = np.where(mask, pm_idx, nearest_pm)

        # --- Render waves, respecting territories ---
        overlay = np.zeros((h, w), dtype=np.float32)
        bright_sigma = self._wave_ring_width * 0.6
        dark_sigma = self._wave_ring_width * 0.25

        for px, py, pt in self._pulse_history:
            dt_elapsed = self._time - pt
            if dt_elapsed < 0.5:
                continue

            front_r = dt_elapsed * self._wave_speed

            dist = np.sqrt(
                ((xx - px).astype(np.float32)) ** 2 +
                ((yy - py).astype(np.float32)) ** 2
            )

            delta = front_r - dist

            bright = np.exp(-0.5 * (np.maximum(0, delta) / bright_sigma) ** 2)
            bright *= (delta > 0).astype(np.float32)

            dark = np.exp(-0.5 * (np.minimum(0, delta) / dark_sigma) ** 2)
            dark *= (delta <= 0).astype(np.float32)

            ring = bright * 10.0 - dark * 3.5

            ring *= np.exp(-dist / (self.width * 0.7))
            ring *= np.exp(-dt_elapsed / (self._pulse_period * 5))

            key = (round(px, 1), round(py, 1))
            pm_idx = pm_lookup[key]
            territory = nearest_pm == pm_idx
            overlay[territory] += ring[territory]

        overlay = np.clip(overlay, -5, 12)

        # Upscale to internal resolution
        if s > 1:
            overlay = cv2.resize(overlay, (self._iw, self._ih),
                                 interpolation=cv2.INTER_LINEAR)
        return overlay

    def _render_darkfield(self) -> np.ndarray:
        """Render dark-field: cells as bright scattering objects on dark background.

        Real dark-field Dictyostelium imaging:
        - Dark substrate with faint agar texture (position-seeded)
        - OD waves as broad brightness modulation bands
        - Individual cells scatter light → bright edges, dimmer centers
        - Nucleus visible as darker region within bright cell body
        - Streaming cells connected by pseudopod extensions
        - Excited cells momentarily brighter (cringe/rounding increases scattering)
        """
        s = self.internal_scale

        # 1. Agar substrate: position-seeded texture (consistent across frames)
        sub_rng = np.random.default_rng(12345)
        bg_coarse = sub_rng.normal(22, 2.0, (self.height // 4, self.width // 4)).astype(np.float32)
        bg_coarse = cv2.resize(bg_coarse, (self._iw, self._ih), interpolation=cv2.INTER_LINEAR)
        bg_fine = self._noise_rng.normal(0, 1.8, (self._ih, self._iw)).astype(np.float32)
        img = np.clip(bg_coarse + bg_fine, 5, 40).astype(np.float64)

        # 2. OD wave overlay
        if self._pulse_history:
            wave_overlay = self._compute_od_wave_overlay()
            if wave_overlay is not None:
                img += wave_overlay

        # 3. Draw pseudopod streams UNDER cells (so cells draw on top)
        self._draw_streams(img)

        # 4. Draw cells with dark-field scattering and internal structure
        for i in range(self.n_cells):
            cx = self._s(self._cell_x[i])
            cy = self._s(self._cell_y[i])
            axes, angle_deg = self._cell_shape(i)

            # Base scattering intensity depends on state
            if self._cell_state[i] == self.STATE_AGGREGATED:
                body_int = 160
                edge_int = 210
            elif self._cell_state[i] == self.STATE_STREAMING:
                body_int = 130 if not self._cell_excited[i] else 160
                edge_int = 180 if not self._cell_excited[i] else 210
            else:
                body_int = 85 if not self._cell_excited[i] else 115
                edge_int = 120 if not self._cell_excited[i] else 155

            contour = self._amoeboid_contour(i)
            if contour is not None:
                # Fill cell body
                cv2.fillPoly(img, [contour], float(body_int))

                # Bright edge (dark-field: edges scatter more)
                cv2.polylines(img, [contour], True, float(edge_int),
                              max(1, s // 2), lineType=cv2.LINE_AA)

                # Outer bright halo (scattering fringe)
                # Dilate contour slightly for halo
                halo_pts = contour.copy().astype(np.float64)
                centroid = halo_pts.mean(axis=0)
                halo_pts = centroid + (halo_pts - centroid) * 1.15
                cv2.polylines(img, [halo_pts.astype(np.int32)], True,
                              float(min(255, edge_int * 1.1)),
                              1, lineType=cv2.LINE_AA)

                # Nucleus: darker oval near cell center
                nuc_r = max(2, int(min(axes) * 0.35))
                # Offset nucleus slightly toward rear (away from heading)
                angle_rad = angle_deg * np.pi / 180.0
                nuc_ox = int(-np.cos(angle_rad) * axes[0] * 0.1)
                nuc_oy = int(-np.sin(angle_rad) * axes[0] * 0.1)
                nuc_int = body_int * 0.6
                cv2.circle(img, (cx + nuc_ox, cy + nuc_oy), nuc_r,
                           float(nuc_int), -1, lineType=cv2.LINE_AA)
            else:
                # Fallback for very small cells: simple circle
                r = max(2, axes[0])
                cv2.circle(img, (cx, cy), r, float(body_int), -1)
                cv2.circle(img, (cx, cy), r + 1, float(edge_int), 1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _draw_streams(self, img: np.ndarray):
        """Draw tapered pseudopod connections between nearby streaming cells.

        Real Dictyostelium streams show cells connected by thin pseudopod
        extensions — visible as tapered bright bridges between adjacent cells.
        Thicker near cell body, thinner at the midpoint. Only drawn between
        cells heading toward the same wave source.
        """
        s = self.internal_scale
        streaming = np.where(
            (self._cell_state == self.STATE_STREAMING) & self._has_source
        )[0]
        if len(streaming) < 2:
            return

        sx = self._cell_x[streaming]
        sy = self._cell_y[streaming]
        src_x = self._wave_source_x[streaming]
        src_y = self._wave_source_y[streaming]

        max_dist = 25.0
        drawn = set()
        for idx in range(len(streaming)):
            dx = sx - sx[idx]
            dy = sy - sy[idx]
            dists = np.sqrt(dx * dx + dy * dy)
            dsrc = np.hypot(src_x - src_x[idx], src_y - src_y[idx])
            candidates = (dists > 0.1) & (dists < max_dist) & (dsrc < 50)
            if not np.any(candidates):
                continue
            cdists = dists.copy()
            cdists[~candidates] = 999
            nearest = np.argmin(cdists)

            edge = (min(idx, nearest), max(idx, nearest))
            if edge in drawn:
                continue
            drawn.add(edge)

            x1, y1 = self._s(sx[idx]), self._s(sy[idx])
            x2, y2 = self._s(sx[nearest]), self._s(sy[nearest])

            # Tapered pseudopod: thick near cells, thin at midpoint
            # Draw as a filled polygon (trapezoid)
            dx_line = x2 - x1
            dy_line = y2 - y1
            length = max(1, np.hypot(dx_line, dy_line))
            nx, ny = -dy_line / length, dx_line / length  # perpendicular

            w_end = max(1, s * 0.8)    # width near cell body
            w_mid = max(1, s * 0.3)    # width at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2

            pts = np.array([
                [x1 + nx * w_end, y1 + ny * w_end],
                [mx + nx * w_mid, my + ny * w_mid],
                [x2 + nx * w_end, y2 + ny * w_end],
                [x2 - nx * w_end, y2 - ny * w_end],
                [mx - nx * w_mid, my - ny * w_mid],
                [x1 - nx * w_end, y1 - ny * w_end],
            ], dtype=np.int32).reshape(-1, 1, 2)

            # Stream intensity: dimmer than cells, brighter than background
            stream_int = 55.0
            cv2.fillPoly(img, [pts], stream_int)
            # Bright edge on pseudopod
            cv2.polylines(img, [pts], True, 70.0, 1, lineType=cv2.LINE_AA)

    def _render_gfp(self) -> np.ndarray:
        """Render GFP fluorescence: cytoplasmic GFP in GFP+ cells.

        GFP is a cytoplasmic marker — nucleus appears as dark hole inside
        bright cell body.  Cell shape uses amoeboid contour. Per-cell
        brightness heterogeneity from expression level variation.
        """
        s = self.internal_scale
        img = np.zeros((self._ih, self._iw), dtype=np.float64)

        # Low autofluorescence background
        bg = self._noise_rng.normal(4, 1.0, (self._ih, self._iw)).astype(np.float64)
        img += np.clip(bg, 0, 10)

        for i in range(self.n_cells):
            if not self._cell_gfp[i]:
                continue
            cx = self._s(self._cell_x[i])
            cy = self._s(self._cell_y[i])

            # Per-cell expression heterogeneity
            expr_var = 0.7 + 0.6 * abs(np.sin(i * 1.37 + 0.5))

            if self._cell_state[i] == self.STATE_AGGREGATED:
                intensity = 200 * expr_var
            elif self._cell_state[i] == self.STATE_STREAMING:
                intensity = 165 * expr_var
            else:
                intensity = 130 * expr_var

            contour = self._amoeboid_contour(i)
            if contour is not None:
                # Cell body
                cv2.fillPoly(img, [contour], float(intensity))

                # Nuclear exclusion: GFP doesn't enter nucleus → dark hole
                axes, angle_deg = self._cell_shape(i)
                nuc_r = max(2, int(min(axes) * 0.35))
                angle_rad = angle_deg * np.pi / 180.0
                nuc_ox = int(-np.cos(angle_rad) * axes[0] * 0.1)
                nuc_oy = int(-np.sin(angle_rad) * axes[0] * 0.1)
                nuc_int = intensity * 0.15  # residual fluorescence in nucleus
                cv2.circle(img, (cx + nuc_ox, cy + nuc_oy), nuc_r,
                           float(nuc_int), -1, lineType=cv2.LINE_AA)
            else:
                r = max(2, self._cell_shape(i)[0][0])
                cv2.circle(img, (cx, cy), r, float(intensity), -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_camp(self) -> np.ndarray:
        """Render cAMP reporter: field intensity as fluorescence.

        cAMP-GFP or Flamindo2 reporter shows the field as fluorescence.
        Background has low autofluorescence noise.
        """
        field_norm = np.clip(self._camp * self._camp_reporter_gain / 8.0, 0, 1.0)
        field_8bit = (field_norm * 220).astype(np.float32)
        img = cv2.resize(field_8bit, (self._iw, self._ih),
                         interpolation=cv2.INTER_LINEAR)
        # Autofluorescence + read noise
        noise = self._noise_rng.normal(0, 1.5, (self.height, self.width)).astype(np.float32)
        if self.internal_scale > 1:
            noise = cv2.resize(noise, (self._iw, self._ih),
                               interpolation=cv2.INTER_LINEAR)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img

    def _read_temperature(self):
        temp_state = self.state_devices.get("Temperature", {})
        label = temp_state.get("label", temp_state.get("Label", "22°C"))
        try:
            self._temperature = float(label.replace("°C", ""))
        except (ValueError, AttributeError):
            pass

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()
        self._read_temperature()

        # Map SLM mask to world coords (bPAC optogenetics)
        if mask is not None and np.any(mask):
            self._slm_mask = self._map_slm_to_world(mask)
        else:
            self._slm_mask = None

        # Auto-step
        if (self.auto_step and self._snap_count > 0
                and self._snap_count % self.snaps_per_step == 0):
            self.step(self.auto_step_dt if self.fixed_dt <= 0 else self.fixed_dt)

        self._snap_count += 1

        if self.mode == 1:
            full = self._render_gfp()
        elif self.mode == 2:
            full = self._render_camp()
        else:
            full = self._render_darkfield()

        crop = self._crop_fov(full)
        crop = self._apply_defocus(crop)
        if self._pipeline.photobleach_rate > 0 and self.mode > 0:
            crop = self._pipeline.apply_with_bleach(crop, exposure_ms=kwargs.get("exposure", exposure))
        else:
            crop = self._pipeline.apply(crop)
        return crop

    # ── Ground truth ──

    def get_ground_truth(self) -> dict:
        n_veg = int(np.sum(self._cell_state == self.STATE_VEGETATIVE))
        n_stream = int(np.sum(self._cell_state == self.STATE_STREAMING))
        n_agg = int(np.sum(self._cell_state == self.STATE_AGGREGATED))

        agg_indices = np.where(self._cell_state == self.STATE_AGGREGATED)[0]
        mound_centers = self._find_mound_centers(agg_indices)

        return {
            "n_cells": self.n_cells,
            "n_vegetative": n_veg,
            "n_streaming": n_stream,
            "n_aggregated": n_agg,
            "n_pacemakers": self._n_pacemakers,
            "pacemaker_positions": [
                (float(self._cell_x[i]), float(self._cell_y[i]))
                for i in self._pacemaker_indices
            ],
            "mound_centers": mound_centers,
            "n_mounds": len(mound_centers),
            "camp_max": round(float(np.max(self._camp)), 2),
            "camp_mean": round(float(np.mean(self._camp)), 4),
            "step_count": self._step_count,
            "time": round(self._time, 1),
        }

    def _find_mound_centers(self, agg_indices: np.ndarray) -> list:
        """Find mound centers by iterative mean-shift-like clustering."""
        if len(agg_indices) == 0:
            return []

        ax = self._cell_x[agg_indices]
        ay = self._cell_y[agg_indices]
        assigned = np.full(len(agg_indices), -1, dtype=np.int32)
        cluster_id = 0
        radius = 80.0

        for i in range(len(agg_indices)):
            if assigned[i] >= 0:
                continue
            # Seed from this cell, iterate to find cluster center
            cx, cy = ax[i], ay[i]
            for _ in range(5):  # mean-shift iterations
                dists = np.hypot(ax - cx, ay - cy)
                mask = dists < radius
                if not np.any(mask):
                    break
                cx = np.mean(ax[mask])
                cy = np.mean(ay[mask])

            # Assign all cells near final center
            dists = np.hypot(ax - cx, ay - cy)
            near = dists < radius
            unassigned_near = near & (assigned < 0)
            if np.sum(unassigned_near) >= 5:
                assigned[unassigned_near] = cluster_id
                cluster_id += 1

        centers = []
        for cid in range(cluster_id):
            mask = assigned == cid
            n = int(np.sum(mask))
            if n >= 5:
                centers.append({
                    "x": round(float(np.mean(ax[mask])), 1),
                    "y": round(float(np.mean(ay[mask])), 1),
                    "n_cells": n,
                })

        return centers

