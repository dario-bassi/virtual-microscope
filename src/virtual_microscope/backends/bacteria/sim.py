"""
BacteriaSim — E. coli run-and-tumble motility simulator.

Simulates rod-shaped bacteria with:
  - Run-and-tumble motility (straight runs + random reorientations)
  - Exponential growth (cell division)
  - Chemotaxis (biased random walk toward attractant)
  - Phase contrast and fluorescence rendering

Usage via SimulationBridge:
    sim = BacteriaSim(world_size=512, n_cells=30, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class BacteriaSim(SimBase):
    """E. coli bacteria simulation with run-and-tumble motility.

    Channels:
      - mode 0: Phase contrast — dark rods with halo on gray background
      - mode 1: GFP fluorescence — bright rods on dark background
      - mode 2: DAPI (DNA stain) — bright spots at cell centers
    """

    continuous = True

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_cells: int = 30,
        seed: int = 42,
        fixed_dt: float = 0.0,
        internal_scale: int = 4,
        boundary_mode: str = "reflect",
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=fixed_dt,
            auto_step=True, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,  # GFP
                ("SCFP2(434/474)", "UV"): 2,       # DAPI
            },
        )

        self._noise_rng = np.random.default_rng(seed + 7777)
        self.boundary_mode = boundary_mode  # "reflect" or "periodic"

        # ── Bacteria state arrays ──
        margin = 60
        self.x = self.rng.uniform(margin, world_size - margin, n_cells).astype(np.float64)
        self.y = self.rng.uniform(margin, world_size - margin, n_cells).astype(np.float64)
        self.theta = self.rng.uniform(0, 2 * np.pi, n_cells)
        self.is_running = np.ones(n_cells, dtype=bool)
        self.run_timer = self.rng.exponential(1.0, n_cells)
        self.tumble_timer = np.zeros(n_cells, dtype=np.float64)
        self.division_timer = self.rng.uniform(5, 25, n_cells)
        self.alive = np.ones(n_cells, dtype=bool)
        self._prev_conc = np.zeros(n_cells, dtype=np.float64)  # for chemotaxis
        self.max_cells = 2000  # cap to prevent runaway growth

        # ── Per-cell size variation ──
        # Each cell has a size_factor (length multiplier) and width_factor
        # drawn from normal distribution. Inherited at division with ±5% mutation.
        self.size_factor = np.clip(
            self.rng.normal(1.0, 0.15, n_cells), 0.6, 1.5
        ).astype(np.float64)
        self.width_factor = np.clip(
            self.rng.normal(1.0, 0.10, n_cells), 0.7, 1.3
        ).astype(np.float64)

        # ── Motion parameters ──
        self.run_speed = 15.0        # px/s (~20 µm/s at 1 px/µm)
        self.mean_run_duration = 1.0  # seconds
        self.mean_tumble_duration = 0.1
        self.division_time = 20.0     # mean seconds between divisions

        # ── Colony / biofilm mode ──
        # When True, daughters stay near parents (reduced motility).
        # Bacteria that get close enough "stick" (motility → 0).
        self.colony_mode = False
        self.adhesion_radius = 6.0     # px — bacteria within this distance stick
        self.colony_speed_factor = 0.05  # stuck bacteria move at 5% normal speed
        self.colony_id = np.full(n_cells, -1, dtype=int)  # -1 = planktonic
        self._next_colony_id = 0

        # ── Chemotaxis ──
        self._attractant = np.zeros((world_size, world_size), dtype=np.float32)
        self._stim_mask = None
        self.chemotaxis_strength = 0.0  # 0 = no chemotaxis, >0 = biased
        self.chemotaxis_mode = "drift"  # "drift" = old gradient drift, "brw" = biased random walk
        self._attractant_sources = []   # list of (cx, cy, radius, emission_rate)
        self._attractant_decay = 0.02   # decay rate per second
        self.slm_injection_rate = 0.2   # attractant added per step at SLM mask

        # ── Phototaxis (SLM-controlled) ──
        self._phototaxis_enabled = False
        self._phototaxis_speed_factor = 0.2  # speed in light (fraction of normal)
        self._slm_mask = None                # world-coordinate boolean mask

        # ── Cold-induced filamentation ──
        # Tracks cumulative elongation per cell at cold temperatures.
        # Below 10°C, cells grow longer without dividing (max 2× extra length).
        self._cold_elongation = np.zeros(n_cells, dtype=np.float64)

        # ── Antibiotic disk diffusion ──
        self._antibiotic_disks = []  # list of (cx, cy, radius, strength)
        self._kill_zone = np.zeros((world_size, world_size), dtype=np.float32)

        # ── GFP expression dynamics ──
        # Per-cell GFP expression level (0.0 = uninduced, 1.0 = max)
        self.gfp_expression = np.zeros(n_cells, dtype=np.float64)
        self._gfp_induced = False
        self._gfp_induction_rate = 0.05    # per step (0→1 in ~20 steps)
        self._gfp_max_expression = 1.0
        self._gfp_decay_rate = 0.0         # 0 = no decay (stable GFP)
        self._gfp_basal_expression = 0.0   # leaky promoter baseline

        # ── Optical pipelines per channel ──
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 300),
            1: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 301),
            2: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 302),
        }

    # ── Antibiotic disk diffusion ──

    def place_antibiotic_disk(self, cx, cy, radius, strength=1.0):
        """Place an antibiotic disk at (cx, cy) with given radius.

        Bacteria within the zone of inhibition (ZOI) are killed over time.
        The ZOI extends beyond the physical disk due to diffusion.

        Args:
            cx, cy: center of the disk in world coordinates
            radius: physical disk radius (pixels). ZOI ≈ 1.5-2x this.
            strength: kill rate (1.0 = strong antibiotic)
        """
        self._antibiotic_disks.append((cx, cy, radius, strength))
        self._update_kill_zone()

    def _update_kill_zone(self):
        """Recompute the antibiotic concentration field from all disks."""
        self._kill_zone[:] = 0
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        for cx, cy, radius, strength in self._antibiotic_disks:
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            # Antibiotic concentration: 1.0 inside disk, exponential decay outside
            # ZOI extends to ~2x disk radius
            conc = np.where(
                dist <= radius,
                strength,
                strength * np.exp(-2.0 * (dist - radius) / radius),
            )
            self._kill_zone = np.maximum(self._kill_zone, conc)

    # ── Attractant sources ──

    def add_attractant_source(self, cx, cy, radius=30, emission_rate=0.5):
        """Place a continuous attractant source at (cx, cy).

        The source emits attractant in a Gaussian pattern centered at (cx, cy).
        Combined with chemotaxis_strength > 0, bacteria will bias their movement
        toward the source.

        Args:
            cx, cy: center of the source in world coordinates
            radius: spread of the source (Gaussian sigma in pixels)
            emission_rate: attractant emitted per second (0.1-1.0 typical)
        """
        self._attractant_sources.append((cx, cy, radius, emission_rate))

    def add_attractant_gradient(self, direction="right", strength=1.0):
        """Set up a linear attractant gradient across the field.

        Args:
            direction: "right", "left", "up", "down"
            strength: peak attractant concentration (0.0-1.0)
        """
        if direction == "right":
            grad = np.linspace(0, strength, self.width, dtype=np.float32)
            self._attractant = np.tile(grad, (self.height, 1))
        elif direction == "left":
            grad = np.linspace(strength, 0, self.width, dtype=np.float32)
            self._attractant = np.tile(grad, (self.height, 1))
        elif direction == "down":
            grad = np.linspace(0, strength, self.height, dtype=np.float32)
            self._attractant = np.tile(grad[:, None], (1, self.width))
        elif direction == "up":
            grad = np.linspace(strength, 0, self.height, dtype=np.float32)
            self._attractant = np.tile(grad[:, None], (1, self.width))

    def enable_phototaxis(self, speed_factor: float = 0.2):
        """Enable SLM-controlled phototaxis (speed reduction in light).

        When the SLM illuminates a region, bacteria in that region swim slower.
        Because they spend more time in slow regions, they naturally accumulate
        in illuminated areas. This models proteorhodopsin-based speed control
        (Frangipane et al. 2018 — light-controlled E. coli density patterns).

        Args:
            speed_factor: fraction of normal speed in illuminated region.
                          0.2 = 80% speed reduction → strong accumulation.
                          0.5 = 50% reduction → moderate accumulation.
        """
        self._phototaxis_enabled = True
        self._phototaxis_speed_factor = max(0.05, min(1.0, speed_factor))

    def _emit_attractant(self, dt):
        """Emit attractant from all sources, diffuse, and decay."""
        yy, xx = np.mgrid[0:self.height, 0:self.width]
        for cx, cy, radius, rate in self._attractant_sources:
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            emission = rate * dt * np.exp(-dist2 / (2 * radius ** 2))
            self._attractant += emission.astype(np.float32)

        # Diffuse (gentle blur)
        if self._attractant.max() > 0.001:
            self._attractant = cv2.GaussianBlur(
                self._attractant, (0, 0), 3.0
            )
            # Decay
            self._attractant *= (1.0 - self._attractant_decay * dt)
            self._attractant = np.clip(self._attractant, 0, 5.0)

    def _render_attractant_full(self) -> np.ndarray:
        """Render attractant field as a heatmap (green channel) at internal res."""
        s = self.internal_scale
        # Normalize to 0-255
        field = self._attractant
        if field.max() < 0.001:
            return np.zeros((self._ih, self._iw, 3), dtype=np.uint8)

        normalized = (field / max(field.max(), 0.01) * 200).clip(0, 255)
        gray = normalized.astype(np.uint8)

        # Upscale to internal resolution
        heatmap = cv2.resize(gray, (self._iw, self._ih), interpolation=cv2.INTER_LINEAR)

        # Green-channel heatmap
        img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)
        img[:, :, 1] = heatmap  # green

        # Overlay bacteria positions as bright white dots
        length, width = self._cell_size_px()
        for i in range(len(self.x)):
            if not self.alive[i]:
                continue
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            cv2.circle(img, (cx, cy), max(2, width), (255, 255, 255), -1)

        return img

    def induce_gfp(self, rate=0.05, max_expression=1.0, decay_rate=0.0,
                   basal=0.0):
        """Enable GFP induction (like adding IPTG to a pLac-GFP culture).

        After calling this, each cell's gfp_expression increases by `rate`
        per step until reaching `max_expression`. The GFP fluorescence in
        the nucleus-channel (mode 1) is proportional to expression level.

        Args:
            rate: Expression increase per step (0.05 → ~20 steps to max).
            max_expression: Maximum GFP expression (0.0-1.0).
            decay_rate: GFP degradation per step (0 = stable GFP like GFPmut3).
            basal: Leaky promoter baseline expression (0 = tight control).
        """
        self._gfp_induced = True
        self._gfp_induction_rate = rate
        self._gfp_max_expression = max_expression
        self._gfp_decay_rate = decay_rate
        self._gfp_basal_expression = basal

    def get_gfp_state(self):
        """Return current GFP expression state for ground truth.

        Returns dict with per-cell expression and population statistics.
        """
        alive_mask = self.alive
        alive_expr = self.gfp_expression[alive_mask]
        return {
            "induced": self._gfp_induced,
            "n_expressing": int((alive_expr > 0.1).sum()),
            "n_bright": int((alive_expr > 0.5).sum()),
            "mean_expression": round(float(alive_expr.mean()), 3) if len(alive_expr) > 0 else 0.0,
            "max_expression": round(float(alive_expr.max()), 3) if len(alive_expr) > 0 else 0.0,
        }

    def enable_attractant_channel(self):
        """Replace the DAPI channel (mode 2 / membrane-channel) with attractant heatmap.

        After calling this, membrane-channel will render the attractant field
        as a green heatmap with bacteria overlaid as white dots.
        """
        self._attractant_channel_enabled = True

    @property
    def _attractant_channel_active(self):
        return getattr(self, "_attractant_channel_enabled", False)

    # ── Temperature response ──

    def _get_temperature(self) -> float:
        """Read current temperature (°C) from the controller device."""
        if "Temperature" not in self.state_devices:
            return 20.0
        return float(self.state_devices["Temperature"].get("label", "20"))

    def _temp_speed_factor(self) -> float:
        """Motility multiplier.  E. coli swims fastest at ~37°C.

        Uses a Q10 ≈ 1.5 model with cold arrest below 8°C and heat-shock
        decline above 40°C.  At default 20°C, factor ≈ 0.36.
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.05  # near-arrested
        factor = 1.5 ** ((temp - 37) / 10.0)
        if temp > 40:
            factor *= max(0.1, 1.0 - (temp - 40) * 0.3)
        return factor

    def _temp_growth_factor(self) -> float:
        """Division-rate multiplier.  E. coli divides fastest at ~37°C.

        Uses a Q10 ≈ 2.0 model.  Below 8°C growth stops entirely.
        Heat shock (>40°C) sharply reduces division.
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.0  # cold arrest — no division
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 40:
            factor *= max(0.05, 1.0 - (temp - 40) * 0.4)
        return factor

    # ── Dynamics ──

    def step(self, dt: float = 1.0):
        """Advance bacteria motility + growth by one timestep."""
        if self.fixed_dt > 0:
            dt = self.fixed_dt

        # Temperature-dependent scaling
        speed_factor = self._temp_speed_factor()
        growth_factor = self._temp_growth_factor()

        # Cold-induced filamentation: below 10°C, cells elongate without
        # dividing. Rate: ~0.05 extra length per sim-second at 4°C.
        temp = self._get_temperature()
        if temp < 10.0:
            rate = 0.05 * dt * (10.0 - temp) / 6.0  # max at 4°C
            n_alive = self.alive.sum()
            if n_alive > 0 and len(self._cold_elongation) == len(self.alive):
                self._cold_elongation[self.alive] = np.minimum(
                    self._cold_elongation[self.alive] + rate, 2.0)  # max 3× total

        # Z-drift accumulation
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        alive = self.alive
        n = len(self.x)
        if n == 0:
            self._time += dt
            return

        # ── Run-and-tumble motility (vectorized) ──
        running = alive & self.is_running
        tumbling = alive & ~self.is_running

        # In colony mode, stuck bacteria have reduced motility
        if self.colony_mode:
            stuck = alive & (self.colony_id >= 0)
            # Stuck cells: tiny Brownian motion only (no runs)
            if stuck.any():
                n_stuck = stuck.sum()
                self.x[stuck] += self.rng.normal(0, 0.3, n_stuck) * dt
                self.y[stuck] += self.rng.normal(0, 0.3, n_stuck) * dt
            # Only planktonic cells can truly run
            running = running & (self.colony_id < 0)
            tumbling = tumbling & (self.colony_id < 0)

        # Running cells: move forward (speed scaled by temperature)
        if running.any():
            effective_speed = self.run_speed * speed_factor
            # Phototaxis: slow down bacteria in SLM-illuminated regions
            if (self._phototaxis_enabled and self._slm_mask is not None
                    and np.any(self._slm_mask)):
                run_idx = np.where(running)[0]
                per_cell_speed = np.full(len(self.x), effective_speed)
                for i in run_idx:
                    xi = int(np.clip(self.x[i], 0, self.width - 1))
                    yi = int(np.clip(self.y[i], 0, self.height - 1))
                    if self._slm_mask[yi, xi]:
                        per_cell_speed[i] *= self._phototaxis_speed_factor
                self.x[running] += per_cell_speed[running] * np.cos(self.theta[running]) * dt
                self.y[running] += per_cell_speed[running] * np.sin(self.theta[running]) * dt
            else:
                self.x[running] += effective_speed * np.cos(self.theta[running]) * dt
                self.y[running] += effective_speed * np.sin(self.theta[running]) * dt
            self.run_timer[running] -= dt

        # Chemotaxis: bias movement toward attractant.
        if self.chemotaxis_strength > 0 and self._attractant.max() > 0.01:
            if self.chemotaxis_mode == "brw":
                # Biased Random Walk: extend runs when swimming up gradient.
                # Compare current concentration to previous (temporal sensing).
                alive_idx = np.where(alive)[0]
                for i in alive_idx:
                    ix = int(np.clip(self.x[i], 0, self.width - 1))
                    iy = int(np.clip(self.y[i], 0, self.height - 1))
                    curr_conc = self._attractant[iy, ix]
                    dconc = curr_conc - self._prev_conc[i]
                    self._prev_conc[i] = curr_conc
                    if self.is_running[i] and dconc > 0:
                        # Swimming up gradient: extend run (suppress tumble)
                        bonus = min(self.chemotaxis_strength * dconc, 2.0)
                        self.run_timer[i] += bonus * dt
                    elif self.is_running[i] and dconc < 0:
                        # Swimming down gradient: shorten run (trigger tumble)
                        penalty = min(self.chemotaxis_strength * abs(dconc), 1.0)
                        self.run_timer[i] -= penalty * dt
            else:
                # Legacy drift mode: spatial gradient force
                grad_y, grad_x = np.gradient(self._attractant)
                alive_idx = np.where(alive)[0]
                for i in alive_idx:
                    ix = int(np.clip(self.x[i], 0, self.width - 1))
                    iy = int(np.clip(self.y[i], 0, self.height - 1))
                    gx = grad_x[iy, ix]
                    gy = grad_y[iy, ix]
                    self.x[i] += self.chemotaxis_strength * gx * dt
                    self.y[i] += self.chemotaxis_strength * gy * dt

        # Check for tumble onset
        tumble_start = running & (self.run_timer <= 0)
        if tumble_start.any():
            count = tumble_start.sum()
            self.is_running[tumble_start] = False
            self.tumble_timer[tumble_start] = self.rng.exponential(
                self.mean_tumble_duration, count
            )

        # Tumbling cells: check for run onset
        if tumbling.any():
            self.tumble_timer[tumbling] -= dt
        run_start = tumbling & (self.tumble_timer <= 0)
        if run_start.any():
            count = run_start.sum()
            self.is_running[run_start] = True
            self.theta[run_start] = self.rng.uniform(0, 2 * np.pi, count)
            self.run_timer[run_start] = self.rng.exponential(
                self.mean_run_duration, count
            )

        # ── Boundary handling ──
        if self.boundary_mode == "periodic":
            self.x[alive] = self.x[alive] % self.width
            self.y[alive] = self.y[alive] % self.height
        else:
            # Elastic reflection (default)
            mask_lo_x = alive & (self.x < 0)
            mask_hi_x = alive & (self.x >= self.width)
            mask_lo_y = alive & (self.y < 0)
            mask_hi_y = alive & (self.y >= self.height)
            self.x[mask_lo_x] = -self.x[mask_lo_x]
            self.x[mask_hi_x] = 2 * self.width - self.x[mask_hi_x] - 1
            self.y[mask_lo_y] = -self.y[mask_lo_y]
            self.y[mask_hi_y] = 2 * self.height - self.y[mask_hi_y] - 1
            self.theta[mask_lo_x | mask_hi_x] = np.pi - self.theta[mask_lo_x | mask_hi_x]
            self.theta[mask_lo_y | mask_hi_y] = -self.theta[mask_lo_y | mask_hi_y]

        # ── Colony adhesion ──
        if self.colony_mode:
            self._detect_colonies()

        # ── Cell division (rate scaled by temperature) ──
        if alive.sum() < self.max_cells and growth_factor > 0:
            self.division_timer[alive] -= dt * growth_factor
            dividing_idx = np.where(alive & (self.division_timer <= 0))[0]
            for i in dividing_idx:
                # Suppress division in antibiotic zone
                if len(self._antibiotic_disks) > 0:
                    ix = int(np.clip(self.x[i], 0, self.width - 1))
                    iy = int(np.clip(self.y[i], 0, self.height - 1))
                    if self._kill_zone[iy, ix] > 0.3:
                        self.division_timer[i] = self.rng.exponential(self.division_time)
                        continue
                self._divide(i)
                # Reset parent's timer (use index, not mask, since arrays grow)
                self.division_timer[i] = self.rng.exponential(self.division_time)

        # ── Antibiotic killing ──
        if len(self._antibiotic_disks) > 0:
            alive_idx = np.where(self.alive)[0]
            for i in alive_idx:
                ix = int(np.clip(self.x[i], 0, self.width - 1))
                iy = int(np.clip(self.y[i], 0, self.height - 1))
                kill_prob = self._kill_zone[iy, ix] * dt * 0.5
                if kill_prob > 0 and self.rng.random() < kill_prob:
                    self.alive[i] = False

        # ── SLM attractant injection ──
        if self._stim_mask is not None and np.any(self._stim_mask):
            self._attractant[self._stim_mask > 0] += self.slm_injection_rate * dt
            # Diffuse attractant (sigma=5 for broader gradient)
            self._attractant = cv2.GaussianBlur(
                self._attractant, (0, 0), 5.0
            )
            # Decay
            self._attractant *= (1.0 - 0.02 * dt)

        # ── Continuous attractant sources ──
        if self._attractant_sources:
            self._emit_attractant(dt)

        # ── GFP expression dynamics ──
        if self._gfp_induced:
            alive_mask = self.alive
            # Induction: expression increases toward max
            self.gfp_expression[alive_mask] += self._gfp_induction_rate * dt
            # Decay (for unstable GFP variants)
            if self._gfp_decay_rate > 0:
                self.gfp_expression[alive_mask] -= (
                    self._gfp_decay_rate * self.gfp_expression[alive_mask] * dt
                )
            # Basal expression floor (leaky promoter even when induced)
            if self._gfp_basal_expression > 0:
                below = self.gfp_expression[alive_mask] < self._gfp_basal_expression
                self.gfp_expression[alive_mask] = np.where(
                    below,
                    np.minimum(
                        self.gfp_expression[alive_mask] + 0.002 * dt,
                        self._gfp_basal_expression,
                    ),
                    self.gfp_expression[alive_mask],
                )
            # Clamp to [0, max]
            np.clip(self.gfp_expression, 0, self._gfp_max_expression,
                    out=self.gfp_expression)
        elif self._gfp_basal_expression > 0:
            # Leaky expression without induction
            alive_mask = self.alive
            self.gfp_expression[alive_mask] = np.minimum(
                self.gfp_expression[alive_mask] + 0.002 * dt,
                self._gfp_basal_expression,
            )

        self._time += dt

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (SLM mask cached between snaps)."""
        self.step(dt)

    def _detect_colonies(self):
        """Find bacteria close enough to stick together into colonies.

        Uses a simple pairwise distance check (O(n^2) but n is typically <500).
        Bacteria within adhesion_radius form/join the same colony.
        """
        alive_idx = np.where(self.alive)[0]
        n = len(alive_idx)
        if n < 2:
            return

        # Only check planktonic cells for new adhesion events
        for ii in range(n):
            i = alive_idx[ii]
            if self.colony_id[i] >= 0:
                continue  # already in a colony
            for jj in range(ii + 1, n):
                j = alive_idx[jj]
                dx = self.x[i] - self.x[j]
                dy = self.y[i] - self.y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < self.adhesion_radius:
                    if self.colony_id[j] >= 0:
                        # j is in a colony — join it
                        self.colony_id[i] = self.colony_id[j]
                    else:
                        # Neither in a colony — create new one
                        cid = self._next_colony_id
                        self._next_colony_id += 1
                        self.colony_id[i] = cid
                        self.colony_id[j] = cid
                    break  # i has been assigned, move to next

    def get_colony_stats(self) -> dict:
        """Get colony statistics for ground truth."""
        alive_mask = self.alive
        alive_colony = self.colony_id[alive_mask]
        planktonic = int((alive_colony < 0).sum())
        colony_cells = alive_colony[alive_colony >= 0]
        if len(colony_cells) == 0:
            return {"n_colonies": 0, "n_planktonic": planktonic,
                    "n_colony_cells": 0, "colony_sizes": []}
        unique_ids, counts = np.unique(colony_cells, return_counts=True)
        return {
            "n_colonies": len(unique_ids),
            "n_planktonic": planktonic,
            "n_colony_cells": int(colony_cells.shape[0]),
            "colony_sizes": sorted(counts.tolist(), reverse=True),
        }

    def _divide(self, i):
        """Create a daughter cell adjacent to parent cell i."""
        if len(self.x) >= self.max_cells:
            return
        offset = 4.0  # pixels separation
        angle = self.theta[i]
        nx = self.x[i] + offset * np.cos(angle)
        ny = self.y[i] - offset * np.sin(angle)

        # Append daughter
        self.x = np.append(self.x, np.clip(nx, 0, self.width - 1))
        self.y = np.append(self.y, np.clip(ny, 0, self.height - 1))
        self.theta = np.append(self.theta, self.rng.uniform(0, 2 * np.pi))
        self.is_running = np.append(self.is_running, True)
        self.run_timer = np.append(self.run_timer, self.rng.exponential(self.mean_run_duration))
        self.tumble_timer = np.append(self.tumble_timer, 0.0)
        self.division_timer = np.append(
            self.division_timer, self.rng.exponential(self.division_time)
        )
        self.alive = np.append(self.alive, True)
        self._prev_conc = np.append(self._prev_conc, 0.0)

        # Inherit size with slight mutation
        self.size_factor = np.append(
            self.size_factor,
            np.clip(self.size_factor[i] + self.rng.normal(0, 0.05), 0.6, 1.5),
        )
        self.width_factor = np.append(
            self.width_factor,
            np.clip(self.width_factor[i] + self.rng.normal(0, 0.03), 0.7, 1.3),
        )

        # Inherit GFP expression (split ~evenly between parent and daughter)
        parent_expr = self.gfp_expression[i]
        daughter_expr = parent_expr * self.rng.uniform(0.4, 0.6)
        self.gfp_expression[i] = parent_expr - daughter_expr + parent_expr * 0.5
        self.gfp_expression[i] = min(self.gfp_expression[i], self._gfp_max_expression)
        self.gfp_expression = np.append(self.gfp_expression, daughter_expr)

        # Cold elongation resets on division (daughter starts at normal size)
        self._cold_elongation[i] = 0.0  # parent resets too
        self._cold_elongation = np.append(self._cold_elongation, 0.0)

        # In colony mode: daughter immediately joins parent's colony
        if self.colony_mode:
            if self.colony_id[i] >= 0:
                self.colony_id = np.append(self.colony_id, self.colony_id[i])
            else:
                # Create a new colony for parent + daughter
                cid = self._next_colony_id
                self._next_colony_id += 1
                self.colony_id[i] = cid
                self.colony_id = np.append(self.colony_id, cid)
        else:
            self.colony_id = np.append(self.colony_id, -1)

    # ── Rendering ──

    @staticmethod
    def _capsule_pts(cx, cy, half_len, half_wid, angle_rad, n_cap=8):
        """Contour points for a capsule (stadium) shape.

        A capsule = rectangle with semicircular end caps. This gives the
        realistic blunt-ended rod shape of E. coli instead of pointed ellipses.
        """
        cap_r = half_wid
        body_half = max(0, half_len - cap_r)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        pts = []
        # Left cap (semicircle at -body_half)
        for i in range(n_cap + 1):
            theta = np.pi / 2 + i * np.pi / n_cap
            px = -body_half + cap_r * np.cos(theta)
            py = cap_r * np.sin(theta)
            pts.append((int(cx + px * cos_a - py * sin_a),
                        int(cy + px * sin_a + py * cos_a)))
        # Right cap (semicircle at +body_half)
        for i in range(n_cap + 1):
            theta = -np.pi / 2 + i * np.pi / n_cap
            px = body_half + cap_r * np.cos(theta)
            py = cap_r * np.sin(theta)
            pts.append((int(cx + px * cos_a - py * sin_a),
                        int(cy + px * sin_a + py * cos_a)))
        return np.array(pts, dtype=np.int32)

    def _draw_capsule(self, img, cx, cy, half_len, half_wid, angle_rad,
                      color, n_cap=8):
        """Draw a filled capsule on img."""
        pts = self._capsule_pts(cx, cy, half_len, half_wid, angle_rad, n_cap)
        cv2.fillPoly(img, [pts], color)

    def _cell_size_px(self, cell_idx=None):
        """Rod dimensions in internal pixels.

        If cell_idx is given, applies per-cell size variation,
        pre-division elongation, and cold-induced filamentation.
        Otherwise returns base dimensions (for attractant overlay etc.).
        """
        s = self.internal_scale
        base_length = 3 * s   # ~3 world px long → 12 internal px
        base_width = s        # ~1 world px wide → 4 internal px
        if cell_idx is not None and cell_idx < len(self.size_factor):
            length = max(2, int(base_length * self.size_factor[cell_idx]))
            width = max(1, int(base_width * self.width_factor[cell_idx]))
            # Pre-division elongation: cells in last 30% of cycle grow longer
            if (cell_idx < len(self.division_timer) and self.alive[cell_idx]
                    and self.division_timer[cell_idx] < self.division_time * 0.3):
                progress = 1.0 - self.division_timer[cell_idx] / (self.division_time * 0.3)
                length = int(length * (1.0 + 0.5 * progress))  # up to 50% longer
            # Cold-induced filamentation: below 10°C, division is blocked
            # and cells elongate without dividing. Filament length grows with
            # cumulative time spent at low temperature (up to 3× normal length).
            if hasattr(self, '_cold_elongation') and cell_idx < len(self._cold_elongation):
                length = int(length * (1.0 + self._cold_elongation[cell_idx]))
        else:
            length = max(2, base_length)
            width = max(1, base_width)
        return length, width

    def _division_progress(self, cell_idx: int) -> float:
        """Return 0.0 (just divided) to 1.0 (about to divide)."""
        if cell_idx >= len(self.division_timer) or not self.alive[cell_idx]:
            return 0.0
        remaining = self.division_timer[cell_idx]
        if remaining >= self.division_time * 0.3:
            return 0.0
        return 1.0 - remaining / (self.division_time * 0.3)

    def _render_bf_full(self) -> np.ndarray:
        """Phase contrast: dark rods with bright halo at internal resolution.

        Mimics Zernike phase contrast microscopy of E. coli:
        - Gray background (~140) with subtle texture (medium/agar)
        - Bright halo ring around each cell (phase contrast artifact)
        - Dark cell body with shade-off (slight interior brightening)
        - Ghost cells (dead) are nearly transparent (lost refractive index)
        """
        s = self.internal_scale

        # Background with subtle texture (simulates imaging medium)
        bg = 140
        img = np.full((self._ih, self._iw, 3), bg, dtype=np.uint8)
        if self._ih > 0:
            # Add very subtle background texture (fixed seed for consistency)
            bg_rng = np.random.default_rng(42)
            texture = bg_rng.normal(0, 2.0, (self._ih, self._iw)).astype(np.float32)
            # Low-pass filter for smooth texture
            texture = cv2.GaussianBlur(texture, (0, 0), 3.0)
            for c in range(3):
                img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) + texture,
                                       bg - 6, bg + 6).astype(np.uint8)

        # Draw antibiotic disks (lighter agar + white disk)
        for cx, cy, radius, strength in self._antibiotic_disks:
            dcx, dcy = self._s(cx), self._s(cy)
            dr = int(radius * s)
            cv2.circle(img, (dcx, dcy), dr, (220, 220, 220), -1)
            cv2.circle(img, (dcx, dcy), dr, (180, 180, 180), max(1, s // 2))

        halo_pad = max(2, s // 2)

        # Draw dead cells first — ghosts are nearly transparent in phase
        # contrast (lost cytoplasmic contents → reduced refractive index →
        # minimal phase shift → faint outline near background intensity)
        for i in range(len(self.x)):
            if self.alive[i]:
                continue
            length, width = self._cell_size_px(i)
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            angle_rad = float(self.theta[i]) if i < len(self.theta) else 0.0
            hl, hw = length // 2, width // 2
            # Faint bright outline (residual halo from cell membrane)
            self._draw_capsule(img, cx, cy, hl + halo_pad, hw + halo_pad,
                               angle_rad, (150, 150, 150))
            # Ghost body: near-background, slightly lighter (lower RI)
            self._draw_capsule(img, cx, cy, hl, hw, angle_rad, (145, 145, 145))

        # Draw alive cells on top with capsule-shaped phase contrast
        for i in range(len(self.x)):
            if not self.alive[i]:
                continue
            length, width = self._cell_size_px(i)
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            angle_rad = float(self.theta[i]) if i < len(self.theta) else 0.0
            hl, hw = length // 2, width // 2

            # Outer diffuse halo (slightly elliptical — wider perpendicular
            # to long axis where the refractive index gradient is steepest)
            cv2.ellipse(
                img, (cx, cy),
                (hl + halo_pad + 2, hw + halo_pad + 2),
                np.degrees(angle_rad), 0, 360, (165, 165, 165), -1,
            )
            # Inner bright halo (capsule-shaped, tight around cell edge)
            self._draw_capsule(img, cx, cy, hl + halo_pad, hw + halo_pad,
                               angle_rad, (195, 195, 195))
            # Cell body — 3-layer rendering for realistic phase contrast:
            # 1. Edge zone (darkest — phase ring acts strongest at boundary)
            self._draw_capsule(img, cx, cy, hl, hw, angle_rad, (75, 75, 75))
            # 2. Mid-body (slightly lighter interior)
            mid_hl = max(1, hl * 3 // 4)
            mid_hw = max(1, hw * 3 // 4)
            self._draw_capsule(img, cx, cy, mid_hl, mid_hw, angle_rad,
                               (90, 90, 90))
            # 3. Shade-off center (characteristic brightening near centre)
            so_hl = max(1, hl * 2 // 5)
            so_hw = max(1, hw * 2 // 3)
            self._draw_capsule(img, cx, cy, so_hl, so_hw, angle_rad,
                               (100, 100, 100))

            # Division septum: visible constriction for cells about to divide
            div_prog = self._division_progress(i)
            if div_prog > 0.6:
                perp_rad = angle_rad + np.pi / 2
                constr_depth = int(max(1, width * 0.3 * (div_prog - 0.6) / 0.4))
                px0 = int(cx - np.cos(perp_rad) * (hw + 1))
                py0 = int(cy - np.sin(perp_rad) * (hw + 1))
                px1 = int(cx + np.cos(perp_rad) * (hw + 1))
                py1 = int(cy + np.sin(perp_rad) * (hw + 1))
                cv2.line(img, (px0, py0), (px1, py1),
                         (115, 115, 115), max(1, constr_depth), cv2.LINE_AA)

        return img

    def _render_gfp_full(self) -> np.ndarray:
        """GFP fluorescence: bright capsule-shaped rods at internal resolution.

        If GFP expression dynamics are enabled, brightness is proportional
        to each cell's gfp_expression level.
        """
        # Low autofluorescence background (real media autofluoresce slightly)
        img = np.full((self._ih, self._iw, 3), 4, dtype=np.uint8)
        has_expression = self._gfp_induced or self._gfp_basal_expression > 0

        # Dead cells: dim residual GFP (leaking from lysed membrane)
        for i in range(len(self.x)):
            if self.alive[i]:
                continue
            length, width = self._cell_size_px(i)
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            angle_rad = float(self.theta[i]) if i < len(self.theta) else 0.0
            if has_expression:
                dim = int(float(35 * self.gfp_expression[i]))
            else:
                dim = 35
            if dim > 0:
                self._draw_capsule(img, cx, cy, length // 2 + 2,
                                   width // 2 + 2, angle_rad, (dim, dim, dim))

        # Alive cells: GFP brightness
        for i in range(len(self.x)):
            if not self.alive[i]:
                continue
            length, width = self._cell_size_px(i)
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            angle_rad = float(self.theta[i]) if i < len(self.theta) else 0.0

            if has_expression:
                base = int(float(200 * self.gfp_expression[i]))
                noise = int(self._noise_rng.integers(-10, 10))
                brightness = max(0, min(255, base + noise))
            else:
                brightness = int(180 + int(self._noise_rng.integers(-20, 20)))

            if brightness > 0:
                self._draw_capsule(img, cx, cy, length // 2, width // 2,
                                   angle_rad, (brightness, brightness, brightness))

        return img

    def _render_dapi_full(self) -> np.ndarray:
        """DAPI: bright spots at cell centers at internal resolution."""
        img = np.zeros((self._ih, self._iw, 3), dtype=np.uint8)

        # Dead cells: dimmer DAPI (released DNA spreads slightly)
        for i in range(len(self.x)):
            if self.alive[i]:
                continue
            _, width = self._cell_size_px(i)
            radius = max(2, width)  # slightly larger spread
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            cv2.circle(img, (cx, cy), radius, (50, 50, 50), -1)

        # Alive cells
        for i in range(len(self.x)):
            if not self.alive[i]:
                continue
            _, width = self._cell_size_px(i)
            radius = max(1, width // 2)
            cx, cy = self._s(self.x[i]), self._s(self.y[i])
            brightness = int(200 + self._noise_rng.integers(-20, 20))
            cv2.circle(img, (cx, cy), radius, (brightness, brightness, brightness), -1)

        return img

    # ── Template-method hooks ──

    def _render_for_mode(self, mode: int) -> np.ndarray:
        """Return full-resolution BGR image for the active channel."""
        if mode == 0:
            return self._render_bf_full()
        elif mode == 1:
            return self._render_gfp_full()
        elif mode == 2:
            if self._attractant_channel_active:
                return self._render_attractant_full()
            else:
                return self._render_dapi_full()
        elif mode in self._extra_channels:
            ch = self._extra_channels[mode]
            if "render" in ch and callable(ch["render"]):
                return ch["render"]()
            else:
                return ch["image"]
        else:
            return self._render_bf_full()

    def _handle_mask(self, mask) -> None:
        """Handle SLM mask for chemotaxis + phototaxis."""
        if np.any(mask):
            m = mask.astype(np.uint8)
            if m.shape[0] != self.height or m.shape[1] != self.width:
                m = cv2.resize(m, (self.width, self.height),
                               interpolation=cv2.INTER_NEAREST)
            self._stim_mask = m
            if self._phototaxis_enabled:
                self._slm_mask = self._map_slm_to_world(mask)
        else:
            # Explicit empty mask = clear SLM (turn off phototaxis)
            self._stim_mask = None
            self._slm_mask = None

    # ── Ground truth ──

    def get_ground_truth(self) -> dict:
        alive_mask = self.alive
        positions = np.column_stack([self.x[alive_mask], self.y[alive_mask]])
        gt = {
            "n_cells": int(alive_mask.sum()),
            "n_dead": int((~alive_mask).sum()),
            "positions": positions.tolist() if len(positions) > 0 else [],
            "mean_speed": round(float(self.run_speed), 1),
            "time": round(self._time, 2),
            "n_total_ever": len(self.x),
            "mean_size_factor": round(float(self.size_factor[alive_mask].mean()), 3) if alive_mask.any() else 1.0,
            "size_std": round(float(self.size_factor[alive_mask].std()), 3) if alive_mask.any() else 0.0,
        }
        # Add antibiotic disk info if present
        if self._antibiotic_disks:
            gt["antibiotic_disks"] = [
                {"cx": cx, "cy": cy, "radius": r, "strength": s}
                for cx, cy, r, s in self._antibiotic_disks
            ]
            # Measure zone of inhibition: distance from disk center to nearest alive bacterium
            for disk_info in gt["antibiotic_disks"]:
                cx, cy = disk_info["cx"], disk_info["cy"]
                if alive_mask.sum() > 0:
                    dists = np.sqrt(
                        (self.x[alive_mask] - cx) ** 2
                        + (self.y[alive_mask] - cy) ** 2
                    )
                    disk_info["zoi_radius"] = round(float(np.min(dists)), 1)
                    disk_info["zoi_diameter"] = round(float(np.min(dists)) * 2, 1)
                else:
                    disk_info["zoi_radius"] = round(float(self.width / 2), 1)
                    disk_info["zoi_diameter"] = round(float(self.width), 1)
        # Add chemotaxis info if present
        if self.chemotaxis_strength > 0:
            gt["chemotaxis_strength"] = self.chemotaxis_strength
        if self._attractant_sources:
            gt["attractant_sources"] = [
                {"cx": cx, "cy": cy, "radius": r, "emission_rate": e}
                for cx, cy, r, e in self._attractant_sources
            ]
            gt["attractant_max"] = round(float(self._attractant.max()), 3)
        # Add GFP expression info
        if self._gfp_induced or self._gfp_basal_expression > 0:
            gt["gfp"] = self.get_gfp_state()
        # Add phototaxis info
        if self._phototaxis_enabled:
            gt["phototaxis_speed_factor"] = self._phototaxis_speed_factor
            gt["slm_active"] = self._slm_mask is not None and np.any(self._slm_mask)
            if gt["slm_active"] and alive_mask.any():
                # Count bacteria in illuminated region
                n_in_light = 0
                for i in range(len(self.x)):
                    if not self.alive[i]:
                        continue
                    xi = int(np.clip(self.x[i], 0, self.width - 1))
                    yi = int(np.clip(self.y[i], 0, self.height - 1))
                    if self._slm_mask[yi, xi]:
                        n_in_light += 1
                gt["n_in_light"] = n_in_light

        return gt

    def get_cell_count(self) -> int:
        return int(self.alive.sum())
