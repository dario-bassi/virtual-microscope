"""
MicrofluidicsSim — cells flowing through microfluidic channels.

Fundamentally different from other backends: confined geometry with
pressure-driven flow. Cells move through channels, can be trapped,
sorted, or exposed to chemical gradients.

Device geometry:
  - Main channel: horizontal, ~100px wide, cells flow left→right
  - Side channels (optional): branch off for sorting/collection
  - Traps (optional): small constrictions that catch cells
  - Gradient zone: chemical gradient across channel width (top→bottom)

Physics:
  - Laminar flow: parabolic velocity profile (faster in center)
  - Cells are carried by flow, with some Brownian diffusion
  - Larger cells move slower (Stokes drag)
  - Cells can be "trapped" at constrictions if larger than trap width
  - Chemical gradient: linear across channel width, affects cell behavior

Perfusion (optional):
  - Drug flows in from left when activated via enable_perfusion()
  - Reagent advects with flow + diffuses laterally
  - Cells accumulate drug exposure over time
  - Above lethal threshold: cells die (stop moving, change fluorescence)
  - Membrane channel shows live reagent concentration field

Channels:
  - mode 0 (brightfield): Channel walls + flowing cells (phase contrast)
  - mode 1 (nucleus): DAPI — nuclear staining of flowing cells
  - mode 2 (membrane): Fluorescent dye gradient OR reagent concentration

Usage:
    sim = MicrofluidicsSim(n_cells=30, seed=42)
    sim.enable_perfusion(lethal_conc=5.0)  # optional
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class MicrofluidicsSim(SimBase):
    """Microfluidic channel simulation with flowing cells."""

    continuous = True

    def __init__(self, n_cells=30, channel_width=100, flow_speed=3.0,
                 world_size=512, seed=42, viewport_width=512, viewport_height=512,
                 n_traps=0, gradient=False, internal_scale: int = 1):
        """
        Parameters:
            n_cells: number of cells in the channel
            channel_width: width of the main channel in pixels
            flow_speed: base flow speed (px/step) at channel center
            world_size: full image size
            seed: random seed
            n_traps: number of trapping constrictions (0 = none)
            gradient: whether to add a chemical gradient across channel width
        """
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=1.0,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("SCFP2(434/474)", "UV"): 1,
                ("TagGFP2(483/506)", "GREEN"): 2,
            },
        )

        self.n_cells = n_cells
        self.channel_width = channel_width
        self.flow_speed = flow_speed
        self._seed = seed
        self.n_traps = n_traps
        self.has_gradient = gradient

        # Optical pipeline per channel
        self._pipeline_bf = OpticalPipeline.fluorescence()
        self._pipeline_nuc = OpticalPipeline.fluorescence()
        self._pipeline_mem = OpticalPipeline.fluorescence()

        # Perfusion (off by default)
        self._perfusion_enabled = False
        self._perfusion_active = False
        self._perfusion_start_time = None
        self._drug_name = "cytotoxic"
        self._lethal_conc = 5.0
        self._drug_diffusion = 2.0
        self._reagent_field = None
        self._cell_drug_exposure = None
        self._drug_flow_speed = None

        # Time (separate from SimBase._time; used for perfusion tracking)
        self.time = 0.0

        # Initialize
        self._rng = np.random.default_rng(seed)
        self._noise_rng = np.random.default_rng(seed + 999)
        self._init_geometry()
        self._init_cells()
        self._init_cell_morphology()

    def _init_geometry(self):
        """Set up channel geometry."""
        w, h = self.width, self.height
        # Main channel: horizontal, centered vertically
        self._channel_y0 = h // 2 - self.channel_width // 2
        self._channel_y1 = h // 2 + self.channel_width // 2

        # Channel walls (list of wall segments for rendering)
        self._walls = [
            (0, self._channel_y0, w, self._channel_y0),  # top wall
            (0, self._channel_y1, w, self._channel_y1),  # bottom wall
        ]

        # Traps: narrow constrictions in the channel
        self._traps = []
        if self.n_traps > 0:
            rng = self._rng
            trap_spacing = w // (self.n_traps + 1)
            for i in range(self.n_traps):
                tx = (i + 1) * trap_spacing + int(rng.normal(0, 10))
                trap_width = rng.integers(8, 16)  # narrow opening
                self._traps.append({
                    "x": tx,
                    "width": trap_width,
                    "y_top": self._channel_y0,
                    "y_bot": self._channel_y1,
                    "protrusion_top": self.channel_width // 2 - trap_width // 2 - rng.integers(0, 5),
                    "protrusion_bot": self.channel_width // 2 - trap_width // 2 - rng.integers(0, 5),
                })

        # Gradient: linear across channel width (top = low, bottom = high)
        if self.has_gradient:
            self._gradient_strength = self._rng.uniform(0.5, 1.0)
        else:
            self._gradient_strength = 0.0

    def _init_cells(self):
        """Place cells in the channel."""
        rng = self._rng
        cy_mid = (self._channel_y0 + self._channel_y1) / 2
        margin = 5

        # Cell properties
        self._cell_x = np.zeros(self.n_cells)
        self._cell_y = np.zeros(self.n_cells)
        self._cell_radius = np.zeros(self.n_cells)
        self._cell_type = []  # 'small', 'medium', 'large'
        self._cell_trapped = np.zeros(self.n_cells, dtype=bool)
        self._cell_intensity = np.zeros(self.n_cells)  # nuclear staining
        self._cell_alive = np.ones(self.n_cells, dtype=bool)

        for i in range(self.n_cells):
            # Distribute along the channel
            self._cell_x[i] = rng.uniform(20, self.width - 20)
            self._cell_y[i] = rng.uniform(
                self._channel_y0 + margin,
                self._channel_y1 - margin
            )

            # Cell types with different sizes
            ctype = rng.choice(["small", "medium", "large"], p=[0.4, 0.4, 0.2])
            self._cell_type.append(ctype)

            if ctype == "small":
                self._cell_radius[i] = rng.uniform(3, 5)
                self._cell_intensity[i] = rng.uniform(160, 200)
            elif ctype == "medium":
                self._cell_radius[i] = rng.uniform(5, 8)
                self._cell_intensity[i] = rng.uniform(175, 215)
            else:  # large
                self._cell_radius[i] = rng.uniform(8, 12)
                self._cell_intensity[i] = rng.uniform(190, 230)

        # Track cells that have exited (for counting throughput)
        self._cells_exited = 0
        self._cells_entered = 0

        # SLM release tracking
        self._release_events = []  # list of {"cell_idx", "cell_type", "time", "x", "y"}
        self._slm_release_threshold = 100  # minimum SLM pixel value to trigger release

    def _init_cell_morphology(self):
        """Pre-compute per-cell rendering details (nucleolus, texture seed)."""
        rng = self._rng
        n = self.n_cells
        # Each cell gets a nucleolus offset and size (visible at 40x+)
        self._nuc_offset_x = rng.uniform(-0.2, 0.2, n)
        self._nuc_offset_y = rng.uniform(-0.2, 0.2, n)
        self._nucleolus_r_frac = rng.uniform(0.15, 0.25, n)
        # Per-cell texture seed for chromatin pattern
        self._cell_texture_seed = rng.integers(0, 100000, n)

    # ── Perfusion ──

    def enable_perfusion(self, drug_name: str = "cytotoxic",
                         lethal_conc: float = 5.0,
                         diffusion_rate: float = 2.0,
                         drug_flow_speed: float | None = None):
        """Enable the perfusion system (drug initially off).

        Call start_perfusion() to begin flowing drug through the channel.

        Parameters:
            drug_name: label for the drug (for ground truth)
            lethal_conc: cumulative exposure that kills a cell
            diffusion_rate: lateral diffusion coefficient (px²/step)
            drug_flow_speed: override flow speed when Drug mode is active
                             (default None → uses speed_map value of 3.0)
        """
        self._perfusion_enabled = True
        self._drug_name = drug_name
        self._lethal_conc = lethal_conc
        self._drug_diffusion = diffusion_rate
        self._drug_flow_speed = drug_flow_speed
        # 1D reagent profile along channel length (x-axis)
        self._reagent_field = np.zeros(self.width, dtype=np.float32)
        self._cell_drug_exposure = np.zeros(self.n_cells, dtype=np.float64)

    def start_perfusion(self):
        """Begin flowing drug into the channel from the left inlet."""
        if not self._perfusion_enabled:
            return
        self._perfusion_active = True
        if self._perfusion_start_time is None:
            self._perfusion_start_time = self.time

    def stop_perfusion(self):
        """Stop drug flow (washout — existing reagent still advects out)."""
        self._perfusion_active = False

    def _update_reagent(self, dt: float):
        """Advect and diffuse the reagent field, update cell exposure."""
        if not self._perfusion_enabled or self._reagent_field is None:
            return

        rf = self._reagent_field

        # Inlet: constant concentration while perfusion is active
        if self._perfusion_active:
            rf[0] = 1.0

        # Advection: shift reagent rightward at mean flow speed
        # Use first-order upwind: new[x] = old[x] - v * dt * (old[x] - old[x-1]) / dx
        v_adv = self.flow_speed * 0.7  # mean flow speed (parabolic avg ≈ 0.67 * v_max)
        shift_px = v_adv * dt
        if shift_px >= 1.0:
            # Discrete shift
            n_shift = int(shift_px)
            rf[n_shift:] = rf[:-n_shift].copy()
            if self._perfusion_active:
                rf[:n_shift] = 1.0
            else:
                rf[:n_shift] = 0.0
        else:
            # Sub-pixel advection via interpolation
            new_rf = np.zeros_like(rf)
            alpha = shift_px
            new_rf[1:] = (1 - alpha) * rf[1:] + alpha * rf[:-1]
            if self._perfusion_active:
                new_rf[0] = 1.0
            rf[:] = new_rf

        # Diffusion: smooth along x (1D Gaussian blur)
        if self._drug_diffusion > 0:
            sigma = np.sqrt(2 * self._drug_diffusion * dt)
            if sigma > 0.3:
                ksize = int(sigma * 6) | 1  # ensure odd
                ksize = max(3, min(ksize, 31))
                rf[:] = cv2.GaussianBlur(
                    rf.reshape(1, -1), (ksize, 1), sigma
                ).flatten()

        # Outlet: reagent exits at right edge
        rf[-1] = rf[-2] * 0.95

        # Update cell drug exposure
        for i in range(self.n_cells):
            if not self._cell_alive[i]:
                continue
            xi = int(np.clip(self._cell_x[i], 0, self.width - 1))
            local_conc = float(rf[xi])
            self._cell_drug_exposure[i] += local_conc * dt

            # Cell death at lethal concentration
            if self._cell_drug_exposure[i] >= self._lethal_conc:
                self._cell_alive[i] = False

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 20.0
        return float(self.state_devices["Temperature"].get("label", "20"))

    def _temp_speed_factor(self) -> float:
        """Temperature-dependent factor for cell motility in the channel.

        Diffusion and active cell movement scale with temperature.
        Q10 ~ 1.5. Reference at 20°C (room temp for microfluidics).
        """
        temp = self._get_temperature()
        if temp < 8:
            return 0.1
        return 1.5 ** ((temp - 20) / 10.0)

    def _read_perfusion_state(self):
        """Read the Perfusion device and apply flow/drug settings."""
        if "Perfusion" not in self.state_devices:
            return
        label = self.state_devices["Perfusion"].get("label", "Off")
        speed_map = {"Off": 0.0, "Slow": 1.0, "Medium": 3.0, "Fast": 10.0, "Drug": 3.0}
        if label == "Drug" and getattr(self, "_drug_flow_speed", None) is not None:
            self.flow_speed = self._drug_flow_speed
        else:
            self.flow_speed = speed_map.get(label, self.flow_speed)
        # Auto-start/stop perfusion drug delivery
        if label == "Drug" and self._perfusion_enabled and not self._perfusion_active:
            self.start_perfusion()
        elif label != "Drug" and self._perfusion_active:
            self._perfusion_active = False

    def step(self, dt: float = 1.0):
        """Advance simulation by one timestep."""
        self.time += dt
        rng = self._rng

        # Read perfusion pump state (controls flow speed + drug)
        self._read_perfusion_state()

        # Temperature factor — scales Brownian diffusion & chemotaxis
        # (flow speed is pump-driven, not temperature-dependent)
        temp_factor = self._temp_speed_factor()

        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        cy_mid = (self._channel_y0 + self._channel_y1) / 2
        half_w = self.channel_width / 2

        for i in range(self.n_cells):
            if self._cell_trapped[i]:
                continue
            # Dead cells still flow but slower (no active movement)
            if not self._cell_alive[i]:
                # Dead cells drift with flow but no active behavior
                y_rel = (self._cell_y[i] - cy_mid) / half_w
                y_rel = np.clip(y_rel, -0.95, 0.95)
                v_flow = self.flow_speed * (1.0 - y_rel ** 2) * 0.5
                self._cell_x[i] += v_flow * dt
                if self._cell_x[i] > self.width + 20:
                    self._cell_x[i] = -10
                    self._cells_exited += 1
                    self._cells_entered += 1
                    r = self._cell_radius[i]
                    self._cell_y[i] = rng.uniform(
                        self._channel_y0 + r + 2,
                        self._channel_y1 - r - 2
                    )
                continue

            # Parabolic flow profile: v(y) = v_max * (1 - ((y-center)/half_w)^2)
            y_rel = (self._cell_y[i] - cy_mid) / half_w
            y_rel = np.clip(y_rel, -0.95, 0.95)
            v_flow = self.flow_speed * (1.0 - y_rel ** 2)

            # Larger cells experience more drag (slower)
            drag_factor = 1.0 / (1.0 + 0.05 * self._cell_radius[i])
            v_x = v_flow * drag_factor

            # Brownian diffusion (smaller cells diffuse more)
            # Temperature scales diffusion coefficient
            diff_scale = 0.5 / (self._cell_radius[i] ** 0.5) * temp_factor
            dx = v_x * dt + rng.normal(0, diff_scale)
            dy = rng.normal(0, diff_scale * 0.5)

            # Gradient chemotaxis: cells drift toward high concentration (bottom)
            if self.has_gradient:
                dy += self._gradient_strength * 0.3 * dt * temp_factor

            self._cell_x[i] += dx
            self._cell_y[i] += dy

            # Keep cells in channel
            r = self._cell_radius[i]
            self._cell_y[i] = np.clip(
                self._cell_y[i],
                self._channel_y0 + r + 1,
                self._channel_y1 - r - 1
            )

            # Check trap collisions
            for trap in self._traps:
                tx = trap["x"]
                tw = trap["width"]
                if abs(self._cell_x[i] - tx) < 15:
                    # Cell is near trap — check if it fits through
                    if self._cell_radius[i] * 2 > tw:
                        # Cell is too large — trapped!
                        self._cell_trapped[i] = True
                        self._cell_x[i] = tx - self._cell_radius[i] - 1

            # Wrap around (cells that exit right re-enter from left)
            if self._cell_x[i] > self.width + 20:
                self._cell_x[i] = -10
                self._cells_exited += 1
                self._cells_entered += 1
                # Re-randomize Y position
                self._cell_y[i] = rng.uniform(
                    self._channel_y0 + r + 2,
                    self._channel_y1 - r - 2
                )

        # Update reagent field
        self._update_reagent(dt)

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (no SLM effects)."""
        self.step(dt)

    # ── Rendering ──

    def _cell_elongation(self, i: int) -> float:
        """Compute flow-induced cell elongation (aspect ratio) for cell i.

        Cells near channel center (fast flow) elongate more.
        Trapped cells are round.  Returns aspect ratio >= 1.0.
        """
        if self._cell_trapped[i] or not self._cell_alive[i]:
            return 1.0
        cy_mid = (self._channel_y0 + self._channel_y1) / 2
        half_w = self.channel_width / 2
        y_rel = abs(self._cell_y[i] - cy_mid) / half_w
        # Elongation proportional to local shear rate (~flow speed)
        speed_frac = max(0, 1.0 - y_rel ** 2)
        # Larger cells deform more; small cells stay round
        size_frac = min(1.0, self._cell_radius[i] / 10.0)
        return 1.0 + 0.35 * speed_frac * size_frac  # up to 1.35 aspect ratio

    def _render_bf(self) -> np.ndarray:
        """Render brightfield with phase-contrast-like optics.

        PDMS walls show bright inner refraction edge.  Cells show dark body
        with shade-off gradient, bright phase halo, and visible nucleus.
        Traps rendered as pillar-like constrictions.
        """
        s = self.internal_scale
        ih, iw = self._ih, self._iw
        bg_pdms = 195   # PDMS substrate (slightly darker than channel)
        bg_channel = 235  # aqueous medium in channel (bright)

        img = np.full((ih, iw), bg_pdms, dtype=np.float32)

        iy0, iy1 = self._s(self._channel_y0), self._s(self._channel_y1)
        img[iy0:iy1, :] = bg_channel

        # --- PDMS wall rendering ---
        # Real PDMS walls: thick dark edge with bright refraction fringe on
        # channel side.  Wall thickness ~6-8 world px.
        wall_thick = max(2, 6 * s)
        fringe_thick = max(1, 3 * s)

        # Top wall: dark band then bright fringe below
        wt0 = max(0, iy0 - wall_thick)
        img[wt0:iy0, :] = 65   # dark PDMS edge
        img[iy0:iy0 + fringe_thick, :] = 250  # bright refraction fringe

        # Bottom wall: bright fringe above then dark band
        wb1 = min(ih, iy1 + wall_thick)
        img[iy1:wb1, :] = 65
        img[iy1 - fringe_thick:iy1, :] = 250

        # PDMS texture (fine granular pattern from casting)
        pdms_rng = np.random.default_rng(self._seed + 777)
        pdms_noise_w = pdms_rng.normal(0, 2.5, (self.height, self.width))
        if s > 1:
            pdms_noise = cv2.resize(pdms_noise_w.astype(np.float32), (iw, ih),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            pdms_noise = pdms_noise_w.astype(np.float32)
        mask_pdms = np.ones((ih, iw), dtype=bool)
        mask_pdms[iy0:iy1, :] = False
        img[mask_pdms] += pdms_noise[mask_pdms]

        # Channel interior: subtle aqueous noise (less than PDMS)
        ch_noise_w = self._noise_rng.normal(0, 1.5, (self.height, self.width))
        if s > 1:
            ch_noise = cv2.resize(ch_noise_w.astype(np.float32), (iw, ih),
                                  interpolation=cv2.INTER_LINEAR)
        else:
            ch_noise = ch_noise_w.astype(np.float32)
        mask_ch = ~mask_pdms
        img[mask_ch] += ch_noise[mask_ch]

        # --- Traps as pillar-like constrictions ---
        for trap in self._traps:
            tx = self._s(trap["x"])
            pt = self._s(trap["protrusion_top"])
            pb = self._s(trap["protrusion_bot"])
            pw = max(2, 4 * s)  # pillar half-width

            # Top pillar: dark PDMS body + bright fringe at tip
            p_top_end = iy0 + pt
            cv2.rectangle(img, (tx - pw, iy0), (tx + pw, p_top_end), 75, -1)
            # Bright fringe at pillar tip
            cv2.rectangle(img, (tx - pw, p_top_end),
                          (tx + pw, p_top_end + max(1, 2 * s)), 245, -1)
            # Pillar outline
            cv2.rectangle(img, (tx - pw, iy0), (tx + pw, p_top_end),
                          55, max(1, s))

            # Bottom pillar
            p_bot_start = iy1 - pb
            cv2.rectangle(img, (tx - pw, p_bot_start), (tx + pw, iy1), 75, -1)
            cv2.rectangle(img, (tx - pw, p_bot_start - max(1, 2 * s)),
                          (tx + pw, p_bot_start), 245, -1)
            cv2.rectangle(img, (tx - pw, p_bot_start), (tx + pw, iy1),
                          55, max(1, s))

        # --- Phase-contrast cells ---
        for alive_pass in (False, True):
            for i in range(self.n_cells):
                if self._cell_alive[i] != alive_pass:
                    continue
                cx = self._s(self._cell_x[i])
                cy = self._s(self._cell_y[i])
                r = max(2, self._s(self._cell_radius[i]))

                if not (0 <= cx < iw and iy0 < cy < iy1):
                    continue

                # Flow-induced elongation
                aspect = self._cell_elongation(i)

                if not self._cell_alive[i]:
                    # Dead cell: faint, swollen, no halo, blebbed outline
                    dead_r = int(r * 1.15)
                    if aspect > 1.05:
                        axes = (int(dead_r * aspect), dead_r)
                        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360,
                                    205.0, -1, cv2.LINE_AA)
                        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360,
                                    195.0, max(1, s), cv2.LINE_AA)
                    else:
                        cv2.circle(img, (cx, cy), dead_r, 205.0, -1, cv2.LINE_AA)
                        cv2.circle(img, (cx, cy), dead_r, 195.0,
                                   max(1, s), cv2.LINE_AA)
                    continue

                # --- Alive cell: phase-contrast rendering ---
                # Halo width scales with cell size (larger cells → wider halos)
                halo_w = max(1, int(1.5 * s + r * 0.15))
                body_val = 55 if self._cell_trapped[i] else 65

                # Smooth shade-off: 5-band radial gradient from dark edge
                # to lighter center (real PC shade-off is continuous)
                shade_bands = [
                    (1.00, body_val),
                    (0.80, body_val + 8),
                    (0.60, body_val + 18),
                    (0.40, body_val + 26),
                    (0.20, body_val + 32),
                ]

                if aspect > 1.05:
                    ax_long = int(r * aspect)
                    ax_short = r

                    # Bright halo (wider perpendicular to long axis)
                    halo_axes = (ax_long + halo_w, ax_short + halo_w + max(1, s))
                    cv2.ellipse(img, (cx, cy), halo_axes, 0, 0, 360,
                                250.0, halo_w, cv2.LINE_AA)

                    # Smooth shade-off bands (outer → inner)
                    for frac, val in shade_bands:
                        ba = (max(1, int(ax_long * frac)),
                              max(1, int(ax_short * frac)))
                        cv2.ellipse(img, (cx, cy), ba, 0, 0, 360,
                                    float(val), -1, cv2.LINE_AA)

                    # Nucleus (dark spot offset from center)
                    nuc_r = max(1, int(r * 0.40))
                    nox = int(self._nuc_offset_x[i] * r * aspect)
                    noy = int(self._nuc_offset_y[i] * r)
                    cv2.ellipse(img, (cx + nox, cy + noy),
                                (max(1, int(nuc_r * aspect * 0.8)), nuc_r),
                                0, 0, 360, float(body_val - 15), -1)
                else:
                    # Round cell — circles
                    # Bright halo ring
                    cv2.circle(img, (cx, cy), r + halo_w, 250.0,
                               halo_w, cv2.LINE_AA)

                    # Smooth shade-off bands (outer → inner)
                    for frac, val in shade_bands:
                        band_r = max(1, int(r * frac))
                        cv2.circle(img, (cx, cy), band_r,
                                   float(val), -1, cv2.LINE_AA)

                    # Nucleus
                    nuc_r = max(1, int(r * 0.40))
                    nox = int(self._nuc_offset_x[i] * r)
                    noy = int(self._nuc_offset_y[i] * r)
                    cv2.circle(img, (cx + nox, cy + noy), nuc_r,
                               float(body_val - 15), -1)

                    # Trapped cell: extra dark outline from compression
                    if self._cell_trapped[i]:
                        cv2.circle(img, (cx, cy), r, 50.0,
                                   max(1, s), cv2.LINE_AA)

        # BF exposure scaling (transmitted light, 2× fluorescence base)
        # Applied in snap_frame via pipeline

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_nuc(self) -> np.ndarray:
        """Render nuclear staining (DAPI-like) with nucleolus and chromatin texture."""
        s = self.internal_scale
        ih, iw = self._ih, self._iw
        img = np.zeros((ih, iw), dtype=np.float32)

        iy0, iy1 = self._s(self._channel_y0), self._s(self._channel_y1)

        # Autofluorescence inside channel (faint, from PDMS + media)
        auto_w = self._noise_rng.normal(4, 1.5, (self.height, self.width)).astype(np.float32)
        if s > 1:
            auto = cv2.resize(auto_w, (iw, ih), interpolation=cv2.INTER_LINEAR)
        else:
            auto = auto_w
        channel_mask = np.zeros((ih, iw), dtype=bool)
        channel_mask[iy0:iy1, :] = True
        img[channel_mask] = auto[channel_mask]

        for i in range(self.n_cells):
            cx = self._s(self._cell_x[i])
            cy = self._s(self._cell_y[i])

            if not (0 <= cx < iw and iy0 < cy < iy1):
                continue

            aspect = self._cell_elongation(i)

            if not self._cell_alive[i]:
                # Dead cell: bright condensed nucleus (pyknotic) + diffuse halo
                r = max(2, self._s(self._cell_radius[i] * 0.4))
                halo_r = r + self._s(2)
                cv2.circle(img, (cx, cy), halo_r, 50.0, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), r, 240.0, -1, cv2.LINE_AA)
            else:
                nuc_r = max(2, self._s(self._cell_radius[i] * 0.65))
                nuc_val = min(245, self._cell_intensity[i] * 1.15)

                # Nuclear envelope: bright rim
                rim_val = min(255, nuc_val + 15)

                if aspect > 1.05:
                    ax_long = max(2, int(nuc_r * aspect * 0.9))
                    ax_short = nuc_r
                    # Rim
                    cv2.ellipse(img, (cx, cy), (ax_long, ax_short), 0, 0, 360,
                                float(rim_val), max(1, s), cv2.LINE_AA)
                    # Fill
                    cv2.ellipse(img, (cx, cy),
                                (max(1, ax_long - max(1, s)),
                                 max(1, ax_short - max(1, s))),
                                0, 0, 360, float(nuc_val), -1, cv2.LINE_AA)
                    # Nucleolus (dark void)
                    nlr = max(1, int(nuc_r * self._nucleolus_r_frac[i]))
                    nox = int(self._nuc_offset_x[i] * nuc_r)
                    noy = int(self._nuc_offset_y[i] * nuc_r)
                    cv2.circle(img, (cx + nox, cy + noy), nlr,
                               float(nuc_val * 0.35), -1)
                else:
                    # Rim
                    cv2.circle(img, (cx, cy), nuc_r, float(rim_val),
                               max(1, s), cv2.LINE_AA)
                    # Fill
                    cv2.circle(img, (cx, cy), max(1, nuc_r - max(1, s)),
                               float(nuc_val), -1, cv2.LINE_AA)
                    # Nucleolus (dark void — rRNA-rich region)
                    nlr = max(1, int(nuc_r * self._nucleolus_r_frac[i]))
                    nox = int(self._nuc_offset_x[i] * nuc_r)
                    noy = int(self._nuc_offset_y[i] * nuc_r)
                    cv2.circle(img, (cx + nox, cy + noy), nlr,
                               float(nuc_val * 0.35), -1)
                    # Chromatin foci (2-3 bright spots)
                    crng = np.random.default_rng(self._cell_texture_seed[i])
                    n_foci = crng.integers(2, 4)
                    for _ in range(n_foci):
                        fa = crng.uniform(0, 2 * np.pi)
                        fr = crng.uniform(0.2, 0.7) * nuc_r
                        fx = cx + int(fr * np.cos(fa))
                        fy = cy + int(fr * np.sin(fa))
                        cv2.circle(img, (fx, fy), max(1, s),
                                   float(min(255, nuc_val * 1.2)), -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_gradient(self) -> np.ndarray:
        """Render chemical gradient/reagent concentration (membrane channel)."""
        ih, iw = self._ih, self._iw
        s = self.internal_scale
        img = np.zeros((ih, iw), dtype=np.float32)

        iy0, iy1 = self._s(self._channel_y0), self._s(self._channel_y1)
        ch_width_int = iy1 - iy0

        if self._perfusion_enabled and self._reagent_field is not None:
            # Show reagent concentration field
            # Expand 1D reagent profile to 2D (uniform across channel width)
            rf_row = self._reagent_field.copy()
            # Upscale to internal resolution
            rf_int = cv2.resize(
                rf_row.reshape(1, -1), (iw, 1),
                interpolation=cv2.INTER_LINEAR
            ).flatten()
            # Fill channel with reagent intensity
            for y in range(iy0, iy1):
                img[y, :] = rf_int * 200  # scale to 0-200 intensity range
        elif self.has_gradient:
            for y in range(iy0, iy1):
                frac = (y - iy0) / max(1, ch_width_int - 1)
                val = frac * 180 * self._gradient_strength
                img[y, :] = val
        else:
            img[iy0:iy1, :] = 15

        noise_w = self._noise_rng.normal(0, 3, (self.height, self.width)).astype(np.float32)
        if s > 1:
            noise = cv2.resize(noise_w, (iw, ih), interpolation=cv2.INTER_LINEAR)
        else:
            noise = noise_w
        img += noise

        return np.clip(img, 0, 255).astype(np.uint8)

    def _apply_slm_release(self, mask: np.ndarray):
        """Release trapped cells that are illuminated by SLM mask.

        Released cells are pushed 20px downstream past the trap to
        prevent immediate re-trapping in the next step.
        """
        if mask is None:
            return
        h, w = mask.shape[:2]
        for i in range(self.n_cells):
            if not self._cell_trapped[i]:
                continue
            # Map cell world coords to mask coords
            mx = int(self._cell_x[i] / self.width * w)
            my = int(self._cell_y[i] / self.height * h)
            mx = max(0, min(mx, w - 1))
            my = max(0, min(my, h - 1))
            val = int(mask[my, mx]) if mask.ndim == 2 else int(mask[my, mx, 0])
            if val >= self._slm_release_threshold:
                self._cell_trapped[i] = False
                # Push cell downstream past the trap
                self._cell_x[i] += 20 + self._cell_radius[i]
                self._release_events.append({
                    "cell_idx": int(i),
                    "cell_type": self._cell_type[i],
                    "time": float(self.time),
                    "x": float(self._cell_x[i]),
                    "y": float(self._cell_y[i]),
                })

    def snap_frame(self, mask=None, exposure=None, intensity=None,
                   **kwargs) -> np.ndarray:
        """Capture one frame from the current mode/objective."""
        self._update_mode()
        self._update_objectif()

        # SLM-controlled cell release (before stepping)
        if mask is not None:
            self._apply_slm_release(mask)

        self._auto_step_tick()
        self._snap_count += 1

        if self.mode == 1:
            full = self._render_nuc()
            crop = self._crop_fov(full)
            crop = self._apply_defocus(crop)
            pipe = self._pipeline_nuc
            if pipe.photobleach_rate > 0:
                return pipe.apply_with_bleach(crop, exposure_ms=exposure or 50)
            return pipe.apply(crop)
        elif self.mode == 2:
            full = self._render_gradient()
            crop = self._crop_fov(full)
            crop = self._apply_defocus(crop)
            pipe = self._pipeline_mem
            if pipe.photobleach_rate > 0:
                return pipe.apply_with_bleach(crop, exposure_ms=exposure or 50)
            return pipe.apply(crop)
        else:
            full = self._render_bf()
            crop = self._crop_fov(full)
            crop = self._apply_defocus(crop)
            return self._pipeline_bf.apply(crop)

    def get_ground_truth(self) -> dict:
        """Return ground truth for grading."""
        # Count cells by type
        type_counts = {"small": 0, "medium": 0, "large": 0}
        for ct in self._cell_type:
            type_counts[ct] += 1

        n_trapped = int(self._cell_trapped.sum())
        n_alive = int(self._cell_alive.sum())
        n_dead = self.n_cells - n_alive
        n_free = self.n_cells - n_trapped

        # Mean flow speed (measured from free, alive cells)
        cy_mid = (self._channel_y0 + self._channel_y1) / 2
        half_w = self.channel_width / 2
        speeds = []
        for i in range(self.n_cells):
            if not self._cell_trapped[i] and self._cell_alive[i]:
                y_rel = (self._cell_y[i] - cy_mid) / half_w
                y_rel = np.clip(y_rel, -0.95, 0.95)
                v = self.flow_speed * (1.0 - y_rel ** 2)
                drag = 1.0 / (1.0 + 0.05 * self._cell_radius[i])
                speeds.append(v * drag)

        # Cell positions
        positions = []
        for i in range(self.n_cells):
            pos = {
                "x": float(self._cell_x[i]),
                "y": float(self._cell_y[i]),
                "radius": float(self._cell_radius[i]),
                "type": self._cell_type[i],
                "trapped": bool(self._cell_trapped[i]),
                "alive": bool(self._cell_alive[i]),
            }
            if self._cell_drug_exposure is not None:
                pos["drug_exposure"] = round(float(self._cell_drug_exposure[i]), 3)
            positions.append(pos)

        gt = {
            "n_cells": self.n_cells,
            "type_counts": type_counts,
            "n_trapped": n_trapped,
            "n_free": n_free,
            "n_alive": n_alive,
            "n_dead": n_dead,
            "mean_flow_speed": round(float(np.mean(speeds)) if speeds else 0, 2),
            "channel_width": self.channel_width,
            "n_traps": self.n_traps,
            "has_gradient": self.has_gradient,
            "cells_exited": self._cells_exited,
            "cell_positions": positions,
            "n_released": len(self._release_events),
            "release_events": self._release_events,
        }

        # Perfusion-specific ground truth
        if self._perfusion_enabled:
            # Reagent wavefront position (first x where conc > 0.5)
            if self._reagent_field is not None:
                above = np.where(self._reagent_field > 0.5)[0]
                wavefront_x = float(above[-1]) if len(above) > 0 else 0.0
            else:
                wavefront_x = 0.0

            gt["perfusion"] = {
                "drug_name": self._drug_name,
                "lethal_conc": self._lethal_conc,
                "perfusion_active": self._perfusion_active,
                "perfusion_start_time": self._perfusion_start_time,
                "wavefront_x": round(wavefront_x, 1),
                "n_alive": n_alive,
                "n_dead": n_dead,
                "viability": round(n_alive / max(1, self.n_cells), 3),
            }

        return gt

    def enable_photobleaching(self, rate: float = 0.001):
        """Enable photobleaching on fluorescence channels.

        Args:
            rate: Fractional signal loss per exposure-ms.
        """
        self._pipeline_nuc.photobleach_rate = rate
        self._pipeline_mem.photobleach_rate = rate

    def reset_photobleaching(self):
        """Reset accumulated photobleaching."""
        self._pipeline_nuc.reset_bleach()
        self._pipeline_mem.reset_bleach()
