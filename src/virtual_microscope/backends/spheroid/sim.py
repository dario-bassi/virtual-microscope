"""
SpheroidSim — 3D tumor spheroid simulation backend.

Simulates a multicellular spheroid (tumor organoid) with concentric layers:
  - Proliferating rim (outer): high viability, bright calcein, GFP+
  - Quiescent mantle (middle): moderate viability, dimmer calcein, some GFP
  - Necrotic core (inner): dead cells, PI+ (propidium iodide), no GFP

The spheroid sits in a well/dish. The agent must navigate Z-planes to
understand the 3D structure, measure diameter, quantify viability gradients,
and determine the necrotic core size.

Channels:
  - mode 0 (brightfield): Phase contrast of spheroid cross-section
  - mode 1 (nucleus): Calcein-AM (live cells, green fluorescence)
  - mode 2 (membrane): Propidium iodide (dead cells, red fluorescence)

Z-stack capability:
  - Different Z-planes show different cross-sections of the sphere
  - Equatorial plane shows maximum diameter
  - Top/bottom planes show small cross-sections
  - Necrotic core only visible in equatorial slices

Usage:
    sim = SpheroidSim(radius=80, n_cells=2000, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class SpheroidSim(SimBase):
    """3D spheroid simulation with Z-stack support."""

    continuous = True

    def __init__(self, radius=80, n_cells=2000, necrotic_fraction=0.45,
                 quiescent_fraction=0.20, world_size=512, seed=42,
                 viewport_width=512, viewport_height=512,
                 internal_scale: int = 1):
        """
        Parameters:
            radius: spheroid radius in pixels (at 10x)
            n_cells: total number of cells in the spheroid
            necrotic_fraction: fraction of radius that is necrotic core
            quiescent_fraction: fraction of radius for quiescent mantle
            world_size: full image size
            seed: random seed
            internal_scale: render at NxN world resolution (4 = high-res)
        """
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=1.0,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
                ("mRFP1-Q667(549/570)", "ORANGE"): 2,
            },
        )

        self.radius = radius
        self.n_cells = n_cells
        self.necrotic_fraction = necrotic_fraction
        self.quiescent_fraction = quiescent_fraction
        self._seed = seed

        # Override DOF table entry for 20x
        self._dof_table[20] = 3.0

        # Radii for layers
        self._necrotic_r = radius * necrotic_fraction
        self._quiescent_r = radius * (necrotic_fraction + quiescent_fraction)
        # Proliferating: quiescent_r to radius

        # ── Optical pipelines per channel ──
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 500),
            1: OpticalPipeline(
                psf_sigma=0.7, noise={"photon_scale": 4.0, "read_std": 2.5},
                vignette=0.12, rng_seed=seed + 501),
            2: OpticalPipeline(
                psf_sigma=0.7, noise={"photon_scale": 4.0, "read_std": 2.5},
                vignette=0.12, rng_seed=seed + 502),
        }

        # ── Growth dynamics (off by default) ──
        self._growth_enabled = False
        self._growth_rate = 0.0         # radius px/step
        self._rim_thickness = 0.0       # proliferating rim thickness (constant)
        self._initial_radius = radius

        # ── Drug response ──
        self._drug_active = False
        self._drug_name = None
        self._drug_effect = 0.0
        self._drug_washing_out = False
        self._drug_killed = 0  # cumulative drug-killed count

        self._drug_profiles = {
            "cisplatin": {
                "prolif_kill_rate": 0.08,   # kills 8% of proliferating cells/step at full effect
                "quiesc_kill_rate": 0.01,   # minimal quiescent killing
                "growth_inhibit": 0.95,     # 95% growth suppression
                "onset_rate": 0.12,
                "washout_rate": 0.06,
                "penetration_depth": 80.0,  # µm — limited penetration
            },
            "doxorubicin": {
                "prolif_kill_rate": 0.10,
                "quiesc_kill_rate": 0.04,   # also kills some quiescent cells
                "growth_inhibit": 0.90,
                "onset_rate": 0.15,
                "washout_rate": 0.08,
                "penetration_depth": 50.0,  # µm — poor penetration
            },
            "staurosporine": {
                "prolif_kill_rate": 0.15,   # broad apoptosis inducer
                "quiesc_kill_rate": 0.10,
                "growth_inhibit": 1.0,      # complete growth arrest
                "onset_rate": 0.20,
                "washout_rate": 0.04,
                "penetration_depth": 120.0,  # µm — small molecule, good penetration
            },
        }

        self._rng = np.random.default_rng(seed)
        self._generate_cells()

    def _generate_cells(self):
        """Generate cell positions within the spheroid (3D)."""
        rng = self._rng

        # Center of spheroid in world coordinates
        self._cx = self.width / 2
        self._cy = self.height / 2
        self._cz = 0.0  # Z center

        # Generate cells uniformly distributed in sphere (rejection sampling)
        cells = []
        while len(cells) < self.n_cells:
            batch = rng.uniform(-self.radius, self.radius,
                                size=(self.n_cells * 3, 3))
            dists = np.linalg.norm(batch, axis=1)
            valid = batch[dists < self.radius]
            cells.extend(valid[:self.n_cells - len(cells)])

        cells = np.array(cells[:self.n_cells])
        self._cell_x = cells[:, 0] + self._cx  # world x
        self._cell_y = cells[:, 1] + self._cy  # world y
        self._cell_z = cells[:, 2]  # Z relative to spheroid center

        # Cell radii (distance from spheroid center)
        self._cell_dist = np.sqrt(
            (self._cell_x - self._cx)**2 +
            (self._cell_y - self._cy)**2 +
            self._cell_z**2
        )

        # Classify cells
        self._is_necrotic = self._cell_dist < self._necrotic_r
        self._is_quiescent = (
            (self._cell_dist >= self._necrotic_r) &
            (self._cell_dist < self._quiescent_r)
        )
        self._is_proliferating = self._cell_dist >= self._quiescent_r

        # Cell properties
        self._cell_radius = rng.uniform(3, 6, self.n_cells)  # pixel radius at 10x

        # Calcein intensity (live cells): bright for proliferating, dim for quiescent
        self._calcein = np.zeros(self.n_cells, dtype=np.float32)
        self._calcein[self._is_proliferating] = rng.uniform(
            160, 220, self._is_proliferating.sum())
        self._calcein[self._is_quiescent] = rng.uniform(
            60, 120, self._is_quiescent.sum())
        # Dead cells: no calcein
        self._calcein[self._is_necrotic] = rng.uniform(0, 10, self._is_necrotic.sum())

        # PI intensity (dead cells): bright in necrotic core
        self._pi = np.zeros(self.n_cells, dtype=np.float32)
        self._pi[self._is_necrotic] = rng.uniform(140, 220, self._is_necrotic.sum())
        # Some quiescent cells are dying
        dying_quiescent = self._is_quiescent & (rng.random(self.n_cells) < 0.15)
        self._pi[dying_quiescent] = rng.uniform(60, 120, dying_quiescent.sum())

        # GFP expression (hypoxia gradient — brighter near surface)
        self._gfp = np.zeros(self.n_cells, dtype=np.float32)
        norm_dist = self._cell_dist / self.radius
        gfp_base = 200 * np.clip(norm_dist, 0, 1)  # brighter at rim
        self._gfp = (gfp_base + rng.normal(0, 15, self.n_cells)).clip(0, 255)
        self._gfp[self._is_necrotic] = rng.uniform(0, 15, self._is_necrotic.sum())

    # ── Depth & OOF helpers ──

    def _depth_attenuation(self):
        """Per-cell depth attenuation (coverslip at z=-R, far side dim).

        Confocal signal decays exponentially with depth into tissue.
        Spheroid interior has additional scattering from overlying cells.
        """
        z = self._cell_z
        R = self.radius
        # Coverslip at bottom: z=-R is closest, z=+R is furthest
        depth_frac = (z + R) / (2 * R)  # 0 at bottom, 1 at top
        atten = 1.0 - 0.55 * depth_frac
        return atten.clip(0.2, 1.0).astype(np.float32)

    def _oof_haze(self, channel="calcein"):
        """Out-of-focus haze from the 3D sphere.

        Creates the characteristic "bowl" pattern in confocal fluorescence:
        bright signal at the outer edge of the equatorial slice, dimmer
        toward center where out-of-focus planes contribute diffuse glow.
        """
        ih, iw = self._ih, self._iw
        haze = np.zeros((ih, iw), dtype=np.float32)

        z = self.focal_plane - self.tissue_z
        R = self.radius
        cx_int = self._sf(self._cx)
        cy_int = self._sf(self._cy)
        yy, xx = np.ogrid[:ih, :iw]
        dist = np.sqrt((xx - cx_int)**2 + (yy - cy_int)**2)

        n_slices = 8
        for dz_frac in np.linspace(-1.0, 1.0, n_slices):
            z_oof = z + dz_frac * R * 0.8
            if abs(dz_frac) < 0.15:
                continue
            r_sq = R * R - z_oof * z_oof
            if r_sq <= 0:
                continue
            r_oof = np.sqrt(r_sq)
            r_int = self._sf(r_oof)

            inside = dist <= r_int
            defocus_dist = abs(dz_frac) * R * 0.8
            atten = 1.0 / (1.0 + (defocus_dist / 8.0)**2)

            # Real widefield: massive OOF haze from thick 3D tissue.
            # Every focal plane captures fluorescence from the entire volume,
            # creating a "fuzzy glowing ball" effect.
            base = 12.0 if channel == "calcein" else 10.0
            haze += np.where(inside, base * atten, 0.0)

        blur_sigma = max(3.0, self._sf(6.0))
        haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=blur_sigma)
        return haze

    # ── Rendering ──

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs):
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()

        self._auto_step_tick()
        self._snap_count += 1

        if self.mode == 0:
            full = self._render_brightfield()
        elif self.mode == 1:
            full = self._render_calcein()
        elif self.mode == 2:
            full = self._render_pi()
        elif self.mode in self._extra_channels:
            full = self._extra_channels[self.mode].get("image",
                                                        self._render_brightfield())
        else:
            full = self._render_brightfield()

        # Crop FOV and resize to viewport
        viewport = self._crop_fov(full)

        # Apply optical pipeline (PSF, noise, vignetting)
        viewport = self._apply_pipeline(viewport, exposure)

        return viewport

    def _visible_cells(self):
        """Get indices of cells visible at the current focal plane."""
        # Z distance from focal plane
        dz = np.abs(self._cell_z - (self.focal_plane - self.tissue_z))
        half_dof = self._dof / 2.0

        # In-focus cells
        in_focus = dz <= half_dof

        # Near-focus cells (visible but blurred)
        near_focus = (dz > half_dof) & (dz < half_dof * 4)

        # Compute opacity and blur for each cell
        opacity = np.zeros(self.n_cells, dtype=np.float32)
        blur = np.zeros(self.n_cells, dtype=np.float32)

        opacity[in_focus] = 1.0
        blur[in_focus] = 0.0

        if near_focus.any():
            defocus = dz[near_focus] - half_dof
            opacity[near_focus] = np.clip(1.0 - defocus / (half_dof * 3), 0.05, 0.8)
            blur[near_focus] = np.clip(defocus * 0.5, 0, 8)

        return opacity, blur

    def _render_brightfield(self):
        """Render brightfield: phase contrast of spheroid cross-section.

        Real spheroids are extremely dark in phase contrast because
        the accumulated OPD (~19 wavelengths for a 300µm spheroid) vastly
        exceeds the quarter-wave design assumption.  The interior is an
        opaque mass; individual cells are only resolvable in the outermost
        2-3 layers at ≥40x.
        """
        s = self.internal_scale
        ih, iw = self._ih, self._iw
        bg_val = 175.0
        img = np.full((ih, iw), bg_val, dtype=np.float32)

        z_from_center = self.focal_plane - self.tissue_z
        slice_r_sq = self.radius**2 - z_from_center**2
        if slice_r_sq > 0:
            slice_r = np.sqrt(slice_r_sq)
            slice_r_int = self._sf(slice_r)
            cx_int = self._sf(self._cx)
            cy_int = self._sf(self._cy)

            yy, xx = np.ogrid[:ih, :iw]
            dist_from_center = np.sqrt((xx - cx_int)**2 + (yy - cy_int)**2)
            inside = dist_from_center < slice_r_int

            # ── Optical thickness profile ──
            # Spheroid cross-section is a circle; thickness = 2*sqrt(R²-d²)
            norm_dist = dist_from_center / max(slice_r_int, 1)
            thickness = np.where(inside,
                                 np.sqrt(np.clip(1.0 - norm_dist**2, 0, 1)),
                                 0)

            # Shade-off: thick centre trends back toward background
            # but real spheroids are SO thick that it barely recovers
            shade_off = 1.0 - 0.15 * thickness

            # Interior: dark.  Viable tissue ~60-80, centre ~40-55.
            interior_val = bg_val - thickness * 130.0 * shade_off
            img = np.where(inside, interior_val, img)

            # ── Necrotic core: darker + granular debris ──
            nec_slice_r_sq = self._necrotic_r**2 - z_from_center**2
            if nec_slice_r_sq > 0:
                nec_r_int = self._sf(np.sqrt(nec_slice_r_sq))
                nec_inside = dist_from_center < nec_r_int
                # Necrotic core is markedly darker than viable tissue —
                # real necrotic cores appear ~30-50 gray, not just slightly dimmer.
                # Gradient: darkest at center, less dark at transition zone.
                nec_norm = dist_from_center / max(nec_r_int, 1)
                nec_gradient = np.where(nec_inside,
                                        1.0 - 0.4 * nec_norm, 0)
                img = np.where(nec_inside, img - 30 * nec_gradient, img)

                # Coarse granular texture (cellular debris)
                if not hasattr(self, '_nec_texture'):
                    rng_tex = np.random.default_rng(self._seed + 7777)
                    tex_w = rng_tex.normal(0, 8,
                                           (self.height, self.width)).astype(np.float32)
                    if s > 1:
                        self._nec_texture = cv2.resize(
                            tex_w, (iw, ih), interpolation=cv2.INTER_LINEAR)
                    else:
                        self._nec_texture = tex_w
                img = np.where(nec_inside, img + self._nec_texture, img)

            # ── Dark rim at spheroid-medium boundary ──
            # Steepest OPD gradient → darkest ring, just inside the edge
            dark_rim_w = max(1.0, 2.0 * s)
            dark_rim = (inside &
                        (dist_from_center >= slice_r_int - dark_rim_w))
            img = np.where(dark_rim, np.minimum(img, 35.0), img)

            # ── Phase contrast halo: gradient bright fringe ──
            # Build a narrow band outside the spheroid edge and let it
            # fade with distance (distance transform style).
            outside = ~inside
            # Distance from the spheroid boundary for exterior pixels
            dist_outside = dist_from_center - slice_r_int
            halo_width = 4.0 * s
            halo_mask = outside & (dist_outside < halo_width)
            halo_falloff = np.where(
                halo_mask,
                np.clip(1.0 - dist_outside / halo_width, 0, 1),
                0)
            halo_val = bg_val + halo_falloff ** 1.3 * 60.0
            img = np.where(halo_mask, halo_val, img)

            # ── Subtle tissue texture inside viable region ──
            if not hasattr(self, '_tissue_texture'):
                rng_t = np.random.default_rng(self._seed + 8888)
                tex = rng_t.normal(0, 2.0,
                                   (self.height, self.width)).astype(np.float32)
                if s > 1:
                    self._tissue_texture = cv2.resize(
                        tex, (iw, ih), interpolation=cv2.INTER_LINEAR)
                else:
                    self._tissue_texture = tex
            img = np.where(inside & ~dark_rim, img + self._tissue_texture, img)

        # ── Per-cell phase features (peripheral cells only) ──
        # In a real spheroid only the outermost 2-3 cell layers are
        # individually resolvable.  Interior is an opaque mass.
        opacity, blur = self._visible_cells()
        for i in range(self.n_cells):
            if opacity[i] < 0.05:
                continue
            # Only draw cells near the spheroid surface
            depth_into_surface = self.radius - self._cell_dist[i]
            if depth_into_surface > 12:  # more than ~2 cell diameters deep
                continue
            cx_i = self._s(self._cell_x[i])
            cy_i = self._s(self._cell_y[i])
            r = max(1, self._s(self._cell_radius[i]))
            # Deeper cells are less visible
            surf_alpha = opacity[i] * np.clip(1.0 - depth_into_surface / 12.0,
                                              0.15, 1.0)

            # Dark cell body with bright halo (phase contrast)
            body_val = 50 if self._is_necrotic[i] else 65
            halo_r = r + max(1, self._s(1.5))

            y0 = max(0, cy_i - halo_r - 1)
            y1 = min(ih, cy_i + halo_r + 2)
            x0 = max(0, cx_i - halo_r - 1)
            x1 = min(iw, cx_i + halo_r + 2)
            if y1 <= y0 or x1 <= x0:
                continue

            region = img[y0:y1, x0:x1]
            lx, ly = cx_i - x0, cy_i - y0

            # Bright halo ring
            halo_mask = np.zeros_like(region)
            cv2.circle(halo_mask, (lx, ly), halo_r, 1.0, -1)
            body_mask = np.zeros_like(region)
            cv2.circle(body_mask, (lx, ly), r, 1.0, -1)
            ring = halo_mask - body_mask
            img[y0:y1, x0:x1] = np.where(
                ring > 0.5,
                region * (1 - ring * surf_alpha * 0.5) + 200 * ring * surf_alpha * 0.5,
                region)

            # Dark cell body
            region = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = np.where(
                body_mask > 0.5,
                region * (1 - body_mask * surf_alpha * 0.4) + body_val * body_mask * surf_alpha * 0.4,
                region)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_calcein(self):
        """Render calcein-AM fluorescence (live cells: green).

        Bowl pattern: bright at periphery, dim toward center.
        Depth attenuation: coverslip side bright, far side dim.
        """
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        # OOF haze from entire 3D sphere
        img += self._oof_haze("calcein")

        opacity, blur = self._visible_cells()
        depth_atten = self._depth_attenuation()

        for i in range(self.n_cells):
            if opacity[i] < 0.05 or self._calcein[i] < 5:
                continue
            cx = self._s(self._cell_x[i])
            cy = self._s(self._cell_y[i])
            r = max(1, self._s(self._cell_radius[i]))
            val = self._calcein[i] * opacity[i] * depth_atten[i]

            if blur[i] > 0.5:
                blur_r = max(r + 1, int(r + self._s(blur[i])))
                cv2.circle(img, (cx, cy), blur_r, float(val * 0.3), -1)
            else:
                cv2.circle(img, (cx, cy), r, float(val), -1)

        # Autofluorescence background
        img += 2.0

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_pi(self):
        """Render propidium iodide fluorescence (dead cells: red).

        PI stains nuclear DNA of dead cells. Bowl pattern + depth attenuation.
        The necrotic core is deep inside, so PI signal is heavily attenuated
        except on the near (coverslip) side.
        """
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        # OOF haze from dead cells throughout the sphere
        img += self._oof_haze("pi")

        opacity, blur = self._visible_cells()
        depth_atten = self._depth_attenuation()

        for i in range(self.n_cells):
            if opacity[i] < 0.05 or self._pi[i] < 5:
                continue
            cx = self._s(self._cell_x[i])
            cy = self._s(self._cell_y[i])
            # PI stains nucleus — slightly smaller than cell body
            r = max(1, self._s(self._cell_radius[i] * 0.7))
            val = self._pi[i] * opacity[i] * depth_atten[i]

            if blur[i] > 0.5:
                blur_r = max(r + 1, int(r + self._s(blur[i])))
                cv2.circle(img, (cx, cy), blur_r, float(val * 0.3), -1)
            else:
                cv2.circle(img, (cx, cy), r, float(val), -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def enable_growth(self, growth_rate=1.5, rim_thickness=None):
        """Enable spheroid growth dynamics.

        Args:
            growth_rate: radius increase per step (px). Default 1.5 px/step.
            rim_thickness: proliferating rim thickness (px). If None,
                uses current proliferating layer width.
        """
        self._growth_enabled = True
        self._growth_rate = growth_rate
        self._rim_thickness = rim_thickness or (
            self.radius - self._quiescent_r
        )

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 37.0  # mammalian default
        return float(self.state_devices["Temperature"].get("label", "37"))

    def _temp_growth_factor(self) -> float:
        """Temperature-dependent growth factor for mammalian spheroid.

        Q10 ~ 2.0. Optimal at 37°C. Cold arrest below 10°C.
        """
        temp = self._get_temperature()
        if temp < 10:
            return 0.05
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.05, 1.0 - (temp - 42) * 0.3)
        return factor

    def step(self, dt: float = 1.0):
        """Advance spheroid growth by one timestep.

        Growth model:
        - Radius increases linearly (proliferating rim adds mass)
        - Necrotic core expands to maintain constant rim+mantle thickness
          (oxygen diffusion limit keeps outer viable layer ~constant)
        - New cells added at the expanding rim
        - Inner cells reclassified as layers shift outward
        - Drug kills proliferating/quiescent cells if active
        """
        # Z-drift accumulation (mechanical — not temperature-dependent)
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self._rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        # Temperature scaling for growth
        temp_factor = self._temp_growth_factor()

        # Update drug effect
        self._update_drug_effect(dt)

        if not self._growth_enabled:
            if self._drug_active or self._drug_washing_out:
                self._apply_drug_killing(dt)
            self._time += dt
            return

        old_radius = self.radius

        # Grow radius (inhibited by drug, scaled by temperature)
        effective_rate = self._growth_rate * temp_factor
        if self._drug_effect > 0:
            profile = self._drug_profiles.get(self._drug_name, {})
            inhibit = profile.get("growth_inhibit", 0.0)
            effective_rate *= (1.0 - inhibit * self._drug_effect)
        self.radius += max(0, effective_rate) * dt

        # Maintain constant viable layer thickness:
        # viable_thickness = rim + quiescent mantle ≈ constant
        viable_thickness = old_radius - self._necrotic_r
        self._necrotic_r = max(0, self.radius - viable_thickness)
        self._quiescent_r = max(
            self._necrotic_r,
            self.radius - self._rim_thickness,
        )

        # ── Add new cells at the proliferating rim ──
        # Number proportional to volume increase
        old_vol = (4/3) * np.pi * old_radius**3
        new_vol = (4/3) * np.pi * self.radius**3
        vol_increase = new_vol - old_vol
        rim_vol = new_vol - (4/3) * np.pi * self._quiescent_r**3
        # New cells: proportion of volume increase in rim
        n_new = max(0, int(vol_increase / rim_vol * self._is_proliferating.sum() * 0.3))
        n_new = min(n_new, 100)  # cap per step

        if n_new > 0:
            self._add_rim_cells(n_new)

        # ── Reclassify existing cells ──
        self._reclassify_cells()

        # ── Drug killing ──
        if self._drug_effect > 0:
            self._apply_drug_killing(dt)

        self._time += dt

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (no SLM effects)."""
        self.step(dt)

    def _add_rim_cells(self, n_new):
        """Add new cells in the proliferating rim (outer shell)."""
        rng = self._rng

        # Sample points uniformly in a spherical shell
        new_cells = []
        inner_r = self._quiescent_r
        outer_r = self.radius
        while len(new_cells) < n_new:
            batch = rng.uniform(-outer_r, outer_r, size=(n_new * 4, 3))
            dists = np.linalg.norm(batch, axis=1)
            valid = batch[(dists >= inner_r) & (dists < outer_r)]
            new_cells.extend(valid[:n_new - len(new_cells)])

        new_cells = np.array(new_cells[:n_new])

        # Append to existing arrays
        self._cell_x = np.append(self._cell_x, new_cells[:, 0] + self._cx)
        self._cell_y = np.append(self._cell_y, new_cells[:, 1] + self._cy)
        self._cell_z = np.append(self._cell_z, new_cells[:, 2])
        self._cell_radius = np.append(
            self._cell_radius, rng.uniform(3, 6, n_new)
        )
        self._cell_dist = np.append(
            self._cell_dist, np.linalg.norm(new_cells, axis=1)
        )

        # New cells are proliferating with bright calcein
        self._is_necrotic = np.append(self._is_necrotic, np.zeros(n_new, dtype=bool))
        self._is_quiescent = np.append(self._is_quiescent, np.zeros(n_new, dtype=bool))
        self._is_proliferating = np.append(self._is_proliferating, np.ones(n_new, dtype=bool))
        self._calcein = np.append(self._calcein, rng.uniform(160, 220, n_new).astype(np.float32))
        self._pi = np.append(self._pi, np.zeros(n_new, dtype=np.float32))
        self._gfp = np.append(self._gfp, rng.uniform(150, 220, n_new).astype(np.float32))

        self.n_cells += n_new

    def _reclassify_cells(self):
        """Reclassify cells based on current layer boundaries."""
        # Recompute distances
        self._cell_dist = np.sqrt(
            (self._cell_x - self._cx)**2 +
            (self._cell_y - self._cy)**2 +
            self._cell_z**2
        )

        old_necrotic = self._is_necrotic.copy()

        self._is_necrotic = self._cell_dist < self._necrotic_r
        self._is_quiescent = (
            (self._cell_dist >= self._necrotic_r) &
            (self._cell_dist < self._quiescent_r)
        )
        self._is_proliferating = self._cell_dist >= self._quiescent_r

        # Cells that just became necrotic: fade calcein, add PI
        newly_dead = self._is_necrotic & ~old_necrotic
        if newly_dead.any():
            n_new_dead = newly_dead.sum()
            self._calcein[newly_dead] = self._rng.uniform(0, 10, n_new_dead).astype(np.float32)
            self._pi[newly_dead] = self._rng.uniform(140, 220, n_new_dead).astype(np.float32)
            self._gfp[newly_dead] = self._rng.uniform(0, 15, n_new_dead).astype(np.float32)

        # Cells that became quiescent: dim calcein
        newly_quiescent = self._is_quiescent & ~old_necrotic & ~self._is_necrotic
        # Only update cells that were previously proliferating
        became_q = newly_quiescent & (self._calcein > 120)
        if became_q.any():
            n_q = became_q.sum()
            self._calcein[became_q] = self._rng.uniform(60, 120, n_q).astype(np.float32)

    # ── Drug response ──

    def apply_drug(self, name):
        """Apply a chemotherapy drug.

        Available drugs:
          - 'cisplatin': DNA crosslinker, kills proliferating cells
          - 'doxorubicin': topoisomerase inhibitor, kills proliferating + some quiescent
          - 'staurosporine': broad kinase inhibitor, rapid apoptosis
        """
        name = name.lower()
        if name not in self._drug_profiles:
            raise ValueError(f"Unknown drug: {name}. Available: {list(self._drug_profiles)}")
        self._drug_active = True
        self._drug_name = name
        self._drug_effect = 0.0
        self._drug_washing_out = False

    def remove_drug(self):
        """Remove drug — begins washout phase."""
        if self._drug_active:
            self._drug_active = False
            self._drug_washing_out = True

    def _update_drug_effect(self, dt: float = 1.0):
        """Update drug effect level (gradual onset/washout, dt-scaled)."""
        if not self._drug_active and not self._drug_washing_out:
            return
        profile = self._drug_profiles.get(self._drug_name, {})
        if self._drug_washing_out:
            rate = profile.get("washout_rate", 0.05) * dt
            self._drug_effect = max(0.0, self._drug_effect - rate)
            if self._drug_effect <= 0.01:
                self._drug_effect = 0.0
                self._drug_washing_out = False
                self._drug_name = None
        else:
            rate = profile.get("onset_rate", 0.10) * dt
            self._drug_effect = min(1.0, self._drug_effect + rate)

    def _apply_drug_killing(self, dt: float = 1.0):
        """Kill cells based on drug effect, cell state, and penetration depth.

        Drug concentration decreases with depth from the spheroid surface:
        local_conc = drug_effect * max(0, 1 - depth / penetration_depth)
        where depth = radius - cell_dist_from_center.
        """
        if self._drug_effect <= 0:
            return
        profile = self._drug_profiles.get(self._drug_name, {})
        rng = self._rng
        eff = self._drug_effect

        # Per-cell drug concentration (penetration gradient)
        pen_depth = profile.get("penetration_depth", 1e6)  # default: uniform
        depth_from_surface = self.radius - self._cell_dist
        local_conc = eff * np.clip(1.0 - depth_from_surface / pen_depth, 0, 1)

        # Kill proliferating cells (probability scaled by local concentration)
        base_prolif_rate = profile.get("prolif_kill_rate", 0.0) * dt
        if base_prolif_rate > 0:
            prolif_mask = self._is_proliferating & ~self._is_necrotic
            n_prolif = prolif_mask.sum()
            if n_prolif > 0:
                kill_prob = base_prolif_rate * local_conc
                kill_roll = rng.random(self.n_cells)
                killed = prolif_mask & (kill_roll < kill_prob)
                n_killed = killed.sum()
                if n_killed > 0:
                    self._is_necrotic[killed] = True
                    self._is_proliferating[killed] = False
                    self._calcein[killed] = rng.uniform(0, 10, n_killed).astype(np.float32)
                    self._pi[killed] = rng.uniform(140, 220, n_killed).astype(np.float32)
                    self._gfp[killed] = rng.uniform(0, 15, n_killed).astype(np.float32)
                    self._drug_killed += n_killed

        # Kill quiescent cells (probability scaled by local concentration)
        base_quiesc_rate = profile.get("quiesc_kill_rate", 0.0) * dt
        if base_quiesc_rate > 0:
            quiesc_mask = self._is_quiescent & ~self._is_necrotic
            n_quiesc = quiesc_mask.sum()
            if n_quiesc > 0:
                kill_prob = base_quiesc_rate * local_conc
                kill_roll = rng.random(self.n_cells)
                killed = quiesc_mask & (kill_roll < kill_prob)
                n_killed = killed.sum()
                if n_killed > 0:
                    self._is_necrotic[killed] = True
                    self._is_quiescent[killed] = False
                    self._calcein[killed] = rng.uniform(0, 10, n_killed).astype(np.float32)
                    self._pi[killed] = rng.uniform(140, 220, n_killed).astype(np.float32)
                    self._gfp[killed] = rng.uniform(0, 15, n_killed).astype(np.float32)
                    self._drug_killed += n_killed

    # ── Extra channels ──

    def add_channel(self, mode_id, name, led, filt, render_fn):
        """Register an extra fluorescence channel."""
        self._extra_channels[mode_id] = {
            "name": name, "led": led, "filter": filt,
            "render_fn": render_fn, "image": None,
        }

    # ── Ground truth ──

    def get_ground_truth(self):
        """Return ground truth for grading."""
        z_from_center = self.focal_plane - self.tissue_z

        # Slice radius at current Z
        slice_r_sq = self.radius**2 - z_from_center**2
        slice_radius = np.sqrt(max(0, slice_r_sq))

        # Necrotic core visible?
        nec_slice_r_sq = self._necrotic_r**2 - z_from_center**2
        nec_slice_radius = np.sqrt(max(0, nec_slice_r_sq))

        # Count cells visible at current Z
        dz = np.abs(self._cell_z - z_from_center)
        visible = dz <= self._dof / 2.0

        n_visible = visible.sum()
        n_live = (visible & ~self._is_necrotic).sum()
        n_dead = (visible & self._is_necrotic).sum()

        viability = n_live / max(1, n_visible)

        gt = {
            "spheroid_radius": round(self.radius, 1),
            "spheroid_diameter": round(self.radius * 2, 1),
            "necrotic_core_radius": round(self._necrotic_r, 1),
            "necrotic_core_diameter": round(self._necrotic_r * 2, 1),
            "quiescent_outer_radius": round(self._quiescent_r, 1),
            "n_cells_total": self.n_cells,
            "n_necrotic": int(self._is_necrotic.sum()),
            "n_quiescent": int(self._is_quiescent.sum()),
            "n_proliferating": int(self._is_proliferating.sum()),
            "overall_viability": round(1.0 - self._is_necrotic.sum() / self.n_cells, 3),
            # At current Z-plane
            "current_z": round(z_from_center, 1),
            "slice_radius": round(slice_radius, 1),
            "necrotic_slice_radius": round(nec_slice_radius, 1),
            "n_visible": int(n_visible),
            "n_visible_live": int(n_live),
            "n_visible_dead": int(n_dead),
            "slice_viability": round(viability, 3),
        }
        if self._growth_enabled:
            gt["time"] = round(self._time, 1)
            gt["initial_radius"] = round(self._initial_radius, 1)
            gt["growth_rate"] = round(self._growth_rate, 2)

        if self._drug_active or self._drug_washing_out:
            profile = self._drug_profiles.get(self._drug_name, {})
            pen = profile.get("penetration_depth", 1e6)
            gt["drug"] = {
                "name": self._drug_name,
                "effect": round(self._drug_effect, 3),
                "active": self._drug_active,
                "washing_out": self._drug_washing_out,
                "total_killed": self._drug_killed,
                "penetration_depth": round(pen, 1),
            }

        return gt

    def get_z_profile(self, n_slices=20):
        """Get viability profile across Z slices."""
        z_values = np.linspace(-self.radius, self.radius, n_slices)
        profile = []
        for z in z_values:
            dz = np.abs(self._cell_z - z)
            visible = dz <= self._dof / 2.0
            n_vis = visible.sum()
            n_live = (visible & ~self._is_necrotic).sum()
            profile.append({
                "z": round(float(z), 1),
                "n_visible": int(n_vis),
                "viability": round(n_live / max(1, n_vis), 3),
            })
        return profile
