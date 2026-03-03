"""
Tissue Dynamics — Cell migration, division, and wound healing.

Extends VoronoiSim with temporal evolution:
  - Wound creation (rectangular or circular cell removal)
  - Edge migration (cells near wound move toward gap)
  - Random motility (Brownian-like cell movement)
  - Cell division (probabilistic, area-dependent)
  - Apoptosis (probabilistic cell death)

Key design: Move centroids → recompute Voronoi → re-render.
For N=200 cells, recomputation takes ~1ms. Acceptable for 10-50 frames.

Usage:
    from tissue_dynamics import DynamicVoronoiSim

    sim = DynamicVoronoiSim(width=1024, height=1024, n_cells=200, ...)
    sim.create_wound(shape="rectangle", center=(512, 512), size=(200, 1024))

    for t in range(20):
        sim.step(dt=1.0)
        # snap_frame() will use updated tissue
"""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.voronoi import VoronoiSim


# Electrode label → unit migration vector (toward cathode)
_ELECTRODE_VECTORS = {
    "+X": np.array([1.0, 0.0]),
    "-X": np.array([-1.0, 0.0]),
    "+Y": np.array([0.0, 1.0]),
    "-Y": np.array([0.0, -1.0]),
}


def _electrode_label_to_vector(label: str) -> list:
    """Convert Electrode device label to [dx, dy] unit vector, or [0,0] if Off."""
    v = _ELECTRODE_VECTORS.get(label, np.zeros(2))
    return v.tolist()


class DynamicVoronoiSim(VoronoiSim):
    """VoronoiSim with temporal dynamics (wound healing, migration, division)."""

    continuous = True

    def __init__(
        self,
        migration_speed: float = 2.0,
        random_motility: float = 0.5,
        division_rate: float = 0.0,
        apoptosis_rate: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            migration_speed: Speed of directed migration toward wound (px/step)
            random_motility: Random walk magnitude (px/step)
            division_rate: Probability of division per cell per step
            apoptosis_rate: Probability of death per cell per step
            **kwargs: Passed to VoronoiSim.__init__
        """
        super().__init__(**kwargs)

        self.migration_speed = migration_speed
        self.random_motility = random_motility
        self.division_rate = division_rate
        self.apoptosis_rate = apoptosis_rate

        # Cell state
        self.alive = np.ones(self.n_cells, dtype=bool)
        self.velocities = np.zeros((self.n_cells, 2))
        self.frame_count = 0

        # Apoptosis state: tracks dying cells through visible morphological stages
        # 0 = healthy, 1+ = apoptosis stage (higher = further along)
        self.apoptosis_stage = np.zeros(self.n_cells, dtype=int)
        self.apoptosis_max_stage = 4  # stages before cell becomes ghost
        # Shrinkage factor per stage (1.0 = full size, 0.0 = vanished)
        self._apoptosis_shrink = np.array([1.0, 0.85, 0.65, 0.45, 0.0])

        # Fluorescence dynamics: per-cell intensity change over time
        # Each entry is a callable(cell_idx, time_step) -> new_intensity or None
        self._nuc_dynamics = None   # function(idx, t) -> float or None
        self._mem_dynamics = None   # function(idx, t) -> float or None

        # Wound state
        self.wound_region = None  # (shape, center, size) or None
        self._wound_mask = None   # bool array [height, width]

        # Auto-step mode: advance dynamics automatically on snap_frame()
        self.auto_step = False         # enable auto-advancing
        self.auto_step_dt = 1.0        # timestep per advance
        self.snaps_per_step = 1        # snap calls between each step
        self._snap_counter = 0         # internal counter

        # SLM optogenetic stimulation: illuminated cells migrate faster
        self._stim_mask = None         # bool array [height, width] from SLM
        self.stim_speed_multiplier = 2.0  # how much faster stimulated cells move

        # Optogenetic gene induction: SLM illumination drives expression
        self._gene_induction_enabled = False
        self._gene_induction_rate = 0.0    # expression increase per step under SLM
        self._gene_induction_max = 1.0     # max expression level (0-1)
        self._gene_decay_rate = 0.0        # passive expression decay per step
        self._gene_expression = np.zeros(self.n_cells, dtype=float)  # per-cell
        self._gene_baseline = None         # baseline intensities before induction
        self._cumulative_illumination = np.zeros(self.n_cells, dtype=float)

        # Ghost weight for wound healing: each ghost starts at weight 1.0
        # (full cell-sized gap). Weight decays over time, displacing the
        # ghost centroid toward its nearest alive neighbor. When weight → 0,
        # the ghost is removed entirely and the alive neighbor's Voronoi
        # polygon expands naturally. No new tiny cells are created.
        self._ghost_weight = np.ones(self.n_cells, dtype=float)

        self.exclude_ghosts_from_voronoi = False

        # Homeostatic mode: balanced division + apoptosis to maintain
        # a target cell count indefinitely. When enabled, the tissue
        # sustains interesting dynamics without growing or shrinking.
        self.homeostatic = False
        self.target_cells = self.n_cells  # target = initial count
        self._homeostatic_base_div = 0.005  # base division rate when homeostatic
        self._homeostatic_base_apo = 0.005  # base apoptosis rate when homeostatic

        # Nuclear translocation: drug-induced protein shuttling (e.g. NF-kB)
        self._transloc_enabled = False
        self._transloc_nuc_fraction = np.full(self.n_cells, 0.2)  # 20% nuclear at rest
        self._transloc_import_rate = 0.08   # nuclear import rate under drug
        self._transloc_export_rate = 0.03   # nuclear export rate (baseline re-export)
        self._transloc_baseline = 0.2       # resting nuclear fraction
        self._transloc_drug_active = False   # set by Perfusion device or manual flag
        self._transloc_total_reporter = np.ones(self.n_cells)  # total GFP per cell
        self._transloc_mode_id = None       # mode_id of extra GFP channel

        # Z-drift: tissue slowly moves out of focus during timelapse
        # Simulates thermal drift, mechanical relaxation, etc.
        self.z_drift_rate = 0.0    # µm/s (positive = tissue drifts up)
        self.z_drift_noise = 0.0   # σ of z-jitter (µm·s⁻½, Brownian)
        self._initial_tissue_z = self.tissue_z  # store for drift measurement

        # Extension hooks: subclasses register callbacks for modular features
        self._step_hooks: list = []       # called after core dynamics in step()
        self._render_hooks: dict = {}     # mode_id → render callable
        self._divide_hooks: list = []     # called in _divide_append_one(parent_idx)

    # ---- Fluorescence dynamics ----

    def set_fluorescence_dynamics(self, channel: str, fn):
        """Set a function that updates per-cell fluorescence each step.

        Args:
            channel: "nucleus" or "membrane"
            fn: Callable(cell_idx: int, time_step: int) -> float or None
                Returns new intensity (0-1) or None to leave unchanged.
                Called for every alive cell at each step().
        """
        if channel == "nucleus":
            self._nuc_dynamics = fn
        elif channel == "membrane":
            self._mem_dynamics = fn

    def _apply_fluorescence_dynamics(self):
        """Update fluorescence intensities based on dynamics functions."""
        if self._nuc_dynamics is not None:
            for i in range(self.n_cells):
                if not self.alive[i]:
                    continue
                val = self._nuc_dynamics(i, self.frame_count)
                if val is not None:
                    self.nucleus_intensity[i] = np.clip(val, 0.0, 1.0)

        if self._mem_dynamics is not None:
            for i in range(self.n_cells):
                if not self.alive[i]:
                    continue
                val = self._mem_dynamics(i, self.frame_count)
                if val is not None:
                    self.membrane_intensity[i] = np.clip(val, 0.0, 1.0)

    # ---- Optogenetic gene induction ----

    def enable_gene_induction(self, rate: float = 0.05, max_expression: float = 1.0,
                              decay_rate: float = 0.01):
        """Enable SLM-driven optogenetic gene induction.

        When the SLM illuminates cells, their nuclear fluorescence gradually
        increases (simulating light-activated gene expression, e.g.
        photoactivatable GFP, CRE-lox, optogenetic promoter systems).

        Expression accumulates while illuminated and decays slowly when
        illumination is removed (protein degradation).

        Args:
            rate: Expression increase per step for illuminated cells (0-1 scale).
                  At rate=0.05, a cell reaches ~50% expression in 10 steps.
            max_expression: Maximum nuclear intensity (0-1 scale).
            decay_rate: Passive decay per step when NOT illuminated.
                        At 0.01, half-life is ~70 steps.
        """
        self._gene_induction_enabled = True
        self._gene_induction_rate = rate
        self._gene_induction_max = max_expression
        self._gene_decay_rate = decay_rate
        self._gene_baseline = self.nucleus_intensity.copy()
        self._gene_expression = np.zeros(self.n_cells, dtype=float)
        self._cumulative_illumination = np.zeros(self.n_cells, dtype=float)

    def _apply_gene_induction(self):
        """Update per-cell gene expression based on SLM illumination."""
        if not self._gene_induction_enabled:
            return

        for i in range(self.n_cells):
            if not self.alive[i]:
                continue

            # Check if cell is under SLM illumination
            illuminated = False
            if self._stim_mask is not None:
                px = int(np.clip(self.centers[i][0], 0, self.width - 1))
                py = int(np.clip(self.centers[i][1], 0, self.height - 1))
                illuminated = bool(self._stim_mask[py, px])

            if illuminated:
                # Expression increases toward max
                self._gene_expression[i] += self._gene_induction_rate * (
                    self._gene_induction_max - self._gene_expression[i]
                )
                self._cumulative_illumination[i] += 1.0
            else:
                # Expression decays toward zero
                self._gene_expression[i] *= (1.0 - self._gene_decay_rate)

            self._gene_expression[i] = np.clip(
                self._gene_expression[i], 0.0, self._gene_induction_max
            )

            # Update nuclear fluorescence: baseline + induction
            base = self._gene_baseline[i] if self._gene_baseline is not None else 0.3
            self.nucleus_intensity[i] = np.clip(
                base + self._gene_expression[i], 0.0, 1.0
            )

    def get_gene_expression(self):
        """Return per-cell gene expression levels.

        Returns:
            dict with:
                expression: array of per-cell expression (0 to max)
                illuminated_cells: indices of currently illuminated cells
                mean_expression: mean over alive cells
                induced_cells: indices where expression > 0.1
        """
        alive_mask = self.alive[:len(self._gene_expression)]
        expr = self._gene_expression[:self.n_cells]

        illuminated = []
        if self._stim_mask is not None:
            for i in range(self.n_cells):
                if not self.alive[i]:
                    continue
                px = int(np.clip(self.centers[i][0], 0, self.width - 1))
                py = int(np.clip(self.centers[i][1], 0, self.height - 1))
                if self._stim_mask[py, px]:
                    illuminated.append(i)

        return {
            "expression": expr.copy(),
            "illuminated_cells": illuminated,
            "mean_expression": float(expr[alive_mask].mean()) if alive_mask.any() else 0.0,
            "induced_cells": list(np.where((expr > 0.1) & alive_mask)[0]),
            "cumulative_illumination": self._cumulative_illumination[:self.n_cells].copy(),
        }

    # ---- Photoconversion ----

    def enable_photoconversion(self, conversion_threshold: int = 1):
        """Enable SLM-driven photoconversion of fluorescent proteins.

        Simulates Kaede/Dendra2/mEos-type photoconvertible proteins.
        Before conversion: cells are bright in nucleus channel (green),
        dim in membrane channel (red).  After SLM illumination:
        converted cells become dim in nucleus (green) and bright in
        membrane (red).  Conversion is PERMANENT and irreversible.

        This enables cell fate tracking: illuminate a region, then
        observe which cells (now red) migrate to new positions.

        Args:
            conversion_threshold: Number of illumination steps needed to
                convert a cell. Default 1 (instant conversion).
        """
        self._photoconversion_enabled = True
        self._conversion_threshold = conversion_threshold
        self._converted = np.zeros(self.n_cells, dtype=bool)
        self._conversion_exposure = np.zeros(self.n_cells, dtype=int)

        # Set initial fluorescence: green-bright, red-dim
        self._pre_nuc = self.nucleus_intensity.copy()    # save green baseline
        self._pre_mem = self.membrane_intensity.copy()    # save red baseline
        self._green_bright = 0.75  # nucleus intensity for unconverted
        self._green_dim = 0.10     # nucleus intensity for converted
        self._red_dim = 0.10       # membrane intensity for unconverted
        self._red_bright = 0.80    # membrane intensity for converted

        # Apply initial state
        self.nucleus_intensity[:] = self._green_bright
        self.membrane_intensity[:] = self._red_dim

    def _apply_photoconversion(self):
        """Check SLM mask and convert illuminated cells."""
        if not getattr(self, '_photoconversion_enabled', False):
            return

        if self._stim_mask is None:
            return

        newly_converted = False
        for i in range(self.n_cells):
            if not self.alive[i] or self._converted[i]:
                continue

            px = int(np.clip(self.centers[i][0], 0, self.width - 1))
            py = int(np.clip(self.centers[i][1], 0, self.height - 1))

            if self._stim_mask[py, px]:
                self._conversion_exposure[i] += 1
                if self._conversion_exposure[i] >= self._conversion_threshold:
                    self._converted[i] = True
                    newly_converted = True

        # Update fluorescence for all cells
        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            if self._converted[i]:
                # Converted: green-dim, red-bright
                self.nucleus_intensity[i] = self._green_dim
                self.membrane_intensity[i] = self._red_bright
            else:
                # Unconverted: green-bright, red-dim
                self.nucleus_intensity[i] = self._green_bright
                self.membrane_intensity[i] = self._red_dim

    def get_photoconversion_state(self):
        """Return per-cell photoconversion state.

        Returns:
            dict with:
                converted: bool array of per-cell conversion status
                n_converted: number of converted cells
                n_unconverted: number of alive unconverted cells
                converted_indices: list of converted cell indices
        """
        if not getattr(self, '_photoconversion_enabled', False):
            return {"converted": np.array([]), "n_converted": 0,
                    "n_unconverted": 0, "converted_indices": []}

        alive_mask = self.alive[:self.n_cells]
        conv = self._converted[:self.n_cells]
        n_conv = int((conv & alive_mask).sum())
        n_unconv = int((~conv & alive_mask).sum())

        return {
            "converted": conv.copy(),
            "n_converted": n_conv,
            "n_unconverted": n_unconv,
            "converted_indices": list(np.where(conv & alive_mask)[0]),
        }

    # ---- Laser ablation ----

    def enable_laser_ablation(self, kill_threshold: float = 3.0,
                               damage_rate: float = 1.0,
                               healing_response: bool = True):
        """Enable SLM-driven laser ablation of cells.

        Simulates UV/two-photon laser ablation: cells under SLM illumination
        accumulate damage each step.  When cumulative damage exceeds
        ``kill_threshold``, the cell dies and becomes a ghost (invisible,
        holds Voronoi space).

        As cells take damage, their nuclei brighten (pyknosis — chromatin
        condensation visible in DAPI channel).  After death, surrounding
        cells respond with wound healing: directed migration toward the gap.

        The SLM mask defines the ablation pattern — the agent draws where
        the laser fires.  Threshold > 1 requires sustained illumination,
        modeling realistic two-photon ablation that needs multiple pulses.

        Args:
            kill_threshold: Cumulative damage needed to kill a cell.
                Default 3.0 (needs 3 steps of illumination at rate 1.0).
            damage_rate: Damage accumulated per step under SLM illumination.
            healing_response: If True, enable directed migration toward
                the ablation site after cell death.
        """
        self._ablation_enabled = True
        self._ablation_threshold = kill_threshold
        self._ablation_damage_rate = damage_rate
        self._ablation_healing = healing_response
        self._laser_damage = np.zeros(self.n_cells, dtype=float)
        self._ablated = np.zeros(self.n_cells, dtype=bool)
        self._ablation_center = None  # mean position of ablated cells

        # Ghost cells stay in tessellation — their regions render as empty
        # background, creating a visible wound gap. Wound closure happens
        # via gradual ghost centroid drift + colonization.
        self.exclude_ghosts_from_voronoi = False

    def _apply_laser_ablation(self):
        """Check SLM mask and apply laser damage to illuminated cells."""
        if not getattr(self, '_ablation_enabled', False):
            return

        if self._stim_mask is None:
            return

        newly_killed = []
        for i in range(self.n_cells):
            if not self.alive[i] or self._ablated[i]:
                continue

            px = int(np.clip(self.centers[i][0], 0, self.width - 1))
            py = int(np.clip(self.centers[i][1], 0, self.height - 1))

            if self._stim_mask[py, px]:
                self._laser_damage[i] += self._ablation_damage_rate

                # Pyknosis: nucleus brightens as chromatin condenses
                if self.has_nucleus_marker[i]:
                    frac = min(1.0, self._laser_damage[i] / self._ablation_threshold)
                    self.nucleus_intensity[i] = min(
                        1.0, self.nucleus_intensity[i] * (1.0 + 0.5 * frac))

                if self._laser_damage[i] >= self._ablation_threshold:
                    # Cell dies — becomes ghost with full weight
                    self.alive[i] = False
                    self._renderable[i] = False
                    self._ablated[i] = True
                    if i < len(self._ghost_weight):
                        self._ghost_weight[i] = 1.0
                    newly_killed.append(i)

        # Update wound region for directed migration
        if newly_killed or (self._ablation_center is not None
                            and self._ablation_healing):
            ablated_idx = np.where(self._ablated[:self.n_cells])[0]
            if len(ablated_idx) > 0:
                self._ablation_center = self.centers[ablated_idx].mean(axis=0)

                if self._ablation_healing:
                    cx, cy = self._ablation_center
                    if len(ablated_idx) > 1:
                        dists = np.sqrt(
                            (self.centers[ablated_idx, 0] - cx) ** 2 +
                            (self.centers[ablated_idx, 1] - cy) ** 2
                        )
                        radius = float(dists.max()) + 30
                    else:
                        radius = 30.0
                    self.wound_region = ("circle", (cx, cy), (radius,))
                    # Build wound mask for get_wound_area() etc.
                    Y, X = np.ogrid[:self.height, :self.width]
                    self._wound_mask = ((X - cx) ** 2 + (Y - cy) ** 2) < radius ** 2

    def get_ablation_state(self):
        """Return laser ablation state for ground truth.

        Returns:
            dict with per-cell damage, kill counts, and wound geometry.
        """
        if not getattr(self, '_ablation_enabled', False):
            return {"n_ablated": 0, "n_damaged": 0, "n_alive": 0,
                    "ablated_indices": [], "ablation_center": None}

        ablated = self._ablated[:self.n_cells]
        damage = self._laser_damage[:self.n_cells]
        alive_mask = self.alive[:self.n_cells]

        return {
            "n_ablated": int(ablated.sum()),
            "n_damaged": int(((damage > 0) & ~ablated & alive_mask).sum()),
            "n_alive": int(alive_mask.sum()),
            "ablated_indices": list(np.where(ablated)[0]),
            "ablation_center": (self._ablation_center.tolist()
                                if self._ablation_center is not None else None),
        }

    # ---- Nuclear translocation ----

    def enable_translocation(self, import_rate: float = 0.08,
                              export_rate: float = 0.03,
                              baseline_nuc_fraction: float = 0.2,
                              heterogeneity: float = 0.15):
        """Enable drug-induced nuclear translocation reporter (e.g. NF-kB-GFP).

        Models a fluorescent protein that shuttles between cytoplasm and nucleus.
        At rest, most protein is cytoplasmic (low nuc_fraction). Drug treatment
        triggers nuclear import (nuc_fraction increases). Washout allows
        re-export to cytoplasm.

        The GFP reporter is registered as an extra channel ("gfp-channel").
        Agent uses DAPI (nucleus-channel) for segmentation, E-cadherin
        (membrane-channel) for cell boundaries, and gfp-channel to measure
        N:C ratio.

        Args:
            import_rate: Rate of nuclear import under drug (per step).
                         At 0.08, nuc_fraction reaches ~0.8 in 20 steps.
            export_rate: Rate of nuclear export when drug removed (per step).
                         At 0.03, half-life ~23 steps.
            baseline_nuc_fraction: Resting nuclear fraction (0-1).
            heterogeneity: Per-cell variation in total reporter (σ, relative).
        """
        self._transloc_enabled = True
        self._transloc_import_rate = import_rate
        self._transloc_export_rate = export_rate
        self._transloc_baseline = baseline_nuc_fraction
        self._transloc_nuc_fraction = np.full(self.n_cells, baseline_nuc_fraction)
        # Per-cell total reporter with heterogeneity (some cells load more dye)
        self._transloc_total_reporter = np.clip(
            1.0 + self.rng.normal(0, heterogeneity, self.n_cells), 0.3, 1.7
        )

    def enable_translocation_channel(self, core=None):
        """Register GFP reporter as extra channel for translocation imaging.

        Call after enable_translocation(). Renders initial GFP distribution
        and registers it as 'gfp-channel' in the channel system.

        Args:
            core: Optional CMMCore to register Channel config group entry.

        Returns:
            mode_id for the GFP channel.
        """
        gfp_img = self._render_translocation_full()
        mode_id = self.add_channel(
            "gfp", gfp_img,
            filter_label="TagGFP2(483/506)", led_label="GREEN",
        )
        self._transloc_mode_id = mode_id
        if core is not None:
            core.defineConfig("Channel", "gfp-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Channel", "gfp-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")
        return mode_id

    def set_translocation_drug(self, active: bool):
        """Set whether translocation-inducing drug is active."""
        self._transloc_drug_active = active

    def _sync_perfusion_drug_state(self):
        """Sync drug-dependent reporters with Perfusion device state.

        Called once per step() to couple Perfusion state 4 ("Drug") to
        translocation and stress granule activation. This avoids requiring
        manual snap_frame hooks in challenge scenarios.
        """
        perf = self.state_devices.get("Perfusion", {})
        if not isinstance(perf, dict):
            return
        drug_on = str(perf.get("state", "0")) == "4"
        if self._transloc_enabled:
            self._transloc_drug_active = drug_on
        if hasattr(self, '_sg_stress_active'):
            self._sg_stress_active = drug_on

    def _apply_translocation(self, dt: float = 1.0):
        """Update nuclear fraction based on drug state."""
        if not self._transloc_enabled:
            return
        n = self.n_cells
        frac = self._transloc_nuc_fraction[:n]
        if self._transloc_drug_active:
            # Nuclear import: fraction increases toward 1.0
            frac += self._transloc_import_rate * (1.0 - frac) * dt
        else:
            # Nuclear export: fraction decays toward baseline
            frac -= self._transloc_export_rate * (frac - self._transloc_baseline) * dt
        self._transloc_nuc_fraction[:n] = np.clip(frac, 0.0, 1.0)

    def get_translocation_state(self):
        """Return translocation state for ground truth.

        Returns:
            dict with mean/std nuclear fraction and per-cell data.
        """
        if not self._transloc_enabled:
            return {"enabled": False}
        n = self.n_cells
        alive_mask = self.alive[:n]
        fracs = self._transloc_nuc_fraction[:n][alive_mask]
        return {
            "enabled": True,
            "drug_active": self._transloc_drug_active,
            "mean_nuc_fraction": round(float(fracs.mean()), 3),
            "std_nuc_fraction": round(float(fracs.std()), 3),
            "n_alive": int(alive_mask.sum()),
            "baseline_nuc_fraction": self._transloc_baseline,
        }

    # ---- Z-drift helpers ----

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift since initialization (µm)."""
        return self.tissue_z - self._initial_tissue_z

    def reset_z_drift(self):
        """Reset tissue Z to initial position."""
        self.tissue_z = self._initial_tissue_z


    # ---- Galvanotaxis (electric field) ----

    def enable_galvanotaxis(self, strength: float = 2.0, noise_frac: float = 0.3):
        """Enable electric-field-directed cell migration (galvanotaxis).

        When enabled, the Electrode device (state_devices["Electrode"]) controls
        which direction the electric field points.  Cells migrate toward the
        cathode (negative electrode) — a well-documented behaviour in epithelial
        and fibroblast cultures.

        Args:
            strength: Base migration speed in the field direction (px/step,
                      scaled by dt and temperature).  Default 2.0 px/step.
            noise_frac: Perpendicular noise as a fraction of strength (0-1).
                        Models the imperfect alignment of real galvanotaxis.
                        Default 0.3 (30 % lateral spread).
        """
        self._galvanotaxis_strength = float(strength)
        self._galvanotaxis_noise = float(noise_frac)

    def get_galvanotaxis_state(self) -> dict:
        """Return current galvanotaxis parameters for ground truth."""
        mode = self.state_devices.get("Electrode", {}).get("label", "Off")
        return {
            "electrode_mode": mode,
            "strength": getattr(self, "_galvanotaxis_strength", 0.0),
            "field_vector": _electrode_label_to_vector(mode),
        }

    # ---- Wound creation ----

    def create_wound(self, shape: str = "rectangle", center: tuple = (512, 512),
                     size: tuple = (200, 1024), jagged: float = 0.0):
        """Create a wound by marking cells as ghost (invisible but space-holding).

        Ghost cells still participate in Voronoi tessellation (holding their
        space) but are not rendered in any channel. This creates a visible gap
        in the tissue, unlike the old approach where removing cells caused
        remaining cells to expand and fill the void.

        Args:
            shape: "rectangle" or "circle"
            center: (x, y) center of the wound
            size: (width, height) for rectangle, or (radius,) for circle
            jagged: 0.0 to 1.0 — stochastic edge irregularity.
                0 = perfectly geometric (default), 1 = very jagged edges.
                Cells near the wound boundary are probabilistically included/
                excluded, creating a realistic irregular scratch pattern.
        """
        self.wound_region = (shape, center, size)
        cx, cy = center

        # Edge zone width: proportional to cell spacing for realistic irregularity
        expected_spacing = np.sqrt(self.width * self.height / max(1, self.n_cells))
        edge_zone = jagged * expected_spacing * 0.8

        # Determine which cells are in the wound
        wounded = []
        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            px, py = self.centers[i]

            if shape == "rectangle":
                w, h = size
                # Signed distance: negative = inside wound, positive = outside
                dx = abs(px - cx) - w / 2
                dy = abs(py - cy) - h / 2
                signed_dist = max(dx, dy)
            elif shape == "circle":
                r = size[0] if isinstance(size, tuple) else size
                signed_dist = np.hypot(px - cx, py - cy) - r
            else:
                continue

            if edge_zone > 0 and abs(signed_dist) < edge_zone:
                # Transition zone: probabilistic removal
                # p=1 at center of wound, p=0 at edge, smooth sigmoid
                t = (signed_dist + edge_zone) / (2 * edge_zone)  # 0..1
                prob_wound = 1.0 - t  # higher = more likely wounded
                if self.rng.random() < prob_wound:
                    wounded.append(i)
            elif signed_dist < 0:
                # Clearly inside wound
                wounded.append(i)

        # Mark wound cells as ghost (alive for Voronoi, invisible for rendering)
        for i in wounded:
            self.alive[i] = False
            self._renderable[i] = False
            if i < len(self._ghost_weight):
                self._ghost_weight[i] = 1.0

        # Build wound mask for rendering
        self._wound_mask = np.zeros((self.height, self.width), dtype=bool)
        if shape == "rectangle":
            w, h = size
            x1 = max(0, int(cx - w / 2))
            x2 = min(self.width, int(cx + w / 2))
            y1 = max(0, int(cy - h / 2))
            y2 = min(self.height, int(cy + h / 2))
            self._wound_mask[y1:y2, x1:x2] = True
        elif shape == "circle":
            r = size[0] if isinstance(size, tuple) else size
            Y, X = np.ogrid[:self.height, :self.width]
            self._wound_mask = ((X - cx)**2 + (Y - cy)**2) < r**2

        # Invalidate render cache (ghost cells now skip rendering)
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None

        n_wounded = len(wounded)
        n_alive = int(self.alive.sum())
        return {"n_killed": n_wounded, "n_alive": n_alive}

    # ---- Temperature response ----

    def _temp_rate_factor(self) -> float:
        """Temperature-dependent rate scaling for mammalian tissue.

        Optimal at 37°C. Q10 ~ 2.0 for cell biology processes.
        Cold arrest below 10°C, heat shock above 42°C.
        """
        temp = self._get_temperature()
        if temp < 10:
            return 0.05  # cold arrest
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.05, 1.0 - (temp - 42) * 0.3)
        return factor

    # ---- Dynamics steps ----

    def step(self, dt: float = 1.0):
        """Advance simulation by one timestep.

        1. Compute forces (wound attraction, random walk, neighbor repulsion)
        2. Move centroids
        3. Colonize ghost cells near leading edge (wound closure)
        4. Handle division and apoptosis
        5. Recompute Voronoi tessellation
        """
        # Scale effective dt by temperature factor
        temp_factor = self._temp_rate_factor()
        effective_dt = dt * temp_factor

        self._migrate(effective_dt)
        self._colonize_ghosts()

        # Homeostatic feedback: adjust rates to maintain target cell count
        if self.homeostatic:
            self._homeostatic_step(effective_dt)
        else:
            if self.division_rate > 0:
                self._divide(effective_dt)
            if self.apoptosis_rate > 0:
                self._apoptose(effective_dt)

        # Update fluorescence intensities before re-rendering
        if self._nuc_dynamics is not None or self._mem_dynamics is not None:
            self._apply_fluorescence_dynamics()

        # SLM-driven gene induction (updates nucleus_intensity based on illumination)
        self._apply_gene_induction()

        # SLM-driven photoconversion (permanent green→red switch)
        self._apply_photoconversion()

        # SLM-driven laser ablation (cell killing + wound healing)
        self._apply_laser_ablation()

        # Auto-couple Perfusion device to drug-dependent reporters
        self._sync_perfusion_drug_state()

        # Drug-induced nuclear translocation (NF-kB, etc.)
        self._apply_translocation(effective_dt)

        # Z-drift: tissue plane moves, causing defocus unless agent refocuses
        self._accumulate_z_drift(dt)

        # Extension hooks (FUCCI, lysosomes, stress granules, etc.)
        for hook in self._step_hooks:
            hook(effective_dt)

        self._recompute_tissue()
        # Render cache invalidated by _recompute_tissue — lazy re-render
        # on next snap_frame(). This avoids expensive rendering during
        # background dynamics or rapid stepping.
        self.frame_count += 1

    def step_autonomous(self, dt: float = 1.0):
        """Advance simulation without SLM-dependent effects.

        Used by RealtimeEngine for background dynamics. Skips gene induction
        and photoconversion (which are observation-coupled via SLM mask).
        Migration, cell cycle, FUCCI, Z-drift all advance normally.
        """
        temp_factor = self._temp_rate_factor()
        effective_dt = dt * temp_factor

        self._migrate(effective_dt)
        self._colonize_ghosts()

        if self.homeostatic:
            self._homeostatic_step(effective_dt)
        else:
            if self.division_rate > 0:
                self._divide(effective_dt)
            if self.apoptosis_rate > 0:
                self._apoptose(effective_dt)

        if self._nuc_dynamics is not None or self._mem_dynamics is not None:
            self._apply_fluorescence_dynamics()

        # SLM-driven effects are continuous illumination — apply in background
        self._apply_gene_induction()
        self._apply_photoconversion()
        self._apply_laser_ablation()

        # Auto-couple Perfusion device to drug-dependent reporters
        self._sync_perfusion_drug_state()

        # Translocation is drug-coupled (not SLM), so advance it
        self._apply_translocation(effective_dt)

        # Extension hooks (FUCCI, lysosomes, stress granules, etc.)
        for hook in self._step_hooks:
            hook(effective_dt)

        self._accumulate_z_drift(dt)

        self._recompute_tissue()
        # Lazy rendering — snap_frame() will re-render on next call
        self.frame_count += 1

    def _colonize_ghosts(self):
        """Gradual wound closure via ghost weight decay.

        Ghost cells stay in the Voronoi tessellation with a decaying weight.
        As weight decays, the ghost centroid moves toward its nearest alive
        neighbor, effectively shrinking the ghost's Voronoi region and letting
        alive neighbors expand into the wound gap.

        When weight reaches 0, the ghost centroid is placed on top of its
        nearest alive neighbor (zero-area Voronoi region = effectively removed).
        No new tiny cells are created — existing cells grow to fill the gap.
        """
        ghost_indices = np.where(~self.alive[:self.n_cells])[0]
        if len(ghost_indices) == 0:
            return

        alive_indices = np.where(self.alive[:self.n_cells])[0]
        if len(alive_indices) == 0:
            return

        alive_centers = self.centers[alive_indices]

        # Weight decay rate: faster migration = faster wound closure
        decay_rate = 0.03 * max(1.0, self.migration_speed / 5.0)

        for gi in ghost_indices:
            if gi >= len(self._ghost_weight):
                continue

            # Decay weight
            self._ghost_weight[gi] = max(0.0, self._ghost_weight[gi] - decay_rate)

            # Find nearest alive neighbor
            gx, gy = self.centers[gi]
            dists = np.sqrt((alive_centers[:, 0] - gx)**2 +
                            (alive_centers[:, 1] - gy)**2)
            nearest_alive = alive_indices[dists.argmin()]
            nx, ny = self.centers[nearest_alive]

            # Interpolate ghost centroid toward alive neighbor based on weight
            # weight=1.0 → ghost stays at original position (full gap)
            # weight=0.0 → ghost overlaps alive neighbor (zero-area region)
            w = self._ghost_weight[gi]
            if w <= 0:
                # Ghost fully absorbed — place on top of nearest alive cell
                self.centers[gi] = self.centers[nearest_alive].copy()
            else:
                # Gradual drift: move a fraction toward the alive neighbor
                t = 1.0 - w  # interpolation parameter (0→original, 1→neighbor)
                # Use the original position saved when cell died
                # Since we drift incrementally, just move the centroid
                dx = nx - gx
                dy = ny - gy
                step = (1.0 - w) * 0.15  # accelerates as weight drops
                self.centers[gi, 0] += dx * step
                self.centers[gi, 1] += dy * step

    def _migrate(self, dt: float):
        """Move cells based on forces.

        If an SLM stimulation mask is active, cells whose centroid falls
        under illuminated pixels have their migration speed multiplied by
        ``stim_speed_multiplier`` (default 2x). This simulates optogenetic
        acceleration of wound-edge cell migration.
        """
        for i in range(self.n_cells):
            if not self.alive[i]:
                continue

            # Check if cell is illuminated by SLM
            speed_mult = 1.0
            if self._stim_mask is not None:
                px, py = int(self.centers[i][0]), int(self.centers[i][1])
                px = np.clip(px, 0, self.width - 1)
                py = np.clip(py, 0, self.height - 1)
                if self._stim_mask[py, px]:
                    speed_mult = self.stim_speed_multiplier

            force = np.zeros(2)

            # Directed migration toward wound
            if self.wound_region is not None and self.migration_speed > 0:
                wound_force = self._wound_attraction(i)
                force += wound_force * self.migration_speed * speed_mult

            # Random motility (also boosted by stimulation)
            if self.random_motility > 0:
                force += self.rng.normal(0, self.random_motility * speed_mult, 2)

            # Galvanotaxis: directed migration in electric field (toward cathode)
            galvano_strength = getattr(self, "_galvanotaxis_strength", 0.0)
            if galvano_strength > 0.0:
                elec_label = self.state_devices.get("Electrode", {}).get("label", "Off")
                field_vec = _ELECTRODE_VECTORS.get(elec_label)
                if field_vec is not None:
                    noise_frac = getattr(self, "_galvanotaxis_noise", 0.3)
                    # Primary component: along field direction
                    galvano_force = field_vec * galvano_strength
                    # Lateral noise: perpendicular to field direction
                    perp = np.array([-field_vec[1], field_vec[0]])
                    galvano_force += perp * self.rng.normal(0, galvano_strength * noise_frac)
                    force += galvano_force

            # Neighbor repulsion (prevent overlap)
            force += self._neighbor_repulsion(i)

            # Apply force with damping
            self.centers[i] += force * dt

            # Keep within bounds
            margin = 10
            self.centers[i][0] = np.clip(self.centers[i][0], margin, self.width - margin)
            self.centers[i][1] = np.clip(self.centers[i][1], margin, self.height - margin)

    def _wound_attraction(self, cell_idx: int) -> np.ndarray:
        """Compute attraction force toward nearest wound edge.

        Only cells near the wound edge feel this force.
        Force magnitude decreases with distance from wound.
        """
        if self._wound_mask is None:
            return np.zeros(2)

        px, py = self.centers[cell_idx]

        # Find nearest wound pixel (approximate: check cardinal directions)
        shape, center, size = self.wound_region
        cx, cy = center

        if shape == "rectangle":
            w, h = size
            # Distance to wound edges
            dx = max(0, abs(px - cx) - w / 2)
            dy = max(0, abs(py - cy) - h / 2)
            dist = np.sqrt(dx**2 + dy**2)

            if dist > w:  # too far from wound, no attraction
                return np.zeros(2)

            # Direction toward wound center
            direction = np.array([cx - px, cy - py])
            norm = np.linalg.norm(direction)
            if norm < 1:
                return np.zeros(2)
            direction /= norm

            # Force decreases with distance, zero inside wound
            strength = max(0, 1.0 - dist / (w * 0.5))
            return direction * strength

        elif shape == "circle":
            r = size[0] if isinstance(size, tuple) else size
            dist = np.hypot(px - cx, py - cy)

            if dist > r * 2:  # too far
                return np.zeros(2)

            direction = np.array([cx - px, cy - py])
            norm = np.linalg.norm(direction)
            if norm < 1:
                return np.zeros(2)
            direction /= norm

            strength = max(0, 1.0 - (dist - r) / r)
            return direction * strength

        return np.zeros(2)

    def _neighbor_repulsion(self, cell_idx: int) -> np.ndarray:
        """Soft repulsion from nearby cells to prevent overlap."""
        px, py = self.centers[cell_idx]
        force = np.zeros(2)

        for j in range(self.n_cells):
            if j == cell_idx or not self.alive[j]:
                continue
            dx = px - self.centers[j][0]
            dy = py - self.centers[j][1]
            dist = np.sqrt(dx**2 + dy**2)

            # Repulsion when closer than expected spacing
            expected_spacing = np.sqrt(self.width * self.height / max(1, self.alive.sum()))
            if dist < expected_spacing * 0.8:
                strength = (expected_spacing * 0.8 - dist) / (expected_spacing * 0.8)
                if dist > 0.1:
                    force[0] += (dx / dist) * strength * 0.5
                    force[1] += (dy / dist) * strength * 0.5

        return force

    def _divide(self, dt: float):
        """Probabilistic cell division."""
        new_centers = []
        new_alive = []
        new_has_nuc = []
        new_has_mem = []
        new_nuc_int = []
        new_mem_int = []
        new_parent_idx = []

        for i in range(self.n_cells):
            if not self.alive[i]:
                continue

            if self.rng.random() < self.division_rate * dt:
                # Cell divides: create daughter cell nearby
                angle = self.rng.uniform(0, 2 * np.pi)
                offset = 5.0  # daughter placed 5px away
                new_center = self.centers[i] + np.array([np.cos(angle), np.sin(angle)]) * offset

                # Keep within bounds
                new_center[0] = np.clip(new_center[0], 10, self.width - 10)
                new_center[1] = np.clip(new_center[1], 10, self.height - 10)

                new_centers.append(new_center)
                new_alive.append(True)
                new_has_nuc.append(self.has_nucleus_marker[i])
                new_has_mem.append(self.has_membrane_marker[i])
                new_nuc_int.append(self.nucleus_intensity[i])
                new_mem_int.append(self.membrane_intensity[i])
                new_parent_idx.append(i)

        if new_centers:
            n_new = len(new_centers)
            self.centers = np.vstack([self.centers, np.array(new_centers)])
            self.alive = np.concatenate([self.alive, np.array(new_alive)])
            self.has_nucleus_marker = np.concatenate([self.has_nucleus_marker, np.array(new_has_nuc)])
            self.has_membrane_marker = np.concatenate([self.has_membrane_marker, np.array(new_has_mem)])
            self.nucleus_intensity = np.concatenate([self.nucleus_intensity, np.array(new_nuc_int)])
            self.membrane_intensity = np.concatenate([self.membrane_intensity, np.array(new_mem_int)])
            self.cell_gray = np.concatenate([self.cell_gray,
                self.rng.uniform(100, 155, n_new).astype(np.uint8)])
            self.apoptosis_stage = np.concatenate([self.apoptosis_stage,
                np.zeros(n_new, dtype=int)])
            self._renderable = np.concatenate([self._renderable,
                np.ones(n_new, dtype=bool)])
            # Nucleus morphology for new cells
            self._nuc_offset_frac = np.concatenate([self._nuc_offset_frac,
                self.rng.uniform(0.0, 0.3, n_new)])
            self._nuc_offset_angle = np.concatenate([self._nuc_offset_angle,
                self.rng.uniform(0, 2 * np.pi, n_new)])
            self._nuc_aspect = np.concatenate([self._nuc_aspect,
                self.rng.uniform(1.0, 1.3, n_new)])
            self._nuc_orient = np.concatenate([self._nuc_orient,
                self.rng.uniform(0, np.pi, n_new)])
            # Laser ablation arrays (new daughters start undamaged)
            if getattr(self, '_ablation_enabled', False):
                self._laser_damage = np.concatenate([self._laser_damage,
                    np.zeros(n_new)])
                self._ablated = np.concatenate([self._ablated,
                    np.zeros(n_new, dtype=bool)])
            # Photoconversion: daughters inherit parent's converted state
            if getattr(self, '_photoconversion_enabled', False):
                daughter_conv = np.array([self._converted[pi] for pi in new_parent_idx])
                self._converted = np.concatenate([self._converted, daughter_conv])
                self._conversion_exposure = np.concatenate([
                    self._conversion_exposure, np.zeros(n_new, dtype=int)])
            # Ghost weight for wound healing
            self._ghost_weight = np.concatenate([self._ghost_weight,
                np.ones(n_new)])
            # FUCCI geminin intensity (new daughters start in G1 → 0.05)
            if hasattr(self, '_geminin_intensity'):
                gem_vals = getattr(self, '_fucci_geminin_intensity_values',
                                   {0: 0.05, 1: 0.25, 2: 0.85, 3: 0.95})
                self._geminin_intensity = np.concatenate([
                    self._geminin_intensity,
                    np.full(n_new, gem_vals.get(0, 0.05))])
            self.n_cells += n_new

    def _apoptose(self, dt: float):
        """Probabilistic cell death with visible stages.

        Cells progress through apoptosis stages:
          0 → healthy
          1 → early apoptosis (slight shrinkage, nucleus brightens)
          2 → mid apoptosis (more shrinkage, nucleus condensing)
          3 → late apoptosis (small, bright nucleus, membrane breaking)
          4 → dead/ghost (invisible, holds Voronoi space)

        At each step, healthy cells may initiate apoptosis, and cells
        already dying advance one stage.
        """
        for i in range(self.n_cells):
            if not self.alive[i]:
                continue

            if self.apoptosis_stage[i] > 0:
                # Advance existing apoptosis
                self.apoptosis_stage[i] += 1
                stage = self.apoptosis_stage[i]

                if stage > self.apoptosis_max_stage:
                    # Cell dies (becomes ghost)
                    self.alive[i] = False
                    self._renderable[i] = False
                    self.apoptosis_stage[i] = 0
                else:
                    # Nucleus condenses (brighter)
                    if self.has_nucleus_marker[i]:
                        self.nucleus_intensity[i] = min(1.0, self.nucleus_intensity[i] * 1.15)

            elif self.rng.random() < self.apoptosis_rate * dt:
                # Initiate apoptosis
                self.apoptosis_stage[i] = 1

    def _homeostatic_step(self, dt: float):
        """Feedback-controlled division and apoptosis to maintain target cell count.

        Uses proportional control: when alive count exceeds target, increase
        apoptosis rate; when below target, increase division rate.  Dead (ghost)
        cells are recycled as daughter positions to cap array growth.
        """
        n_alive = int(self.alive.sum())
        error = n_alive - self.target_cells  # positive = too many cells

        # Proportional gain: 0.002 per cell of error
        gain = 0.002
        div_rate = max(0.0, self._homeostatic_base_div - gain * error)
        apo_rate = max(0.0, self._homeostatic_base_apo + gain * error)

        # Division with ghost recycling
        ghost_indices = list(np.where(~self.alive & ~self._renderable)[0])
        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            if self.rng.random() < div_rate * dt:
                angle = self.rng.uniform(0, 2 * np.pi)
                new_center = self.centers[i] + np.array([
                    np.cos(angle), np.sin(angle)
                ]) * 5.0
                new_center[0] = np.clip(new_center[0], 10, self.width - 10)
                new_center[1] = np.clip(new_center[1], 10, self.height - 10)

                if ghost_indices:
                    # Recycle a ghost cell slot
                    gi = ghost_indices.pop()
                    self.centers[gi] = new_center
                    self.alive[gi] = True
                    self._renderable[gi] = True
                    self.has_nucleus_marker[gi] = self.has_nucleus_marker[i]
                    self.has_membrane_marker[gi] = self.has_membrane_marker[i]
                    self.nucleus_intensity[gi] = self.nucleus_intensity[i]
                    self.membrane_intensity[gi] = self.membrane_intensity[i]
                    self.apoptosis_stage[gi] = 0
                    # Photoconversion: daughter inherits parent state
                    if getattr(self, '_photoconversion_enabled', False):
                        self._converted[gi] = self._converted[i]
                        self._conversion_exposure[gi] = 0
                else:
                    # Append new cell (only if no ghosts to recycle)
                    self._divide_append_one(i, new_center)

        # Apoptosis (same as _apoptose but with adjusted rate)
        for i in range(self.n_cells):
            if not self.alive[i]:
                continue
            if self.apoptosis_stage[i] > 0:
                self.apoptosis_stage[i] += 1
                if self.apoptosis_stage[i] > self.apoptosis_max_stage:
                    self.alive[i] = False
                    self._renderable[i] = False
                    self.apoptosis_stage[i] = 0
                else:
                    if self.has_nucleus_marker[i]:
                        self.nucleus_intensity[i] = min(
                            1.0, self.nucleus_intensity[i] * 1.15
                        )
            elif self.rng.random() < apo_rate * dt:
                self.apoptosis_stage[i] = 1

    def _divide_append_one(self, parent_idx: int, new_center: np.ndarray):
        """Append a single daughter cell (used when no ghost slots available)."""
        self.centers = np.vstack([self.centers, new_center.reshape(1, 2)])
        self.alive = np.concatenate([self.alive, [True]])
        self.has_nucleus_marker = np.concatenate([
            self.has_nucleus_marker, [self.has_nucleus_marker[parent_idx]]
        ])
        self.has_membrane_marker = np.concatenate([
            self.has_membrane_marker, [self.has_membrane_marker[parent_idx]]
        ])
        self.nucleus_intensity = np.concatenate([
            self.nucleus_intensity, [self.nucleus_intensity[parent_idx]]
        ])
        self.membrane_intensity = np.concatenate([
            self.membrane_intensity, [self.membrane_intensity[parent_idx]]
        ])
        self.cell_gray = np.concatenate([
            self.cell_gray, self.rng.uniform(100, 155, 1).astype(np.uint8)
        ])
        self.apoptosis_stage = np.concatenate([self.apoptosis_stage, [0]])
        self._renderable = np.concatenate([self._renderable, [True]])
        # Nucleus morphology for new cell
        self._nuc_offset_frac = np.concatenate([
            self._nuc_offset_frac, self.rng.uniform(0.0, 0.3, 1)])
        self._nuc_offset_angle = np.concatenate([
            self._nuc_offset_angle, self.rng.uniform(0, 2 * np.pi, 1)])
        self._nuc_aspect = np.concatenate([
            self._nuc_aspect, self.rng.uniform(1.0, 1.3, 1)])
        self._nuc_orient = np.concatenate([
            self._nuc_orient, self.rng.uniform(0, np.pi, 1)])
        # Laser ablation arrays (new daughter starts undamaged)
        if getattr(self, '_ablation_enabled', False):
            self._laser_damage = np.concatenate([self._laser_damage, [0.0]])
            self._ablated = np.concatenate([self._ablated, [False]])
        # Photoconversion: daughter inherits parent's converted state
        if getattr(self, '_photoconversion_enabled', False):
            self._converted = np.concatenate([
                self._converted, [self._converted[parent_idx]]])
            self._conversion_exposure = np.concatenate([
                self._conversion_exposure, [0]])
        # Ghost weight
        self._ghost_weight = np.concatenate([self._ghost_weight, [1.0]])
        self.n_cells += 1
        # Extension hooks (FUCCI geminin propagation, etc.)
        for hook in self._divide_hooks:
            hook(parent_idx)

    # ---- Tissue recomputation ----

    def _recompute_tissue(self):
        """Recompute Voronoi tessellation after cell movement.

        If ``exclude_ghosts_from_voronoi`` is True, ghost cells are
        temporarily moved far off-screen so alive cells' Voronoi polygons
        expand into the wound gap — producing visible wound closure.
        Otherwise ghost cells hold their space (preserving a static wound).
        """
        ghost_saved = None
        if self.exclude_ghosts_from_voronoi:
            ghost_idx = np.where(~self.alive)[0]
            if len(ghost_idx) > 0:
                ghost_saved = (ghost_idx, self.centers[ghost_idx].copy())
                # Scatter ghosts far outside the visible area
                for k, gi in enumerate(ghost_idx):
                    self.centers[gi] = [-10000 - k * 100, -10000 - k * 100]

        self._compute_voronoi()
        self._compute_cell_properties()
        self._compute_ruffled_polygons()

        # Restore ghost positions (needed for colonization distance calc)
        if ghost_saved is not None:
            idx, pos = ghost_saved
            self.centers[idx] = pos

        # Apply apoptosis shrinkage to nucleus radii (after recomputation)
        dying = self.apoptosis_stage > 0
        if dying.any():
            for i in np.where(dying)[0]:
                stage = min(self.apoptosis_stage[i], len(self._apoptosis_shrink) - 1)
                self.nucleus_radii[i] *= self._apoptosis_shrink[stage]
                self.nucleus_radii[i] = max(3, self.nucleus_radii[i])

        # Invalidate render cache (lazy re-rendering)
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None

    # ---- Template method overrides for SimBase pipeline ----

    def _handle_mask(self, mask: np.ndarray) -> None:
        """Map SLM mask to world coordinates for optogenetic stimulation.

        Illuminated cells migrate faster during subsequent step() calls.
        Also drives gene induction, photoconversion, and laser ablation.
        """
        if np.any(mask):
            self._stim_mask = self._map_slm_to_world(mask)
        else:
            self._stim_mask = None

    def _auto_step_tick(self):
        """No-op: DynamicVoronoiSim steps AFTER rendering, not before."""

    def _render_for_mode(self, mode):
        """Lazy re-render tissue on cache miss, then delegate to VoronoiSim."""
        if self._bf_full is None:
            self._render_full_tissue()
            # Re-render dynamic extra channels (translocation GFP)
            if self._transloc_enabled and self._transloc_mode_id is not None:
                gfp_img = self._render_translocation_full()
                self._extra_channels[self._transloc_mode_id]["image"] = gfp_img
            # Re-render extension hook channels (FUCCI, lysosomes, etc.)
            for mode_id, render_fn in self._render_hooks.items():
                self._extra_channels[mode_id]["image"] = render_fn()
        return super()._render_for_mode(mode)

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs):
        """Thin wrapper: run full template pipeline, then post-render step.

        DynamicVoronoiSim advances dynamics AFTER rendering so that
        multi-channel acquisitions (e.g. BF + membrane) within one
        timepoint see the same tissue state.
        """
        result = super().snap_frame(
            mask=mask, exposure=exposure, intensity=intensity, **kwargs,
        )

        # Auto-advance dynamics after N snaps (post-render)
        if self.auto_step:
            self._snap_counter += 1
            if self._snap_counter >= self.snaps_per_step:
                self._snap_counter = 0
                self.step(dt=self.auto_step_dt)

        return result

    # ---- Tracking ----

    def get_tracking_snapshot(self):
        """Capture current cell state for tracking ground truth.

        Cell indices are stable tracking IDs (same cell keeps same index
        unless new cells are added via division at the end).

        Returns:
            dict with frame_count and per-cell state.
        """
        cells = []
        for i in range(self.n_cells):
            cells.append({
                "id": i,
                "x": round(float(self.centers[i][0]), 1),
                "y": round(float(self.centers[i][1]), 1),
                "alive": bool(self.alive[i]),
                "visible": bool(self._renderable[i]),
                "apoptosis_stage": int(self.apoptosis_stage[i]),
            })
        return {
            "frame": self.frame_count,
            "n_alive": int(self.alive.sum()),
            "n_visible": int(self._renderable.sum()),
            "cells": cells,
        }

    # ---- Metrics ----

    def get_wound_area(self) -> float:
        """Estimate current wound area (pixels without cells)."""
        if self._wound_mask is None:
            return 0.0

        # Render membrane to find cell-covered area in wound region
        if self._mem_full is None:
            self._render_full_tissue()

        # Count wound pixels not covered by membrane
        mem_gray = self._mem_full[:, :, 0] if self._mem_full.ndim == 3 else self._mem_full
        # Scale wound mask to match internal resolution if needed
        s = getattr(self, 'internal_scale', 1)
        if s > 1 and self._wound_mask.shape != mem_gray.shape:
            wound_mask = cv2.resize(
                self._wound_mask.astype(np.uint8),
                (mem_gray.shape[1], mem_gray.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            wound_mask = self._wound_mask
        wound_pixels = wound_mask.sum()
        covered = (mem_gray[wound_mask] > 20).sum()
        uncovered = wound_pixels - covered
        # Normalize back to world-resolution pixel count
        return float(max(0, uncovered / (s * s)))

    def get_migration_front(self) -> dict:
        """Get position of the migration front (nearest cells to wound center)."""
        if self.wound_region is None:
            return {}

        _, center, _ = self.wound_region
        cx, cy = center

        # Find the closest alive cell to wound center
        dists = [np.hypot(self.centers[i][0] - cx, self.centers[i][1] - cy)
                 for i in range(self.n_cells) if self.alive[i]]

        if not dists:
            return {"min_dist": float("inf"), "mean_dist": float("inf")}

        return {
            "min_dist": round(float(min(dists)), 1),
            "mean_dist": round(float(np.mean(sorted(dists)[:5])), 1),  # mean of 5 closest
            "n_alive": int(self.alive.sum()),
        }

    def get_wound_gap_width(self) -> float:
        """Measure mean wound gap width from alive-cell positions.

        For a vertical wound, computes the mean horizontal gap between
        the rightmost left-side cells and leftmost right-side cells,
        sampled at multiple y-levels.
        """
        if self.wound_region is None:
            return 0.0

        _, center, size = self.wound_region
        cx, cy = center

        alive_centers = self.centers[self.alive]
        if len(alive_centers) == 0:
            return float(size[0]) if isinstance(size, tuple) else 0.0

        # Split cells into left and right of wound center
        left = alive_centers[alive_centers[:, 0] < cx]
        right = alive_centers[alive_centers[:, 0] >= cx]

        if len(left) == 0 or len(right) == 0:
            return float(size[0]) if isinstance(size, tuple) else 0.0

        # Sample at multiple y-levels: find gap width at each
        if isinstance(size, tuple) and len(size) >= 2:
            y_min = max(30, int(cy - size[1] / 3))
            y_max = min(self.height - 30, int(cy + size[1] / 3))
        elif isinstance(size, tuple) and len(size) == 1:
            # Circle wound: use radius for y-range
            r = size[0]
            y_min = max(30, int(cy - r))
            y_max = min(self.height - 30, int(cy + r))
        else:
            y_min, y_max = 30, self.height - 30
        y_samples = np.linspace(y_min, y_max, 10)

        gaps = []
        for y in y_samples:
            # Cells within ±30 px of this y-level
            band = 40.0
            left_band = left[np.abs(left[:, 1] - y) < band]
            right_band = right[np.abs(right[:, 1] - y) < band]

            if len(left_band) > 0 and len(right_band) > 0:
                left_edge = left_band[:, 0].max()
                right_edge = right_band[:, 0].min()
                gap = max(0, right_edge - left_edge)
                gaps.append(gap)

        if not gaps:
            return float(size[0]) if isinstance(size, tuple) else 0.0

        return round(float(np.mean(gaps)), 1)
