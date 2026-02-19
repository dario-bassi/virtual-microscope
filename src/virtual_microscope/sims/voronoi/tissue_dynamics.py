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

    sim = DynamicVoronoiSim(width=1024, height=1024, nb_cells=200, ...)
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
        self.alive = np.ones(self.nb_cells, dtype=bool)
        self.velocities = np.zeros((self.nb_cells, 2))
        self.frame_count = 0

        # Apoptosis state: tracks dying cells through visible morphological stages
        # 0 = healthy, 1+ = apoptosis stage (higher = further along)
        self.apoptosis_stage = np.zeros(self.nb_cells, dtype=int)
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
        self._gene_expression = np.zeros(self.nb_cells, dtype=float)  # per-cell
        self._gene_baseline = None         # baseline intensities before induction
        self._cumulative_illumination = np.zeros(self.nb_cells, dtype=float)

        # Ghost weight for wound healing: each ghost starts at weight 1.0
        # (full cell-sized gap). Weight decays over time, displacing the
        # ghost centroid toward its nearest alive neighbor. When weight → 0,
        # the ghost is removed entirely and the alive neighbor's Voronoi
        # polygon expands naturally. No new tiny cells are created.
        self._ghost_weight = np.ones(self.nb_cells, dtype=float)

        self.exclude_ghosts_from_voronoi = False

        # Homeostatic mode: balanced division + apoptosis to maintain
        # a target cell count indefinitely. When enabled, the tissue
        # sustains interesting dynamics without growing or shrinking.
        self.homeostatic = False
        self.target_cells = self.nb_cells  # target = initial count
        self._homeostatic_base_div = 0.005  # base division rate when homeostatic
        self._homeostatic_base_apo = 0.005  # base apoptosis rate when homeostatic

        # Nuclear translocation: drug-induced protein shuttling (e.g. NF-kB)
        self._transloc_enabled = False
        self._transloc_nuc_fraction = np.full(self.nb_cells, 0.2)  # 20% nuclear at rest
        self._transloc_import_rate = 0.08   # nuclear import rate under drug
        self._transloc_export_rate = 0.03   # nuclear export rate (baseline re-export)
        self._transloc_baseline = 0.2       # resting nuclear fraction
        self._transloc_drug_active = False   # set by Perfusion device or manual flag
        self._transloc_total_reporter = np.ones(self.nb_cells)  # total GFP per cell
        self._transloc_mode_id = None       # mode_id of extra GFP channel

        # Z-drift: tissue slowly moves out of focus during timelapse
        # Simulates thermal drift, mechanical relaxation, etc.
        self.z_drift_rate = 0.0    # µm/s (positive = tissue drifts up)
        self.z_drift_noise = 0.0   # σ of z-jitter (µm·s⁻½, Brownian)
        self._initial_tissue_z = self.tissue_z  # store for drift measurement

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
            for i in range(self.nb_cells):
                if not self.alive[i]:
                    continue
                val = self._nuc_dynamics(i, self.frame_count)
                if val is not None:
                    self.nucleus_intensity[i] = np.clip(val, 0.0, 1.0)

        if self._mem_dynamics is not None:
            for i in range(self.nb_cells):
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
        self._gene_expression = np.zeros(self.nb_cells, dtype=float)
        self._cumulative_illumination = np.zeros(self.nb_cells, dtype=float)

    def _apply_gene_induction(self):
        """Update per-cell gene expression based on SLM illumination."""
        if not self._gene_induction_enabled:
            return

        for i in range(self.nb_cells):
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
        expr = self._gene_expression[:self.nb_cells]

        illuminated = []
        if self._stim_mask is not None:
            for i in range(self.nb_cells):
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
            "cumulative_illumination": self._cumulative_illumination[:self.nb_cells].copy(),
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
        self._converted = np.zeros(self.nb_cells, dtype=bool)
        self._conversion_exposure = np.zeros(self.nb_cells, dtype=int)

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
        for i in range(self.nb_cells):
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
        for i in range(self.nb_cells):
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

        alive_mask = self.alive[:self.nb_cells]
        conv = self._converted[:self.nb_cells]
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
        self._laser_damage = np.zeros(self.nb_cells, dtype=float)
        self._ablated = np.zeros(self.nb_cells, dtype=bool)
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
        for i in range(self.nb_cells):
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
            ablated_idx = np.where(self._ablated[:self.nb_cells])[0]
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

        ablated = self._ablated[:self.nb_cells]
        damage = self._laser_damage[:self.nb_cells]
        alive_mask = self.alive[:self.nb_cells]

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
        self._transloc_nuc_fraction = np.full(self.nb_cells, baseline_nuc_fraction)
        # Per-cell total reporter with heterogeneity (some cells load more dye)
        self._transloc_total_reporter = np.clip(
            1.0 + self.rng.normal(0, heterogeneity, self.nb_cells), 0.3, 1.7
        )

    def enable_translocation_channel(self, core=None):
        """Register GFP reporter as extra channel for translocation imaging.

        Call after enable_translocation(). Renders initial GFP distribution
        and registers it as 'gfp-channel' in the channel system.

        Args:
            core: Optional CMMCore to register Fake config group entry.

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
            core.defineConfig("Fake", "gfp-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "gfp-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")
        return mode_id

    def set_translocation_drug(self, active: bool):
        """Set whether translocation-inducing drug is active."""
        self._transloc_drug_active = active

    def _apply_translocation(self, dt: float = 1.0):
        """Update nuclear fraction based on drug state."""
        if not self._transloc_enabled:
            return
        n = self.nb_cells
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
        n = self.nb_cells
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

    # ---- FUCCI cell cycle reporter ----

    def enable_fucci_reporter(self, g1_duration: int = 8, s_duration: int = 5,
                               g2_duration: int = 4, m_duration: int = 2,
                               division_on_m: bool = True, core=None):
        """Enable FUCCI (Fluorescent Ubiquitination Cell Cycle Indicator) reporter.

        Assigns each cell a cell cycle phase (G1/S/G2/M) and tracks progression.
        Cell phase maps to two-channel fluorescence:
          - Nucleus channel (mode 1) = mCherry-Cdt1: bright in G1, dim in S/G2/M
          - Membrane channel (mode 2) = mVenus-Geminin: dim in G1, bright in S/G2/M
          - Early S phase: both moderate (overlap → yellow in merged view)

        Real FUCCI: Cdt1 (G1 marker) degrades at S-phase onset,
        Geminin (S/G2/M marker) degrades at M-phase exit.

        Args:
            g1_duration: Steps in G1 phase (default 8).
            s_duration: Steps in S phase (default 5).
            g2_duration: Steps in G2 phase (default 4).
            m_duration: Steps in M phase (default 2).
            division_on_m: If True, cells divide at end of M phase.
        """
        self._fucci_enabled = True
        self._fucci_division_on_m = division_on_m

        # Phase durations (in simulation steps)
        self._fucci_durations = {
            0: g1_duration,   # G1
            1: s_duration,    # S
            2: g2_duration,   # G2
            3: m_duration,    # M
        }
        total_cycle = g1_duration + s_duration + g2_duration + m_duration

        # Phase names for ground truth
        self._fucci_phase_names = {0: "G1", 1: "S", 2: "G2", 3: "M"}

        # Per-cell phase state
        self._fucci_phase = np.zeros(self.nb_cells, dtype=int)  # 0=G1,1=S,2=G2,3=M
        self._fucci_timer = np.zeros(self.nb_cells, dtype=float)  # time in current phase

        # Randomize starting phase so cells aren't synchronized
        for i in range(self.nb_cells):
            if not self.alive[i]:
                continue
            t = self.rng.integers(0, total_cycle)
            if t < g1_duration:
                self._fucci_phase[i] = 0
                self._fucci_timer[i] = t
            elif t < g1_duration + s_duration:
                self._fucci_phase[i] = 1
                self._fucci_timer[i] = t - g1_duration
            elif t < g1_duration + s_duration + g2_duration:
                self._fucci_phase[i] = 2
                self._fucci_timer[i] = t - g1_duration - s_duration
            else:
                self._fucci_phase[i] = 3
                self._fucci_timer[i] = t - g1_duration - s_duration - g2_duration

        # FUCCI intensity maps (phase → channel intensity on 0-1 scale)
        # mCherry-Cdt1 (nucleus channel): bright in G1, decays in S, off in G2/M
        # mVenus-Geminin (membrane channel): off in G1, rises in S, bright in G2/M
        self._fucci_nuc_intensity = {
            0: 0.85,   # G1: mCherry-Cdt1 bright
            1: 0.45,   # S: Cdt1 being degraded (overlap period)
            2: 0.08,   # G2: Cdt1 fully degraded
            3: 0.05,   # M: Cdt1 gone
        }
        self._fucci_mem_intensity = {
            0: 0.05,   # G1: Geminin absent
            1: 0.40,   # S: Geminin accumulating (overlap period)
            2: 0.80,   # G2: Geminin bright
            3: 0.90,   # M: Geminin very bright (peak before degradation)
        }

        # Geminin-GFP extra channel (registers 'geminin-channel')
        self._geminin_mode_id = None
        if core is not None:
            geminin_img = self._render_fucci_geminin_full()
            mode_id = self.add_channel("geminin", geminin_img,
                                       filter_label="TagGFP2(483/506)",
                                       led_label="GREEN")
            self._geminin_mode_id = mode_id
            core.defineConfig("Fake", "geminin-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "geminin-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")

        # Dedicated geminin intensity array (step function, no smooth interp)
        # Avoids S/G2 boundary confusion caused by smooth membrane_intensity
        # G1=0.05, S=0.25, G2=0.85, M=0.95 — clear gap at threshold ~0.65
        self._fucci_geminin_intensity_values = {0: 0.05, 1: 0.25, 2: 0.85, 3: 0.95}
        self._geminin_intensity = np.full(self.nb_cells, 0.05, dtype=float)

        # Apply initial intensities
        self._apply_fucci_intensities()

    def _render_fucci_geminin_full(self) -> np.ndarray:
        """Render Geminin-GFP channel: nuclear signal, bright in G2/M.

        G1 cells: very dim (Geminin absent, membrane_intensity ~0.05)
        S  cells: moderate (Geminin accumulating, ~0.40)
        G2 cells: bright (Geminin peak, ~0.80)
        M  cells: very bright (Geminin peak, ~0.90; degrades post-mitosis)

        Renders as circular nuclear blobs (like nucleus-channel but for Geminin).
        """
        from scipy.ndimage import gaussian_filter
        s = self.internal_scale
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)

        n_cells = len(self.alive)
        nuc_radii = getattr(self, 'nucleus_radii', None)
        for i in range(n_cells):
            if not self.alive[i]:
                continue
            cx_i = int(round(float(self.centers[i][0]) * s))
            cy_i = int(round(float(self.centers[i][1]) * s))
            nuc_r = float(nuc_radii[i]) if (nuc_radii is not None and i < len(nuc_radii)) else 15.0
            nr_i = max(2, int(round(nuc_r * s)))
            # Geminin brightness: use dedicated step-function intensity (no smooth interp)
            # Fallback to membrane_intensity if _geminin_intensity not available
            gem_arr = getattr(self, '_geminin_intensity', None)
            if gem_arr is not None and i < len(gem_arr):
                mem_int = float(gem_arr[i])
            else:
                mem_int = float(self.membrane_intensity[i]) if i < len(self.membrane_intensity) else 0.05
            brightness = mem_int * 220.0  # slightly brighter than before (was 210)
            if 0 <= cx_i < self._iw and 0 <= cy_i < self._ih and nr_i > 0:
                cv2.circle(buf, (cx_i, cy_i), nr_i, brightness, -1)

        # PSF blur (slightly more diffuse than nucleus-channel)
        sigma = max(0.8 * s, 1.5)
        if buf.max() > 0:
            buf = gaussian_filter(buf, sigma=sigma)

        img_g = np.clip(buf, 0, 255).astype(np.uint8)
        return cv2.merge([img_g, img_g, img_g])

    def _apply_fucci_intensities(self):
        """Update nucleus/membrane intensities based on current FUCCI phase."""
        if not getattr(self, '_fucci_enabled', False):
            return

        arrest_phase = getattr(self, '_fucci_arrest_phase', None)

        for i in range(self.nb_cells):
            if not self.alive[i]:
                continue
            phase = self._fucci_phase[i]
            timer = self._fucci_timer[i]
            duration = self._fucci_durations[phase]

            # Smooth transitions: interpolate toward next phase intensity
            next_phase = (phase + 1) % 4
            progress = timer / max(1, duration)  # 0→1 within phase

            # Arrested cells: show pure arrest-phase intensity
            # (don't interpolate toward next phase).
            if (arrest_phase is not None and phase == arrest_phase
                    and timer >= duration):
                progress = 0.0

            # Interpolate from current to next phase (smooth ramp)
            nuc_curr = self._fucci_nuc_intensity[phase]
            nuc_next = self._fucci_nuc_intensity[next_phase]
            self.nucleus_intensity[i] = nuc_curr + (nuc_next - nuc_curr) * progress

            mem_curr = self._fucci_mem_intensity[phase]
            mem_next = self._fucci_mem_intensity[next_phase]
            self.membrane_intensity[i] = mem_curr + (mem_next - mem_curr) * progress

            # Geminin channel: step function (no interpolation) for clear S/G2 separation
            # G1=0.05, S=0.25, G2=0.85, M=0.95 — gap at ~0.65 separates S from G2/M
            if hasattr(self, '_geminin_intensity') and i < len(self._geminin_intensity):
                gem_vals = getattr(self, '_fucci_geminin_intensity_values',
                                   {0: 0.05, 1: 0.25, 2: 0.85, 3: 0.95})
                self._geminin_intensity[i] = gem_vals.get(phase, 0.05)

    def _advance_fucci(self, dt: float):
        """Advance cell cycle phases by dt time units."""
        if not getattr(self, '_fucci_enabled', False):
            return

        # Track drug arrest delay (accumulated time, not step count)
        arrest_phase = getattr(self, '_fucci_arrest_phase', None)
        if arrest_phase is not None:
            delay = getattr(self, '_fucci_arrest_delay', 0)
            elapsed = getattr(self, '_fucci_arrest_step', 0)
            self._fucci_arrest_step = elapsed + dt
            drug_active = (elapsed >= delay)
        else:
            drug_active = False

        divisions = []  # (parent_idx,) for cells that complete M phase

        for i in range(self.nb_cells):
            if not self.alive[i]:
                continue
            if self.apoptosis_stage[i] > 0:
                continue  # dying cells don't progress

            self._fucci_timer[i] += dt
            phase = self._fucci_phase[i]
            duration = self._fucci_durations[phase]

            if self._fucci_timer[i] >= duration:
                # Check drug arrest: block transition FROM arrested phase
                if drug_active and phase == arrest_phase:
                    # Cell is stuck — timer stays at max (arrested)
                    self._fucci_timer[i] = duration
                    continue

                # Phase transition
                self._fucci_timer[i] = 0

                if phase == 3:  # M → G1 (completed mitosis)
                    self._fucci_phase[i] = 0
                    if self._fucci_division_on_m:
                        divisions.append(i)
                else:
                    self._fucci_phase[i] = phase + 1

        # Handle divisions from M-phase completion
        for parent in divisions:
            angle = self.rng.uniform(0, 2 * np.pi)
            offset = 5.0
            new_center = self.centers[parent] + np.array([
                np.cos(angle), np.sin(angle)
            ]) * offset
            new_center[0] = np.clip(new_center[0], 10, self.width - 10)
            new_center[1] = np.clip(new_center[1], 10, self.height - 10)

            # Try to recycle a ghost cell slot
            ghost_indices = np.where(~self.alive & ~self._renderable)[0]
            if len(ghost_indices) > 0:
                gi = ghost_indices[0]
                self.centers[gi] = new_center
                self.alive[gi] = True
                self._renderable[gi] = True
                self.has_nucleus_marker[gi] = self.has_nucleus_marker[parent]
                self.has_membrane_marker[gi] = self.has_membrane_marker[parent]
                self.apoptosis_stage[gi] = 0
                # Daughter starts in G1
                self._fucci_phase[gi] = 0
                self._fucci_timer[gi] = 0
            else:
                self._divide_append_one(parent, new_center)
                # Extend FUCCI arrays for new cell
                self._fucci_phase = np.concatenate([self._fucci_phase, [0]])
                self._fucci_timer = np.concatenate([self._fucci_timer, [0]])

        # Update intensities after phase changes
        self._apply_fucci_intensities()
        # Update Geminin-GFP channel image
        if hasattr(self, '_geminin_mode_id') and self._geminin_mode_id is not None:
            gem_img = self._render_fucci_geminin_full()
            if hasattr(self, '_extra_channels') and self._geminin_mode_id in self._extra_channels:
                self._extra_channels[self._geminin_mode_id]['image'] = gem_img

    def get_fucci_state(self):
        """Return per-cell FUCCI reporter state for ground truth.

        Returns:
            dict with:
                phases: array of phase indices (0=G1, 1=S, 2=G2, 3=M)
                phase_names: array of phase name strings
                phase_counts: dict mapping phase name → count of alive cells
                timers: array of steps spent in current phase
                mean_nuc_intensity: mean nucleus intensity of alive cells
                mean_mem_intensity: mean membrane intensity of alive cells
        """
        if not getattr(self, '_fucci_enabled', False):
            return {"error": "FUCCI not enabled"}

        alive_mask = self.alive[:self.nb_cells]
        phases = self._fucci_phase[:self.nb_cells]
        timers = self._fucci_timer[:self.nb_cells]

        phase_counts = {}
        for phase_idx, name in self._fucci_phase_names.items():
            phase_counts[name] = int(((phases == phase_idx) & alive_mask).sum())

        phase_names = np.array([self._fucci_phase_names[p] for p in phases])

        return {
            "phases": phases.copy(),
            "phase_names": phase_names,
            "phase_counts": phase_counts,
            "timers": timers.copy(),
            "mean_nuc_intensity": float(self.nucleus_intensity[alive_mask].mean())
                if alive_mask.any() else 0.0,
            "mean_mem_intensity": float(self.membrane_intensity[alive_mask].mean())
                if alive_mask.any() else 0.0,
            "n_alive": int(alive_mask.sum()),
        }

    def set_drug_arrest(self, arrest_phase: int, arrest_delay: int = 0):
        """Block cell cycle progression at a specific phase.

        Simulates a drug that arrests cells at a checkpoint:
          - arrest_phase=0: G1 arrest (e.g. CDK4/6 inhibitor like palbociclib)
          - arrest_phase=1: S-phase arrest (e.g. hydroxyurea, aphidicolin)
          - arrest_phase=2: G2 arrest (e.g. CDK1 inhibitor)
          - arrest_phase=3: M arrest (e.g. nocodazole, taxol)

        Cells already past the arrest point continue normally until they
        cycle back to the arrested phase.

        Args:
            arrest_phase: Phase index (0=G1, 1=S, 2=G2, 3=M) to block at.
            arrest_delay: Number of steps before drug takes effect (default 0).
        """
        if not getattr(self, '_fucci_enabled', False):
            raise RuntimeError("Must enable FUCCI reporter before setting drug arrest")
        self._fucci_arrest_phase = arrest_phase
        self._fucci_arrest_delay = arrest_delay
        self._fucci_arrest_step = 0.0  # accumulated time since drug applied

    def clear_drug_arrest(self):
        """Remove drug arrest, allowing cells to resume cycling."""
        self._fucci_arrest_phase = None
        self._fucci_arrest_delay = 0

    # ---- Z-drift helpers ----

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift since initialization (µm)."""
        return self.tissue_z - self._initial_tissue_z

    def reset_z_drift(self):
        """Reset tissue Z to initial position."""
        self.tissue_z = self._initial_tissue_z

    # ---- Lysosome simulation ----

    def enable_lysosomes(
        self,
        core=None,
        n_min: int = 5,
        n_max: int = 20,
        radius_min: float = 1.2,
        radius_max: float = 2.5,
        intensity_mean: float = 200.0,
        intensity_std: float = 30.0,
        diffusion_rate: float = 0.5,
    ):
        """Add lysosomal puncta to each cell (LysoTracker-like stain).

        Creates bright puncta in each cell, distributed throughout the cytoplasm.
        Optionally registers a 'lysotracker-channel' hardware config.

        Args:
            core: Optional UniMMCore — if given, registers 'lysotracker-channel'.
            n_min/n_max: Per-cell lysosome count range (uniform random).
            radius_min/max: Lysosome radius in world px (internal scale applied).
            intensity_mean/std: Brightness distribution (0–255 clipped).
            diffusion_rate: Random walk step size per sim step (world px/step).
                0 = static. Used in step() to drift lysosomes slowly.

        Returns:
            mode_id for the lysotracker channel.
        """
        rng = self.rng
        s = self.internal_scale

        # For each cell, generate lysosomes in cytoplasm (exclude nucleus region)
        self._lyso_cells = []  # List[List[dict]] — per cell, per lysosome
        renderable = getattr(self, '_renderable', None)

        for i, center in enumerate(self.centers[:self.nb_cells]):
            if renderable is not None and not renderable[i]:
                self._lyso_cells.append([])
                continue
            cx, cy = float(center[0]), float(center[1])
            n_lyso = int(rng.integers(n_min, n_max + 1))
            # Use actual cell area (from VoronoiSim) for effective radius
            if hasattr(self, 'cell_areas'):
                cell_r = float(np.sqrt(self.cell_areas[i] / np.pi))
            else:
                cell_r = 30.0  # fallback
            # Nucleus exclusion: use nucleus_radii if available
            nuc_r = float(self.nucleus_radii[i]) if hasattr(self, 'nucleus_radii') else cell_r * 0.3

            lysosomes = []
            attempts = 0
            while len(lysosomes) < n_lyso and attempts < n_lyso * 20:
                attempts += 1
                # Random position within cell polygon (approximate with circle)
                angle = float(rng.uniform(0, 2 * np.pi))
                r = float(rng.uniform(nuc_r * 1.1, cell_r * 0.85))
                lx = cx + r * np.cos(angle)
                ly = cy + r * np.sin(angle)
                # Clip to world
                lx = float(np.clip(lx, 2, self.width - 2))
                ly = float(np.clip(ly, 2, self.height - 2))
                radius = float(rng.uniform(radius_min, radius_max))
                intensity = float(np.clip(rng.normal(intensity_mean, intensity_std), 50, 255))
                lysosomes.append({
                    "x": lx, "y": ly, "r": radius,
                    "intensity": intensity,
                    "cell_idx": i,
                })
            self._lyso_cells.append(lysosomes)

        self._lyso_diffusion_rate = float(diffusion_rate)
        # Gaussian PSF sigma in internal pixels.
        # Target: ~1 µm FWHM (diffraction-limited) → sigma ≈ 0.4 µm.
        # At 40x (pixel_size=0.25 µm/px), sigma_output ≈ 1.6 px.
        # At 40x internal (1:1): sigma_internal = sigma_output = 1.6 * s
        # Using sigma = 0.4 * s gives ~1.6 internal px at s=4 → ~1-2 px at 40x output.
        # This makes puncta appear as tight diffraction-limited spots, not large blobs.
        self._lyso_sigma_internal = max(1.0, 0.4 * s)

        # Render lysotracker channel image
        lyso_img = self._render_lysosomes_full()
        mode_id = self.add_channel(
            "lysotracker", lyso_img,
            filter_label="TagGFP2(483/506)", led_label="GREEN",
        )
        self._lyso_mode_id = mode_id

        if core is not None:
            core.defineConfig("Fake", "lysotracker-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "lysotracker-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")
        return mode_id

    def _render_lysosomes_full(self) -> np.ndarray:
        """Render lysosomal puncta at internal resolution.

        Uses Gaussian PSF rendering for realistic diffraction-limited spots.
        Each lysosome is a bright point convolved with a Gaussian, making them
        detectable at lower magnifications (20x) while still precise at 40x.
        """
        from scipy.ndimage import gaussian_filter
        s = self.internal_scale
        # Float accumulation buffer for Gaussian rendering
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)
        if not hasattr(self, '_lyso_cells'):
            return np.zeros((self._ih, self._iw, 3), dtype=np.uint8)

        for cell_lysosomes in self._lyso_cells:
            for lyso in cell_lysosomes:
                lx = int(round(lyso["x"] * s))
                ly = int(round(lyso["y"] * s))
                # Clamp to buffer bounds
                if not (0 <= lx < self._iw and 0 <= ly < self._ih):
                    continue
                intensity = float(lyso["intensity"])
                # Point source at (lx, ly) — Gaussian spread comes from filter below
                buf[ly, lx] += intensity

        # Gaussian PSF: sigma = lysosome_radius * internal_scale
        # Default radius ~1.5 world px → sigma ~6 internal px → ~1.5 output px at 10x
        # At 20x (2x magnification): appears as ~3 output px sigma — clearly visible
        lyso_sigma = float(getattr(self, '_lyso_sigma_internal', 3.5 * s))
        if buf.max() > 0:
            buf = gaussian_filter(buf, sigma=lyso_sigma)
            # Normalize to preserve peak intensity
            buf = buf / max(buf.max(), 1e-6) * np.max([l["intensity"]
                for cl in self._lyso_cells for l in cl] or [200.0])

        img_g = np.clip(buf, 0, 255).astype(np.uint8)
        return cv2.merge([img_g, img_g, img_g])

    def _drift_lysosomes(self, dt: float):
        """Slowly diffuse lysosomes within their parent cells."""
        rate = self._lyso_diffusion_rate * np.sqrt(dt)
        for i, cell_lysosomes in enumerate(self._lyso_cells):
            if not cell_lysosomes:
                continue
            cx = float(self.centers[i][0])
            cy = float(self.centers[i][1])
            if hasattr(self, 'cell_areas'):
                cell_r = float(np.sqrt(self.cell_areas[i] / np.pi))
            else:
                cell_r = 30.0
            nuc_r = float(self.nucleus_radii[i]) if hasattr(self, 'nucleus_radii') else cell_r * 0.3
            for lyso in cell_lysosomes:
                # Random walk
                dx = float(self.rng.normal(0, rate))
                dy = float(self.rng.normal(0, rate))
                nx = lyso["x"] + dx
                ny = lyso["y"] + dy
                # Clamp to cell area (simple circular approximation)
                dist = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                if dist > cell_r * 0.85:
                    # Bounce back
                    scale = cell_r * 0.85 / dist
                    nx = cx + (nx - cx) * scale
                    ny = cy + (ny - cy) * scale
                # Nucleus exclusion
                dist_nuc = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                if dist_nuc < nuc_r * 1.1:
                    scale = nuc_r * 1.1 / max(dist_nuc, 0.1)
                    nx = cx + (nx - cx) * scale
                    ny = cy + (ny - cy) * scale
                lyso["x"] = float(np.clip(nx, 1, self.width - 1))
                lyso["y"] = float(np.clip(ny, 1, self.height - 1))

    def get_lysosome_stats(self) -> dict:
        """Return ground truth lysosome statistics per cell."""
        if not hasattr(self, '_lyso_cells'):
            return {"enabled": False}
        per_cell = []
        for i, lysosomes in enumerate(self._lyso_cells):
            per_cell.append({
                "cell_id": i,
                "n_lysosomes": len(lysosomes),
                "mean_intensity": round(
                    float(np.mean([l["intensity"] for l in lysosomes]))
                    if lysosomes else 0.0, 1),
                "positions_world": [
                    [round(l["x"], 1), round(l["y"], 1)] for l in lysosomes
                ],
            })
        n_cells = len([c for c in self._lyso_cells if c])
        all_counts = [len(c) for c in self._lyso_cells]
        return {
            "enabled": True,
            "n_cells": n_cells,
            "total_lysosomes": sum(all_counts),
            "mean_per_cell": round(float(np.mean(all_counts)), 1),
            "std_per_cell": round(float(np.std(all_counts)), 1),
            "per_cell": per_cell,
        }

    # ---- Lipid Droplet Staining ----

    def enable_lipid_droplets(
        self,
        core=None,
        normal_n_range: tuple = (1, 5),
        steatotic_n_range: tuple = (15, 40),
        steatotic_fraction: float = 0.0,
        radius_range: tuple = (3.0, 7.0),
        intensity_mean: float = 210.0,
        intensity_std: float = 25.0,
    ):
        """Add lipid droplet puncta to each cell (BODIPY 493/503-like stain).

        Lipid droplets are cytoplasmic lipid-rich organelles, 1–5 µm diameter in
        hepatocytes.  Steatotic cells accumulate many large droplets.

        Args:
            core: Optional UniMMCore — registers 'bodipy-channel' if provided.
            normal_n_range: (min, max) LD count for normal cells.
            steatotic_n_range: (min, max) LD count for steatotic cells.
            steatotic_fraction: Fraction of cells that are steatotic (lipid-laden).
            radius_range: LD radius in world px (world scale, not internal).
            intensity_mean/std: BODIPY brightness per LD (0–255).

        Returns:
            mode_id for the bodipy channel.
        """
        from scipy.ndimage import gaussian_filter
        rng = self.rng
        s = self.internal_scale

        self._ld_cells = []  # List[List[dict]] per cell
        self._ld_steatotic = []  # bool: is this cell steatotic?

        for i, center in enumerate(self.centers[:self.nb_cells]):
            cx, cy = float(center[0]), float(center[1])
            if hasattr(self, 'cell_areas'):
                cell_r = float(np.sqrt(self.cell_areas[i] / np.pi))
            else:
                cell_r = 30.0
            nuc_r = float(self.nucleus_radii[i]) if hasattr(self, 'nucleus_radii') else cell_r * 0.3

            is_steatotic = float(rng.random()) < steatotic_fraction
            self._ld_steatotic.append(is_steatotic)

            n_range = steatotic_n_range if is_steatotic else normal_n_range
            n_ld = int(rng.integers(n_range[0], n_range[1] + 1))

            droplets = []
            attempts = 0
            while len(droplets) < n_ld and attempts < n_ld * 30:
                attempts += 1
                angle = float(rng.uniform(0, 2 * np.pi))
                # LDs are larger → need more space, tend to cluster near nucleus
                r_place = float(rng.uniform(nuc_r * 1.05, cell_r * 0.80))
                lx = cx + r_place * np.cos(angle)
                ly = cy + r_place * np.sin(angle)
                lx = float(np.clip(lx, 2, self.width - 2))
                ly = float(np.clip(ly, 2, self.height - 2))
                ld_radius = float(rng.uniform(radius_range[0], radius_range[1]))
                intensity = float(np.clip(rng.normal(intensity_mean, intensity_std), 80, 255))

                # Non-overlapping constraint (LDs don't fuse in healthy cells)
                overlap = False
                for existing in droplets:
                    dist = np.sqrt((lx - existing["x"])**2 + (ly - existing["y"])**2)
                    if dist < ld_radius + existing["r"] + 1.0:
                        overlap = True
                        break
                if not overlap:
                    droplets.append({"x": lx, "y": ly, "r": ld_radius, "intensity": intensity})
            self._ld_cells.append(droplets)

        # PSF edge-smoothing sigma: mild blur to simulate focus-plane PSF
        self._ld_sigma_internal = max(1.0, 0.4 * s)

        # Render full BODIPY image using filled circles + PSF blur
        ld_img = self._render_lipid_droplets_full()

        mode_id = self.add_channel(
            "bodipy", ld_img,
            filter_label="TagGFP2(483/506)", led_label="GREEN",
        )
        self._ld_mode_id = mode_id

        if core is not None:
            core.defineConfig("Fake", "bodipy-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "bodipy-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")
        return mode_id

    def get_lipid_droplet_stats(self) -> dict:
        """Return ground truth lipid droplet statistics per cell."""
        if not hasattr(self, '_ld_cells'):
            return {"enabled": False}
        per_cell = []
        all_counts = []
        for i, droplets in enumerate(self._ld_cells):
            n = len(droplets)
            all_counts.append(n)
            per_cell.append({
                "cell_id": i,
                "n_droplets": n,
                "is_steatotic": bool(getattr(self, '_ld_steatotic', [False] * (i + 1))[i]),
                "mean_radius_px": round(float(np.mean([d["r"] for d in droplets])) if droplets else 0.0, 1),
                "positions_world": [[round(d["x"], 1), round(d["y"], 1)] for d in droplets],
            })
        n_steatotic = sum(1 for x in getattr(self, '_ld_steatotic', []) if x)
        return {
            "enabled": True,
            "n_cells": len(per_cell),
            "n_steatotic": n_steatotic,
            "n_normal": len(per_cell) - n_steatotic,
            "steatotic_fraction": round(n_steatotic / max(1, len(per_cell)), 3),
            "total_droplets": sum(all_counts),
            "mean_per_cell": round(float(np.mean(all_counts)), 1),
            "per_cell": per_cell,
        }

    def _render_lipid_droplets_full(self) -> np.ndarray:
        """Render BODIPY lipid droplet channel at internal resolution.

        Each LD is drawn as a filled circle (cv2.circle) then the whole image
        is convolved with a mild PSF Gaussian to smooth disc edges.
        Larger LDs appear as proportionally larger blobs.
        """
        from scipy.ndimage import gaussian_filter
        s = self.internal_scale
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)
        if not hasattr(self, '_ld_cells'):
            return np.zeros((self._ih, self._iw, 3), dtype=np.uint8)

        all_intensities = []
        for cell_droplets in self._ld_cells:
            for ld in cell_droplets:
                lx_i = int(round(ld["x"] * s))
                ly_i = int(round(ld["y"] * s))
                r_i = max(2, int(round(ld["r"] * s)))
                if not (0 <= lx_i < self._iw and 0 <= ly_i < self._ih):
                    continue
                # Draw filled circle — no rectangle artifacts
                cv2.circle(buf, (lx_i, ly_i), r_i, float(ld["intensity"]), -1)
                all_intensities.append(ld["intensity"])

        sigma = float(getattr(self, '_ld_sigma_internal', max(1.0, 0.4 * s)))
        if buf.max() > 0:
            buf = gaussian_filter(buf, sigma=sigma)
            # Re-scale to preserve LD peak brightness
            target = max(all_intensities) if all_intensities else 210.0
            buf = buf / max(buf.max(), 1e-6) * target

        img_g = np.clip(buf, 0, 255).astype(np.uint8)
        return cv2.merge([img_g, img_g, img_g])

    # ---- Stress Granule / Condensate Dynamics ----

    def enable_stress_granules(
        self,
        core=None,
        foci_radius_range: tuple = (2.0, 4.0),
        max_foci_per_cell: int = 12,
        formation_rate: float = 0.25,
        dissolution_rate: float = 0.12,
        intensity_mean: float = 205.0,
        intensity_std: float = 20.0,
        heterogeneity: float = 0.35,
        baseline_foci: int = 0,
    ):
        """Enable GFP-tagged stress granule dynamics (G3BP1-GFP model).

        Stress granules (SGs) are biomolecular condensates that form in the
        cytoplasm in response to stress (arsenite, heat shock, UV). G3BP1-GFP
        is a canonical SG marker visible as bright puncta appearing within
        minutes of stress onset.

        Dynamics:
          - Baseline: diffuse cytoplasmic GFP, 0–1 foci per cell.
          - Under stress (``_sg_stress_active=True``): foci form stochastically
            with probability ``formation_rate × sensitivity_i`` per step.
          - Washout: foci dissolve with probability ``dissolution_rate`` per
            focus per step.

        Args:
            core: Optional UniMMCore — registers 'granule-channel' (GFP).
            foci_radius_range: (min, max) granule radius in world px.
            max_foci_per_cell: Maximum foci count per cell.
            formation_rate: Per-stressed-cell probability of new focus per step.
            dissolution_rate: Per-focus dissolution probability per step (washout).
            intensity_mean/std: Peak GFP brightness per focus (0–255).
            heterogeneity: Cell-to-cell variation in stress sensitivity (σ rel.).
            baseline_foci: Initial foci per cell (0 = fully pre-stress).

        Returns:
            mode_id for the granule channel.
        """
        rng = self.rng

        self._sg_stress_active = False
        self._sg_formation_rate = formation_rate
        self._sg_dissolution_rate = dissolution_rate
        self._sg_max_foci = max_foci_per_cell
        self._sg_radius_range = foci_radius_range
        self._sg_intensity_mean = intensity_mean
        self._sg_intensity_std = intensity_std

        # Per-cell stress sensitivity (heterogeneity in SG response)
        raw_sens = 1.0 + rng.normal(0, heterogeneity, self.nb_cells)
        self._sg_sensitivity = np.clip(raw_sens, 0.2, 1.8).astype(np.float32)

        # Initialise foci (list-of-lists)
        self._sg_cells = []
        for i in range(self.nb_cells):
            cell_foci = []
            if baseline_foci > 0:
                for _ in range(baseline_foci):
                    foc = self._sg_new_focus(i)
                    if foc is not None:
                        cell_foci.append(foc)
            self._sg_cells.append(cell_foci)

        self._sg_mode_id = None

        # Render initial GFP image and register channel
        sg_img = self._render_stress_granules_full()
        mode_id = self.add_channel(
            "granule", sg_img,
            filter_label="TagGFP2(483/506)", led_label="GREEN",
        )
        self._sg_mode_id = mode_id

        if core is not None:
            core.defineConfig("Fake", "granule-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "granule-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")
        return mode_id

    def _sg_new_focus(self, cell_idx: int):
        """Place a new stress granule in the cytoplasm of cell_idx.

        Returns a dict {x, y, r, intensity} or None if placement fails.
        """
        rng = self.rng
        i = cell_idx
        cx, cy = float(self.centers[i][0]), float(self.centers[i][1])
        cell_r = float(np.sqrt(self.cell_areas[i] / np.pi)) if hasattr(self, 'cell_areas') else 30.0
        nuc_r = float(self.nucleus_radii[i]) if hasattr(self, 'nucleus_radii') else cell_r * 0.3
        existing = self._sg_cells[i]

        for _ in range(40):
            angle = float(rng.uniform(0, 2 * np.pi))
            r_place = float(rng.uniform(nuc_r * 1.1, cell_r * 0.82))
            fx = cx + r_place * np.cos(angle)
            fy = cy + r_place * np.sin(angle)
            fx = float(np.clip(fx, 2, self.width - 2))
            fy = float(np.clip(fy, 2, self.height - 2))
            fr = float(rng.uniform(self._sg_radius_range[0], self._sg_radius_range[1]))
            intensity = float(np.clip(
                rng.normal(self._sg_intensity_mean, self._sg_intensity_std),
                120, 255,
            ))
            # Check overlap with existing foci
            overlap = False
            for ef in existing:
                d = np.sqrt((fx - ef['x'])**2 + (fy - ef['y'])**2)
                if d < fr + ef['r'] + 1.0:
                    overlap = True
                    break
            if not overlap:
                return {'x': fx, 'y': fy, 'r': fr, 'intensity': intensity}
        return None

    def _apply_stress_granules(self, dt: float = 1.0):
        """Update stress granule state based on current stress flag.

        Called every step(). Forms new foci when stressed, dissolves foci
        during washout. Re-renders the granule channel.
        """
        if not hasattr(self, '_sg_cells') or self._sg_mode_id is None:
            return

        rng = self.rng
        changed = False

        for i in range(self.nb_cells):
            if not self.alive[i]:
                continue
            foci = self._sg_cells[i]
            sens = float(self._sg_sensitivity[i])

            if self._sg_stress_active:
                # Stochastic formation (capped)
                if len(foci) < self._sg_max_foci:
                    prob = self._sg_formation_rate * sens * dt
                    if float(rng.random()) < prob:
                        foc = self._sg_new_focus(i)
                        if foc is not None:
                            foci.append(foc)
                            changed = True
            else:
                # Stochastic dissolution
                if foci:
                    survive = [
                        f for f in foci
                        if float(rng.random()) >= self._sg_dissolution_rate * dt
                    ]
                    if len(survive) < len(foci):
                        self._sg_cells[i] = survive
                        changed = True

        if changed:
            # Update cached channel image in _extra_channels dict
            new_img = self._render_stress_granules_full()
            if hasattr(self, '_extra_channels') and self._sg_mode_id in self._extra_channels:
                self._extra_channels[self._sg_mode_id]['image'] = new_img

    def _render_stress_granules_full(self) -> np.ndarray:
        """Render G3BP1-GFP channel: diffuse cytoplasmic + bright puncta.

        Cytoplasm dim (G3BP1 diffuse ~30 intensity), nucleus excluded (~15),
        foci bright (peak ~200). Uses cv2.circle + Gaussian blur.
        """
        from scipy.ndimage import gaussian_filter
        s = self.internal_scale
        buf = np.zeros((self._ih, self._iw), dtype=np.float32)

        # --- Diffuse cytoplasmic GFP background ---
        # Use cell masks from rendered tissue if available; otherwise paint
        # a dim uniform background over the whole image
        cyto_level = 35.0
        nuc_level = 12.0

        # Paint each alive cell's cytoplasm dim and nucleus very dim
        for i in range(self.nb_cells):
            if not self.alive[i]:
                continue
            cx_i = int(round(float(self.centers[i][0]) * s))
            cy_i = int(round(float(self.centers[i][1]) * s))
            cell_r = float(np.sqrt(self.cell_areas[i] / np.pi)) if hasattr(self, 'cell_areas') else 30.0
            nuc_r = float(self.nucleus_radii[i]) if hasattr(self, 'nucleus_radii') else cell_r * 0.3
            cr_i = int(round(cell_r * s))
            nr_i = max(2, int(round(nuc_r * s)))
            if cr_i > 0:
                cv2.circle(buf, (cx_i, cy_i), cr_i, cyto_level, -1)
            if nr_i > 0:
                cv2.circle(buf, (cx_i, cy_i), nr_i, nuc_level, -1)

        # --- Stress granule puncta (bright) ---
        if hasattr(self, '_sg_cells'):
            for cell_foci in self._sg_cells:
                for f in cell_foci:
                    fx_i = int(round(f['x'] * s))
                    fy_i = int(round(f['y'] * s))
                    fr_i = max(2, int(round(f['r'] * s)))
                    if 0 <= fx_i < self._iw and 0 <= fy_i < self._ih:
                        cv2.circle(buf, (fx_i, fy_i), fr_i,
                                   float(f['intensity']), -1)

        # PSF blur
        sigma = max(0.5 * s, 1.0)
        if buf.max() > 0:
            buf = gaussian_filter(buf, sigma=sigma)

        img_g = np.clip(buf, 0, 255).astype(np.uint8)
        return cv2.merge([img_g, img_g, img_g])

    def set_stress_granule_active(self, active: bool):
        """Set whether arsenite/heat stress is active (triggers SG formation)."""
        if hasattr(self, '_sg_stress_active'):
            self._sg_stress_active = active

    def get_stress_granule_state(self) -> dict:
        """Return ground truth stress granule statistics.

        Returns:
            dict with per-cell foci counts, n_stressed (≥3 foci), mean_foci.
        """
        if not hasattr(self, '_sg_cells'):
            return {"enabled": False}
        counts = [len(self._sg_cells[i]) for i in range(self.nb_cells)
                  if self.alive[i]]
        n_stressed = sum(1 for c in counts if c >= 3)
        return {
            "enabled": True,
            "stress_active": bool(self._sg_stress_active),
            "n_alive": len(counts),
            "n_stressed_cells": n_stressed,  # ≥3 foci
            "stress_fraction": round(n_stressed / max(1, len(counts)), 3),
            "mean_foci_per_cell": round(float(np.mean(counts)), 2) if counts else 0.0,
            "total_foci": int(sum(counts)),
            "per_cell_counts": counts,
        }

    # ---- Live/Dead Viability Staining ----

    def enable_viability_staining(
        self,
        core=None,
        live_fraction: float = 0.85,
        calcein_intensity: float = 190.0,
        pi_intensity: float = 220.0,
        calcein_dead_fraction: float = 0.05,
        rng_seed: int = None,
    ):
        """Add live/dead viability staining (Calcein-AM / Propidium Iodide).

        Dead cells (fraction = 1 - live_fraction) are randomly selected.
        - Calcein-AM (green, 'calcein-channel'): bright in LIVE cells,
            very dim in dead cells (calcein is excluded from dead cell cytoplasm).
        - PI (red, 'pi-channel'): absent in live cells, bright in DEAD cells
            (PI intercalates into DNA of cells with compromised membranes).

        Typical appearance:
          - Live cell: bright uniform green fill (calcein throughout cytoplasm)
          - Dead cell: bright red nuclear blob (PI), very dim green background

        Args:
            core: Optional UniMMCore — registers 'calcein-channel' and 'pi-channel'.
            live_fraction: Fraction of cells that are alive (0-1).
            calcein_intensity: Calcein brightness in live cells (0-255).
            pi_intensity: PI brightness in dead cells (0-255).
            calcein_dead_fraction: Residual calcein in dead cells (0-0.2).
            rng_seed: Optional RNG seed for dead cell selection.

        Returns:
            dict with calcein_mode_id, pi_mode_id, dead_cell_indices.
        """
        rng = np.random.default_rng(rng_seed) if rng_seed is not None else self.rng

        # Randomly assign dead/live status
        n_cells = self.nb_cells
        renderable_cells = [i for i in range(n_cells) if
                            (self._renderable[i] if hasattr(self, '_renderable') else True)]
        n_renderable = len(renderable_cells)
        n_dead = max(0, round(n_renderable * (1.0 - live_fraction)))

        dead_indices_rel = rng.choice(n_renderable, size=n_dead, replace=False)
        dead_set = set(renderable_cells[i] for i in dead_indices_rel)

        self._viab_dead_set = dead_set
        self._viab_live_fraction = live_fraction
        self._calcein_intensity = calcein_intensity
        self._pi_intensity = pi_intensity
        self._calcein_dead_fraction = calcein_dead_fraction

        # Render both channels
        calcein_img = self._render_calcein_full()
        pi_img = self._render_pi_full()

        # Register channels
        # Use available filter wheel labels:
        # Calcein-AM (~517nm emission): obeYFP(514/528) + GREEN LED
        # PI (~617nm emission): mRFP1-Q667(549/570) + ORANGE LED
        calcein_mode_id = self.add_channel(
            "calcein", calcein_img,
            filter_label="obeYFP(514/528)", led_label="GREEN",
        )
        self._calcein_mode_id = calcein_mode_id

        pi_mode_id = self.add_channel(
            "pi", pi_img,
            filter_label="mRFP1-Q667(549/570)", led_label="ORANGE",
        )
        self._pi_mode_id = pi_mode_id

        if core is not None:
            # Register hardware configs using defineConfig(group, config, device, prop, value)
            core.defineConfig("Fake", "calcein-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Fake", "calcein-channel",
                              "Filter Wheel", "Label", "obeYFP(514/528)")
            # PI channel (mRFP1-Q667 filter, ORANGE LED)
            core.defineConfig("Fake", "pi-channel",
                              "LED", "Label", "ORANGE")
            core.defineConfig("Fake", "pi-channel",
                              "Filter Wheel", "Label", "mRFP1-Q667(549/570)")

        return {
            "calcein_mode_id": calcein_mode_id,
            "pi_mode_id": pi_mode_id,
            "dead_cell_indices": sorted(dead_set),
            "n_dead": n_dead,
            "n_alive": n_renderable - n_dead,
            "actual_live_fraction": (n_renderable - n_dead) / max(1, n_renderable),
        }

    def _render_calcein_full(self) -> np.ndarray:
        """Render Calcein-AM channel: bright cytoplasmic fill in live cells.

        Live cells: bright uniform green across entire cell territory.
        Dead cells: very dim (residual calcein fraction).
        Returns 3-channel BGR uint8 image (same format as other channels).
        """
        s = self.internal_scale
        H, W = self._ih, self._iw
        img = np.zeros((H, W), dtype=np.float32)

        if not hasattr(self, '_viab_dead_set'):
            out = np.zeros((H, W, 3), dtype=np.uint8)
            return out

        # Render using membrane channel as territory reference
        if self._mem_full is None:
            self._render_full_tissue()

        dead_set = self._viab_dead_set
        calcein_bright = self._calcein_intensity
        calcein_dim = calcein_bright * self._calcein_dead_fraction

        # Draw filled circles per cell (approximates Voronoi territory)
        for i in range(self.nb_cells):
            if not (self._renderable[i] if hasattr(self, '_renderable') else True):
                continue
            intensity = float(calcein_dim if i in dead_set else calcein_bright)
            if intensity < 2:
                continue
            cx = int(round(float(self.centers[i][0]) * s))
            cy = int(round(float(self.centers[i][1]) * s))
            if hasattr(self, 'cell_areas'):
                cell_r = int(round(float(np.sqrt(self.cell_areas[i] / np.pi)) * s * 0.88))
            else:
                cell_r = int(30 * s)
            cell_r = max(1, cell_r)
            cv2.circle(img, (cx, cy), cell_r, intensity, -1)
            # Softer outer ring
            cv2.circle(img, (cx, cy), int(cell_r * 1.15), intensity * 0.35, max(1, cell_r // 4))

        # Slight nuclear exclusion: calcein dimmer in nucleus
        if self._nuc_full is not None:
            nuc_gray = self._nuc_full[:, :, 0] if self._nuc_full.ndim == 3 else self._nuc_full
            nuc_mask = nuc_gray > 20
            img[nuc_mask] *= 0.55

        # Apply blur for realism
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5 * s)

        # Shot noise
        noise = self.rng.poisson(np.maximum(0, img * 0.08)).astype(np.float32)
        img = np.clip(img + noise * 0.25, 0, 255).astype(np.uint8)

        # Return 3-channel BGR (required by snap_frame → cvtColor)
        return cv2.merge([img, img, img])

    def _render_pi_full(self) -> np.ndarray:
        """Render PI channel: nuclear staining in dead cells only.

        Dead cells: bright nuclear blob (PI intercalates into DNA).
        Live cells: background noise only.
        Returns 3-channel BGR uint8 image.
        """
        s = self.internal_scale
        H, W = self._ih, self._iw
        img = np.zeros((H, W), dtype=np.float32)

        if not hasattr(self, '_viab_dead_set'):
            return np.zeros((H, W, 3), dtype=np.uint8)

        dead_set = self._viab_dead_set
        pi_intensity = float(self._pi_intensity)

        for i in dead_set:
            if i >= self.nb_cells:
                continue
            cx = int(round(float(self.centers[i][0]) * s))
            cy = int(round(float(self.centers[i][1]) * s))
            if hasattr(self, 'nucleus_radii'):
                nuc_r = max(2, int(round(float(self.nucleus_radii[i]) * s)))
            elif hasattr(self, 'cell_areas'):
                nuc_r = max(2, int(round(float(np.sqrt(self.cell_areas[i] / np.pi)) * s * 0.3)))
            else:
                nuc_r = max(2, int(12 * s))

            # Bright nuclear fill
            cv2.circle(img, (cx, cy), nuc_r, pi_intensity, -1)
            # Soft outer glow
            cv2.circle(img, (cx, cy), int(nuc_r * 1.4), pi_intensity * 0.3, max(1, nuc_r // 3))

        # Slight blur
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0 * s)

        # Minimal noise floor (autofluorescence)
        noise_floor = self.rng.poisson(3.0, size=(H, W)).astype(np.float32)
        img = np.clip(img + noise_floor, 0, 255).astype(np.uint8)

        # Return 3-channel BGR
        return cv2.merge([img, img, img])

    def get_viability_state(self) -> dict:
        """Get ground truth viability information.

        Returns:
            dict with n_alive, n_dead, live_fraction, dead_cell_positions.
        """
        if not hasattr(self, '_viab_dead_set'):
            return {"error": "viability staining not enabled"}

        dead_set = self._viab_dead_set
        n_dead = len(dead_set)
        n_alive = self.nb_cells - n_dead

        dead_positions = []
        for i in sorted(dead_set):
            if i < len(self.centers):
                cx, cy = float(self.centers[i][0]), float(self.centers[i][1])
                dead_positions.append({
                    "cell_id": int(i),
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                })

        return {
            "n_dead": n_dead,
            "n_alive": n_alive,
            "live_fraction": round(n_alive / max(1, n_alive + n_dead), 3),
            "dead_cell_positions": dead_positions,
        }

    # ---- FISH Probes (Gene Copy Number) ----

    def enable_fish_probes(
        self,
        core=None,
        locus_copies: int = 2,
        amplified_fraction: float = 0.15,
        deleted_fraction: float = 0.10,
        amplified_copies_range: tuple = (3, 6),
        probe_intensity: float = 215.0,
        probe_intensity_std: float = 20.0,
        fwhm_world_px: float = 1.8,
    ):
        """Add FISH probes for gene copy number analysis (Cy3-labeled, 'fish-channel').

        Simulates FISH (Fluorescence In Situ Hybridization) with a gene-of-interest
        locus probe.  Bright, sub-resolution Gaussian spots appear inside each
        nucleus — one spot per gene copy:

          Normal diploid cells:  2 spots (locus_copies=2)
          Amplified cells:       3–6 spots (oncogene amplification)
          Deleted / LOH cells:   0–1 spots

        Rendering: each spot is a diffraction-limited Gaussian (FWHM ~0.18 µm).
        At 40x the spots are unresolved single-pixel blobs.  At 100x they become
        clearly separated point sources inside the nucleus.

        Typical workflow for the agent:
          1. 10x survey → count bright nuclei with ≥ 3 fish spots (amplified fraction)
          2. 40x → navigate to candidate amplified cell
          3. 100x → count individual spots per nucleus precisely

        Args:
            core: Optional UniMMCore — registers 'fish-channel' config.
            locus_copies: Normal diploid copy number (typically 2).
            amplified_fraction: Fraction of cells with copy number amplification.
            deleted_fraction: Fraction of cells with deletion (0–1 copies).
            amplified_copies_range: (min, max) copy numbers for amplified cells.
            probe_intensity: Mean spot brightness (0–255).
            probe_intensity_std: Cell-to-cell brightness heterogeneity.
            fwhm_world_px: Full-width at half-maximum per spot in world px.
                1.8 px ≈ 0.18 µm (sub-diffraction at 40x, just resolved at 100x).

        Returns:
            mode_id for the fish channel.
        """
        rng = self.rng
        s = self.internal_scale

        # Convert FWHM to Gaussian σ in internal pixels
        sigma_internal = (fwhm_world_px * s) / 2.355

        self._fish_spots = []        # List[List[dict]]: per cell, per spot
        self._fish_copy_numbers = [] # GT copy number per cell
        self._fish_locus_copies = locus_copies

        renderable = getattr(self, '_renderable', None)

        for i in range(self.nb_cells):
            cx, cy = float(self.centers[i][0]), float(self.centers[i][1])
            nuc_r = (float(self.nucleus_radii[i])
                     if hasattr(self, 'nucleus_radii') else 20.0)

            # Determine copy number for this cell
            rv = float(rng.random())
            if rv < amplified_fraction:
                n_copies = int(rng.integers(amplified_copies_range[0],
                                            amplified_copies_range[1] + 1))
            elif rv < amplified_fraction + deleted_fraction:
                n_copies = int(rng.integers(0, 2))   # 0 or 1 (deletion/LOH)
            else:
                n_copies = locus_copies

            self._fish_copy_numbers.append(n_copies)

            # Ghost / non-renderable cells: record count, skip rendering
            if renderable is not None and not renderable[i]:
                self._fish_spots.append([])
                continue

            # Place n_copies spots randomly within the nucleus (uniform disc)
            spots = []
            for _ in range(n_copies):
                angle = float(rng.uniform(0, 2 * np.pi))
                r_place = float(rng.uniform(0, nuc_r * 0.80))
                sx = cx + r_place * np.cos(angle)
                sy = cy + r_place * np.sin(angle)
                intensity = float(np.clip(
                    rng.normal(probe_intensity, probe_intensity_std), 100, 255))
                spots.append({"x": sx, "y": sy, "intensity": intensity})
            self._fish_spots.append(spots)

        self._fish_sigma_internal = sigma_internal

        # Render initial FISH image and register channel
        fish_img = self._render_fish_full()

        # Cy3-like emission (550/570 nm) → orange/red filter wheel slot
        mode_id = self.add_channel(
            "fish", fish_img,
            filter_label="mRFP1-Q667(549/570)", led_label="ORANGE",
        )
        self._fish_mode_id = mode_id

        if core is not None:
            core.defineConfig("Fake", "fish-channel",
                              "LED", "Label", "ORANGE")
            core.defineConfig("Fake", "fish-channel",
                              "Filter Wheel", "Label", "mRFP1-Q667(549/570)")
        return mode_id

    def get_fish_copy_numbers(self) -> dict:
        """Return FISH ground truth: copy number per cell.

        Returns dict with:
          n_cells, locus_copies_normal, copy_number_histogram,
          n_normal / n_amplified / n_deleted, per_cell list.
        """
        if not hasattr(self, '_fish_copy_numbers'):
            return {"enabled": False}

        counts = self._fish_copy_numbers
        locus = getattr(self, '_fish_locus_copies', 2)

        from collections import Counter
        cn_hist = Counter(counts)

        per_cell = []
        for i, n in enumerate(counts):
            cx, cy = 0.0, 0.0
            if i < len(self.centers):
                cx, cy = float(self.centers[i][0]), float(self.centers[i][1])
            spots = self._fish_spots[i] if i < len(self._fish_spots) else []
            per_cell.append({
                "cell_id": int(i),
                "copy_number": int(n),
                "n_spots": int(len(spots)),
                "x": round(cx, 1), "y": round(cy, 1),
                "is_amplified": bool(n > locus),
                "is_deleted": bool(n < locus),
            })

        n_amp = sum(1 for p in per_cell if p["is_amplified"])
        n_del = sum(1 for p in per_cell if p["is_deleted"])

        return {
            "enabled": True,
            "n_cells": int(len(per_cell)),
            "locus_copies_normal": int(locus),
            "copy_number_histogram": {int(k): int(v)
                                       for k, v in sorted(cn_hist.items())},
            "n_normal": int(len(per_cell) - n_amp - n_del),
            "n_amplified": int(n_amp),
            "n_deleted": int(n_del),
            "amplified_fraction": round(n_amp / max(1, len(per_cell)), 3),
            "deleted_fraction": round(n_del / max(1, len(per_cell)), 3),
            "per_cell": per_cell,
        }

    def _render_fish_full(self) -> np.ndarray:
        """Render FISH channel: tight sub-resolution Gaussians inside nuclei."""
        s = self.internal_scale
        sigma = float(getattr(self, '_fish_sigma_internal', 2.5))
        r_px = max(4, int(sigma * 4.0))

        buf = np.zeros((self._ih, self._iw), dtype=np.float32)

        for cell_spots in self._fish_spots:
            for spot in cell_spots:
                xi = spot["x"] * s
                yi = spot["y"] * s
                intensity = float(spot["intensity"])
                xi_i, yi_i = int(round(xi)), int(round(yi))
                x_lo = max(0, xi_i - r_px)
                x_hi = min(self._iw, xi_i + r_px + 1)
                y_lo = max(0, yi_i - r_px)
                y_hi = min(self._ih, yi_i + r_px + 1)
                if x_lo >= x_hi or y_lo >= y_hi:
                    continue
                px_arr = np.arange(x_lo, x_hi, dtype=np.float32)
                py_arr = np.arange(y_lo, y_hi, dtype=np.float32)
                Xp, Yp = np.meshgrid(px_arr, py_arr)
                r2 = (Xp - xi) ** 2 + (Yp - yi) ** 2
                buf[y_lo:y_hi, x_lo:x_hi] += intensity * np.exp(
                    -r2 / (2 * sigma ** 2))

        img_g = np.clip(buf, 0, 255).astype(np.uint8)
        return cv2.merge([img_g, img_g, img_g])

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
        expected_spacing = np.sqrt(self.width * self.height / max(1, self.nb_cells))
        edge_zone = jagged * expected_spacing * 0.8

        # Determine which cells are in the wound
        wounded = []
        for i in range(self.nb_cells):
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

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 37.0  # mammalian default
        return float(self.state_devices["Temperature"].get("label", "37"))

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

        # FUCCI cell cycle progression (advances phase, updates intensities)
        self._advance_fucci(effective_dt)

        # Update fluorescence intensities before re-rendering
        if self._nuc_dynamics is not None or self._mem_dynamics is not None:
            self._apply_fluorescence_dynamics()

        # SLM-driven gene induction (updates nucleus_intensity based on illumination)
        self._apply_gene_induction()

        # SLM-driven photoconversion (permanent green→red switch)
        self._apply_photoconversion()

        # SLM-driven laser ablation (cell killing + wound healing)
        self._apply_laser_ablation()

        # Drug-induced nuclear translocation (NF-kB, etc.)
        self._apply_translocation(effective_dt)

        # Stress granule dynamics (arsenite/heat stress → cytoplasmic condensates)
        self._apply_stress_granules(effective_dt)

        # Z-drift: tissue plane moves, causing defocus unless agent refocuses
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        # Lysosome diffusion: slow random walk within cell (if enabled)
        if hasattr(self, '_lyso_cells') and self._lyso_diffusion_rate > 0:
            self._drift_lysosomes(effective_dt)

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

        self._advance_fucci(effective_dt)

        if self._nuc_dynamics is not None or self._mem_dynamics is not None:
            self._apply_fluorescence_dynamics()

        # Skip _apply_gene_induction() and _apply_photoconversion()
        # — those are observation-coupled (SLM mask)

        # Translocation is drug-coupled (not SLM), so advance it
        self._apply_translocation(effective_dt)

        # Stress granules are drug-coupled, advance in background too
        self._apply_stress_granules(effective_dt)

        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

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
        ghost_indices = np.where(~self.alive[:self.nb_cells])[0]
        if len(ghost_indices) == 0:
            return

        alive_indices = np.where(self.alive[:self.nb_cells])[0]
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
        for i in range(self.nb_cells):
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

        for j in range(self.nb_cells):
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

        for i in range(self.nb_cells):
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
            self.nb_cells += n_new

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
        for i in range(self.nb_cells):
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
        for i in range(self.nb_cells):
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
                else:
                    # Append new cell (only if no ghosts to recycle)
                    self._divide_append_one(i, new_center)

        # Apoptosis (same as _apoptose but with adjusted rate)
        for i in range(self.nb_cells):
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
        # Ghost weight
        self._ghost_weight = np.concatenate([self._ghost_weight, [1.0]])
        # FUCCI geminin intensity (daughter starts in G1 → 0.05)
        if hasattr(self, '_geminin_intensity'):
            gem_vals = getattr(self, '_fucci_geminin_intensity_values',
                               {0: 0.05, 1: 0.25, 2: 0.85, 3: 0.95})
            self._geminin_intensity = np.concatenate([
                self._geminin_intensity, [gem_vals.get(0, 0.05)]])
        self.nb_cells += 1

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

    # ---- Override snap_frame for lazy rendering ----

    def _map_slm_to_world(self, mask: np.ndarray) -> np.ndarray:
        """Map a viewport-space SLM mask to world coordinates.

        The SLM mask is 512x512 (viewport pixels). At 10x the viewport maps
        1:1 to world pixels. At higher magnifications, the viewport covers a
        smaller world region, so the mask must be placed at the correct FOV
        position in world space.

        Returns a bool array of shape (self.height, self.width).
        """
        # Determine FOV in world pixels (same logic as _crop_fov)
        obj = self.current_objectiv
        if obj == 100:
            fov_world = 64
        elif obj == 40:
            fov_world = 128
        elif obj == 20:
            fov_world = 256
        else:
            fov_world = min(512, self.width)

        # Stage center in world coords (matching _crop_fov, excluding drift)
        cx = int(self.camera_offset[0]) + self.viewport_width // 2
        cy = int(self.camera_offset[1]) + self.viewport_height // 2

        # Resize SLM mask from viewport (512x512) to FOV world pixels
        mask_fov = cv2.resize(
            mask.astype(np.uint8), (fov_world, fov_world),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # Create world-sized mask with SLM pattern placed at FOV position
        world_mask = np.zeros((self.height, self.width), dtype=bool)
        half = fov_world // 2
        x0 = max(0, min(cx - half, self.width - fov_world))
        y0 = max(0, min(cy - half, self.height - fov_world))

        # Clip to world bounds
        wx1 = min(self.width, x0 + fov_world)
        wy1 = min(self.height, y0 + fov_world)
        mw = wx1 - x0
        mh = wy1 - y0
        world_mask[y0:y0 + mh, x0:x0 + mw] = mask_fov[:mh, :mw]

        return world_mask

    def snap_frame(self, mask=None, **kwargs):
        """Snap frame with lazy re-rendering after dynamics.

        When auto_step=True, dynamics advance every `snaps_per_step` calls.
        This lets agents acquire multiple channels per timepoint (e.g. BF +
        membrane) before the simulation steps forward.

        If an SLM mask is provided, illuminated cells migrate faster during
        subsequent step() calls (optogenetic stimulation).
        """
        # Update objective BEFORE mapping SLM mask (otherwise we use stale
        # magnification from previous snap)
        self._update_objectif()

        # Map SLM mask to world coordinates (FOV-aware)
        if mask is not None and np.any(mask):
            self._stim_mask = self._map_slm_to_world(mask)
        else:
            self._stim_mask = None

        if self._bf_full is None:
            self._render_full_tissue()
            # Re-render dynamic extra channels (translocation GFP)
            if self._transloc_enabled and self._transloc_mode_id is not None:
                gfp_img = self._render_translocation_full()
                self._extra_channels[self._transloc_mode_id]["image"] = gfp_img
            # Re-render lysosome channel (positions may have drifted)
            if hasattr(self, '_lyso_mode_id') and self._lyso_mode_id is not None:
                lyso_img = self._render_lysosomes_full()
                self._extra_channels[self._lyso_mode_id]["image"] = lyso_img
            # Re-render stress granule channel (foci positions may have changed)
            if hasattr(self, '_sg_mode_id') and self._sg_mode_id is not None:
                sg_img = self._render_stress_granules_full()
                self._extra_channels[self._sg_mode_id]["image"] = sg_img
            # Re-render Geminin-GFP channel (FUCCI phase may have changed)
            if hasattr(self, '_geminin_mode_id') and self._geminin_mode_id is not None:
                gem_img = self._render_fucci_geminin_full()
                self._extra_channels[self._geminin_mode_id]["image"] = gem_img
        result = super().snap_frame(mask=mask, **kwargs)

        # Auto-advance dynamics after N snaps
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
        for i in range(self.nb_cells):
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
                 for i in range(self.nb_cells) if self.alive[i]]

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
