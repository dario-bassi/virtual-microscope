"""FUCCI cell-cycle reporter — backend-specific tissue simulation."""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim


class FucciSim(DynamicVoronoiSim):
    """DynamicVoronoiSim with FUCCI cell-cycle reporter."""

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

        # Override _mode_map for FUCCI channels
        self._mode_map = {
            ("mScarlet3(569/582)", "ORANGE"): 1,  # mCherry-Cdt1
            ("miRFP670(642/670)", "RED"): 2,      # mVenus-Geminin
        }

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
            core.defineConfig("Channel", "geminin-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Channel", "geminin-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")

        # Dedicated geminin intensity array (step function, no smooth interp)
        # Avoids S/G2 boundary confusion caused by smooth membrane_intensity
        # G1=0.05, S=0.25, G2=0.85, M=0.95 — clear gap at threshold ~0.65
        self._fucci_geminin_intensity_values = {0: 0.05, 1: 0.25, 2: 0.85, 3: 0.95}
        self._geminin_intensity = np.full(self.nb_cells, 0.05, dtype=float)

        # Apply initial intensities
        self._apply_fucci_intensities()

        # Register hooks
        self._step_hooks.append(self._advance_fucci)
        if self._geminin_mode_id is not None:
            self._render_hooks[self._geminin_mode_id] = self._render_fucci_geminin_full
        self._divide_hooks.append(self._fucci_on_divide)

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

    def _fucci_on_divide(self, parent_idx: int):
        """Propagate FUCCI geminin intensity to newly appended daughter cell."""
        if hasattr(self, '_geminin_intensity'):
            gem_vals = getattr(self, '_fucci_geminin_intensity_values',
                               {0: 0.05, 1: 0.25, 2: 0.85, 3: 0.95})
            self._geminin_intensity = np.concatenate([
                self._geminin_intensity, [gem_vals.get(0, 0.05)]])
