"""Stress granule dynamics — backend-specific tissue simulation."""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim


class StressGranuleSim(DynamicVoronoiSim):
    """DynamicVoronoiSim with G3BP1-GFP stress granule dynamics."""

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
            core.defineConfig("Channel", "granule-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Channel", "granule-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")

        # Register hooks
        self._step_hooks.append(self._apply_stress_granules)
        self._render_hooks[self._sg_mode_id] = self._render_stress_granules_full

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
