"""Lysosome imaging — backend-specific tissue simulation."""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim


class LysosomeSim(DynamicVoronoiSim):
    """DynamicVoronoiSim with LysoTracker-stained lysosomes."""

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
        self._lyso_sigma_internal = max(1.0, 0.4 * s)

        # Render lysotracker channel image
        lyso_img = self._render_lysosomes_full()
        mode_id = self.add_channel(
            "lysotracker", lyso_img,
            filter_label="TagGFP2(483/506)", led_label="GREEN",
        )
        self._lyso_mode_id = mode_id

        if core is not None:
            core.defineConfig("Channel", "lysotracker-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Channel", "lysotracker-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")

        # Register hooks
        if self._lyso_diffusion_rate > 0:
            self._step_hooks.append(self._drift_lysosomes)
        self._render_hooks[self._lyso_mode_id] = self._render_lysosomes_full

        return mode_id

    def _render_lysosomes_full(self) -> np.ndarray:
        """Render lysosomal puncta at internal resolution.

        Uses Gaussian PSF rendering for realistic diffraction-limited spots.
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

        # Gaussian PSF
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
