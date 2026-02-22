"""Lipid droplet imaging — backend-specific tissue simulation."""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim


class LipidDropletSim(DynamicVoronoiSim):
    """DynamicVoronoiSim with BODIPY-stained lipid droplets."""

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

        for i, center in enumerate(self.centers[:self.n_cells]):
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
            core.defineConfig("Channel", "bodipy-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Channel", "bodipy-channel",
                              "Filter Wheel", "Label", "TagGFP2(483/506)")

        # Register render hook (lipid droplets are static — no step hook needed)
        self._render_hooks[self._ld_mode_id] = self._render_lipid_droplets_full

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
