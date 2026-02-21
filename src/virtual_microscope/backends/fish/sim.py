"""FISH probe imaging — backend-specific tissue simulation."""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim


class FishSim(DynamicVoronoiSim):
    """DynamicVoronoiSim with FISH probes for gene copy number analysis."""

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

        Args:
            core: Optional UniMMCore — registers 'fish-channel' config.
            locus_copies: Normal diploid copy number (typically 2).
            amplified_fraction: Fraction of cells with copy number amplification.
            deleted_fraction: Fraction of cells with deletion (0–1 copies).
            amplified_copies_range: (min, max) copy numbers for amplified cells.
            probe_intensity: Mean spot brightness (0–255).
            probe_intensity_std: Cell-to-cell brightness heterogeneity.
            fwhm_world_px: Full-width at half-maximum per spot in world px.

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
            core.defineConfig("Channel", "fish-channel",
                              "LED", "Label", "ORANGE")
            core.defineConfig("Channel", "fish-channel",
                              "Filter Wheel", "Label", "mRFP1-Q667(549/570)")

        # Register render hook (FISH is static — no step hook needed)
        self._render_hooks[self._fish_mode_id] = self._render_fish_full

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
