"""Viability staining — backend-specific tissue simulation."""

import numpy as np
import cv2
from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim


class ViabilitySim(DynamicVoronoiSim):
    """DynamicVoronoiSim with Calcein-AM / PI viability staining."""

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
            core.defineConfig("Channel", "calcein-channel",
                              "LED", "Label", "GREEN")
            core.defineConfig("Channel", "calcein-channel",
                              "Filter Wheel", "Label", "obeYFP(514/528)")
            core.defineConfig("Channel", "pi-channel",
                              "LED", "Label", "ORANGE")
            core.defineConfig("Channel", "pi-channel",
                              "Filter Wheel", "Label", "mRFP1-Q667(549/570)")

        # Register render hooks (viability is static — no step hook needed)
        self._render_hooks[self._calcein_mode_id] = self._render_calcein_full
        self._render_hooks[self._pi_mode_id] = self._render_pi_full

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
