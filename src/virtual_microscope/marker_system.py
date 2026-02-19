"""
Marker System — N-channel fluorescence labeling.

Replaces the hardcoded 2-channel (nucleus + membrane) system with a
generic N-channel marker architecture. Each marker targets a cellular
structure and has configurable expression, intensity, and spectral properties.

Usage:
    markers = MarkerSet([
        Marker("DAPI", target="nucleus", fraction=1.0,
               intensity_range=(0.7, 1.0)),
        Marker("E-cadherin-AF488", target="membrane", fraction=1.0,
               intensity_range=(0.6, 0.9)),
        Marker("Ki67-AF647", target="nucleus", fraction=0.3,
               intensity_range=(0.5, 1.0), pattern="random"),
    ])

    # Apply to a VoronoiSim
    markers.apply(sim, rng)

    # Render a specific channel
    img = markers.render("DAPI", sim)
"""

import numpy as np
import cv2


class Marker:
    """A single fluorescent marker definition."""

    def __init__(
        self,
        name: str,
        target: str = "nucleus",
        fraction: float = 1.0,
        pattern: str = "all",
        intensity_range: tuple = (0.6, 1.0),
        intensity_mode: str = "uniform",
        excitation_nm: float = 0,
        emission_nm: float = 0,
        bright_range: tuple | None = None,
        dim_range: tuple | None = None,
        bright_fraction: float = 0.5,
    ):
        """
        Args:
            name: Marker name (e.g., "DAPI", "E-cadherin-AF488")
            target: Cellular structure to label:
                "nucleus", "membrane", "cytoplasm", "whole_cell"
            fraction: Fraction of cells expressing this marker (0-1)
            pattern: Spatial pattern: "all", "random", "left", "right", "gradient_x"
            intensity_range: (lo, hi) for uniform intensity mode
            intensity_mode: "uniform", "bimodal", or "gradient_x"
            excitation_nm: Excitation wavelength (for spectral modeling)
            emission_nm: Emission wavelength (for spectral modeling)
            bright_range: (lo, hi) for bright subpopulation in bimodal mode
            dim_range: (lo, hi) for dim subpopulation in bimodal mode
            bright_fraction: Fraction of positive cells that are bright (bimodal)
        """
        self.name = name
        self.target = target
        self.fraction = fraction
        self.pattern = pattern
        self.intensity_range = intensity_range
        self.intensity_mode = intensity_mode
        self.excitation_nm = excitation_nm
        self.emission_nm = emission_nm
        self.bright_range = bright_range or (0.85, 1.0)
        self.dim_range = dim_range or (0.4, 0.6)
        self.bright_fraction = bright_fraction

        # Per-cell state (populated by apply())
        self.positive = None  # bool array [n_cells]
        self.intensity = None  # float array [n_cells]

    def to_config(self) -> dict:
        """Convert to dict for apply_marker_config compatibility."""
        cfg = {
            "fraction": self.fraction,
            "pattern": self.pattern,
            "intensity": {"mode": self.intensity_mode, "range": list(self.intensity_range)},
        }
        if self.intensity_mode == "bimodal":
            cfg["intensity"]["bright_range"] = list(self.bright_range)
            cfg["intensity"]["dim_range"] = list(self.dim_range)
            cfg["intensity"]["bright_fraction"] = self.bright_fraction
        return cfg


class MarkerSet:
    """Collection of markers that can be applied to a simulation."""

    def __init__(self, markers: list[Marker] | None = None):
        self.markers = markers or []
        self._by_name = {m.name: m for m in self.markers}

    def add(self, marker: Marker):
        """Add a marker to the set."""
        self.markers.append(marker)
        self._by_name[marker.name] = marker

    def get(self, name: str) -> Marker | None:
        """Get marker by name."""
        return self._by_name.get(name)

    @property
    def channel_names(self) -> list[str]:
        """List of all marker/channel names."""
        return [m.name for m in self.markers]

    def apply(self, sim, rng=None):
        """Apply all markers to a VoronoiSim, setting per-cell expression.

        For each marker:
        1. Select which cells are positive (based on fraction and pattern)
        2. Assign intensities (based on mode)
        3. Store state on the Marker object
        """
        if rng is None:
            rng = np.random.default_rng()

        n = sim.nb_cells

        for marker in self.markers:
            n_pos = max(1, int(n * marker.fraction))

            # Select positive cells
            if marker.pattern == "all" or marker.fraction >= 1.0:
                indices = list(range(n))
            elif marker.pattern == "random":
                indices = sorted(rng.choice(n, n_pos, replace=False).tolist())
            elif marker.pattern == "left":
                left = [i for i in range(n) if sim.cell_centroids[i][0] < sim.width / 2]
                indices = sorted(rng.choice(left, min(n_pos, len(left)), replace=False).tolist())
            elif marker.pattern == "right":
                right = [i for i in range(n) if sim.cell_centroids[i][0] >= sim.width / 2]
                indices = sorted(rng.choice(right, min(n_pos, len(right)), replace=False).tolist())
            elif marker.pattern == "gradient_x":
                probs = np.array([sim.cell_centroids[i][0] / sim.width for i in range(n)])
                probs /= probs.sum()
                indices = sorted(rng.choice(n, n_pos, replace=False, p=probs).tolist())
            else:
                indices = sorted(rng.choice(n, n_pos, replace=False).tolist())

            # Set expression
            positive = np.zeros(n, dtype=bool)
            positive[indices] = True
            marker.positive = positive

            # Set intensities
            intensity = np.zeros(n, dtype=np.float64)
            if marker.intensity_mode == "uniform":
                lo, hi = marker.intensity_range
                intensity[:] = rng.uniform(lo, hi, n)
            elif marker.intensity_mode == "bimodal":
                n_bright = int(len(indices) * marker.bright_fraction)
                for j, idx in enumerate(indices):
                    if j < n_bright:
                        intensity[idx] = rng.uniform(*marker.bright_range)
                    else:
                        intensity[idx] = rng.uniform(*marker.dim_range)
            elif marker.intensity_mode == "gradient_x":
                lo, hi = marker.intensity_range
                for idx in indices:
                    frac_x = sim.cell_centroids[idx][0] / sim.width
                    intensity[idx] = lo + (hi - lo) * frac_x

            marker.intensity = intensity

            # Also update legacy VoronoiSim arrays for backward compatibility
            if marker.target == "nucleus":
                sim.has_nucleus_marker[:] = positive
                sim.nucleus_intensity[:] = intensity
            elif marker.target == "membrane":
                sim.has_membrane_marker[:] = positive
                sim.membrane_intensity[:] = intensity

    def render(self, channel_name: str, sim, width: int = None, height: int = None) -> np.ndarray:
        """Render a specific marker channel.

        Args:
            channel_name: Name of the marker to render
            sim: VoronoiSim instance
            width: Image width (defaults to sim.width)
            height: Image height (defaults to sim.height)

        Returns:
            Rendered image (uint8, HxWx3)
        """
        marker = self._by_name.get(channel_name)
        if marker is None:
            raise ValueError(f"Unknown marker: {channel_name}")

        w = width or sim.width
        h = height or sim.height
        img = np.zeros((h, w, 3), dtype=np.uint8)

        if marker.target == "nucleus":
            img = self._render_nucleus(marker, sim, img)
        elif marker.target == "membrane":
            img = self._render_membrane(marker, sim, img)
        elif marker.target == "cytoplasm":
            img = self._render_cytoplasm(marker, sim, img)
        elif marker.target == "whole_cell":
            img = self._render_whole_cell(marker, sim, img)

        return img

    def _render_nucleus(self, marker: Marker, sim, img: np.ndarray) -> np.ndarray:
        """Render nuclear marker (filled circles at centroids)."""
        # Autofluorescence in all cells
        for i, poly in enumerate(sim.cell_polygons[:sim.nb_cells]):
            if len(poly) < 3:
                continue
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            auto = int(sim.rng.uniform(5, 12))
            cv2.fillPoly(img, [pts], (auto, auto, auto))

        # Nucleus signal in positive cells
        for i in range(sim.nb_cells):
            if not marker.positive[i]:
                continue
            cx, cy = sim.cell_centroids[i]
            r = max(3, int(sim.nucleus_radii[i]))
            if 0 <= cx < sim.width and 0 <= cy < sim.height:
                val = int(marker.intensity[i] * 200)
                cv2.circle(img, (int(cx), int(cy)), r, (val, val, val), -1, cv2.LINE_AA)
                bright = min(255, int(val * 1.2))
                cv2.circle(img, (int(cx), int(cy)), max(1, r // 2),
                           (bright, bright, bright), -1, cv2.LINE_AA)
        return img

    def _render_membrane(self, marker: Marker, sim, img: np.ndarray) -> np.ndarray:
        """Render membrane marker (cell boundary polylines + shared boundaries)."""
        # Autofluorescence in all cells
        for i, poly in enumerate(sim.cell_polygons[:sim.nb_cells]):
            if len(poly) < 3:
                continue
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            auto = int(sim.rng.uniform(3, 8))
            cv2.fillPoly(img, [pts], (auto, auto, auto))

        # Base membrane
        for i, poly in enumerate(sim.cell_polygons[:sim.nb_cells]):
            if len(poly) < 3 or not marker.positive[i]:
                continue
            val = int(marker.intensity[i] * 140)
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (val, val, val), thickness=2, lineType=cv2.LINE_AA)

        # Shared boundaries
        n = sim.nb_cells
        for ridge_idx, (p1, p2) in enumerate(sim.vor.ridge_points):
            if p1 >= n or p2 >= n:
                continue
            if not (marker.positive[p1] and marker.positive[p2]):
                continue
            vert_indices = sim.vor.ridge_vertices[ridge_idx]
            if -1 in vert_indices:
                continue
            v1 = np.clip(sim.vor.vertices[vert_indices[0]], [0, 0], [sim.width - 1, sim.height - 1])
            v2 = np.clip(sim.vor.vertices[vert_indices[1]], [0, 0], [sim.width - 1, sim.height - 1])
            avg = (marker.intensity[p1] + marker.intensity[p2]) / 2
            bright = min(255, int(avg * 220))
            cv2.line(img, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])),
                     (bright, bright, bright), thickness=2, lineType=cv2.LINE_AA)
        return img

    def _render_cytoplasm(self, marker: Marker, sim, img: np.ndarray) -> np.ndarray:
        """Render cytoplasmic marker (cell interior minus nucleus)."""
        for i, poly in enumerate(sim.cell_polygons[:sim.nb_cells]):
            if len(poly) < 3 or not marker.positive[i]:
                continue

            # Fill cell polygon
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            val = int(marker.intensity[i] * 160)
            cv2.fillPoly(img, [pts], (val, val, val))

            # Subtract nucleus (make it darker)
            cx, cy = sim.cell_centroids[i]
            r = max(3, int(sim.nucleus_radii[i]))
            if 0 <= cx < sim.width and 0 <= cy < sim.height:
                dark = max(0, int(val * 0.3))  # nucleus region is dimmer
                cv2.circle(img, (int(cx), int(cy)), r, (dark, dark, dark), -1, cv2.LINE_AA)
        return img

    def _render_whole_cell(self, marker: Marker, sim, img: np.ndarray) -> np.ndarray:
        """Render whole-cell marker (entire cell polygon)."""
        for i, poly in enumerate(sim.cell_polygons[:sim.nb_cells]):
            if len(poly) < 3 or not marker.positive[i]:
                continue
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            val = int(marker.intensity[i] * 180)
            cv2.fillPoly(img, [pts], (val, val, val))
        return img

    def get_ground_truth(self, sim) -> dict:
        """Get ground truth for all markers."""
        per_marker = {}
        for marker in self.markers:
            n_pos = int(marker.positive.sum()) if marker.positive is not None else 0
            per_marker[marker.name] = {
                "target": marker.target,
                "n_positive": n_pos,
                "fraction": round(n_pos / sim.nb_cells, 3) if sim.nb_cells > 0 else 0,
            }
        return {
            "n_channels": len(self.markers),
            "channels": per_marker,
        }

    @classmethod
    def from_config(cls, config: list[dict]) -> "MarkerSet":
        """Create MarkerSet from a list of marker config dicts.

        Config format (each item):
            {
                "name": "DAPI",
                "target": "nucleus",
                "fraction": 1.0,
                "pattern": "all",
                "intensity": {"mode": "uniform", "range": [0.7, 1.0]},
                "excitation_nm": 360,
                "emission_nm": 460,
            }
        """
        markers = []
        for cfg in config:
            icfg = cfg.get("intensity", {})
            markers.append(Marker(
                name=cfg["name"],
                target=cfg.get("target", "nucleus"),
                fraction=cfg.get("fraction", 1.0),
                pattern=cfg.get("pattern", "all"),
                intensity_range=tuple(icfg.get("range", [0.6, 1.0])),
                intensity_mode=icfg.get("mode", "uniform"),
                excitation_nm=cfg.get("excitation_nm", 0),
                emission_nm=cfg.get("emission_nm", 0),
                bright_range=tuple(icfg["bright_range"]) if "bright_range" in icfg else None,
                dim_range=tuple(icfg["dim_range"]) if "dim_range" in icfg else None,
                bright_fraction=icfg.get("bright_fraction", 0.5),
            ))
        return cls(markers)
