"""
Voronoi-based confluent epithelial tissue simulator.

Generates tissue-like images where cells tile the field with shared
boundaries — fundamentally different from the scattered particle model.

Usage:
    tissue = VoronoiTissue(width=512, height=512, n_cells=80, seed=42)
    bf_image = tissue.render_brightfield()
    nuc_image = tissue.render_nuclei()
    mem_image = tissue.render_membrane()
"""

import numpy as np
import cv2
from scipy.spatial import Voronoi


class VoronoiTissue:
    """Confluent epithelial monolayer using Voronoi tessellation.

    Each cell is a Voronoi polygon. Shared edges are cell-cell junctions.
    Each cell has a nucleus at its centroid.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        n_cells: int = 80,
        seed: int = 42,
        nucleus_fraction: float = 0.3,
        jitter: float = 0.7,
    ):
        """
        Args:
            width, height: image dimensions in pixels
            n_cells: number of cells in the tissue
            seed: RNG seed for reproducibility
            nucleus_fraction: nucleus radius as fraction of cell equivalent radius
            jitter: 0=regular grid, 1=fully random placement
        """
        self.width = width
        self.height = height
        self.n_cells = n_cells
        self.seed = seed
        self.nucleus_fraction = nucleus_fraction
        self.rng = np.random.default_rng(seed)

        # Generate cell centers with optional jitter from hex grid
        self.centers = self._generate_centers(jitter)

        # Compute Voronoi tessellation with mirrored boundary points
        self.vor, self.regions, self.point_region_map = self._compute_voronoi()

        # Per-cell properties
        self.has_nucleus_marker = np.ones(n_cells, dtype=bool)
        self.has_membrane_marker = np.ones(n_cells, dtype=bool)
        self.nucleus_intensity = self.rng.uniform(0.7, 1.0, n_cells)
        self.membrane_intensity = self.rng.uniform(0.6, 0.9, n_cells)

        # Compute cell polygons and nuclei
        self.cell_polygons = self._extract_cell_polygons()
        self.cell_areas = self._compute_areas()
        self.cell_centroids = self._compute_centroids()
        self.nucleus_radii = self._compute_nucleus_radii()

        # Per-cell grayscale level (for brightfield variation)
        self.cell_gray = self.rng.uniform(100, 150, n_cells).astype(np.uint8)

    def _generate_centers(self, jitter: float) -> np.ndarray:
        """Generate cell centers using jittered hex grid."""
        # Estimate grid spacing from density
        area_per_cell = (self.width * self.height) / self.n_cells
        spacing = np.sqrt(area_per_cell / 0.866)  # hex packing factor

        # Create hex grid
        centers = []
        ny = int(self.height / (spacing * 0.866)) + 2
        nx = int(self.width / spacing) + 2
        for row in range(ny):
            for col in range(nx):
                x = col * spacing + (0.5 * spacing if row % 2 else 0)
                y = row * spacing * 0.866
                # Offset to center grid
                x -= spacing * 0.5
                y -= spacing * 0.5
                centers.append([x, y])

        centers = np.array(centers)

        # Apply jitter
        if jitter > 0:
            noise = self.rng.normal(0, spacing * 0.25 * jitter, centers.shape)
            centers += noise

        # Keep only points within bounds (with small margin)
        margin = -spacing * 0.3
        mask = (
            (centers[:, 0] >= margin)
            & (centers[:, 0] <= self.width - margin)
            & (centers[:, 1] >= margin)
            & (centers[:, 1] <= self.height - margin)
        )
        centers = centers[mask]

        # Trim or pad to exact n_cells
        if len(centers) > self.n_cells:
            # Keep the ones closest to center first, then random
            idx = self.rng.choice(len(centers), self.n_cells, replace=False)
            centers = centers[idx]
        elif len(centers) < self.n_cells:
            # Add random points to fill
            extra = self.n_cells - len(centers)
            new_pts = self.rng.uniform(
                [10, 10], [self.width - 10, self.height - 10], (extra, 2)
            )
            centers = np.vstack([centers, new_pts])

        return centers[:self.n_cells]

    def _compute_voronoi(self):
        """Compute Voronoi with mirrored boundary points for clean edges."""
        pts = self.centers.copy()
        n = len(pts)

        # Mirror points across all 4 boundaries
        mirror_pts = []
        mirror_pts.append(np.column_stack([-pts[:, 0], pts[:, 1]]))  # left
        mirror_pts.append(
            np.column_stack([2 * self.width - pts[:, 0], pts[:, 1]])
        )  # right
        mirror_pts.append(np.column_stack([pts[:, 0], -pts[:, 1]]))  # top
        mirror_pts.append(
            np.column_stack([pts[:, 0], 2 * self.height - pts[:, 1]])
        )  # bottom

        all_pts = np.vstack([pts] + mirror_pts)
        vor = Voronoi(all_pts)

        # Extract regions for original points only
        regions = []
        point_region_map = []
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                regions.append(region)
                point_region_map.append(i)

        return vor, regions, point_region_map

    def _extract_cell_polygons(self):
        """Get polygon vertices for each cell, clipped to image bounds."""
        polygons = []
        for region in self.regions:
            verts = self.vor.vertices[region]
            # Clip to image bounds
            verts[:, 0] = np.clip(verts[:, 0], 0, self.width - 1)
            verts[:, 1] = np.clip(verts[:, 1], 0, self.height - 1)
            polygons.append(verts)

        # Pad if we have fewer regions than cells
        while len(polygons) < self.n_cells:
            # Create a small polygon at the missing cell center
            idx = len(polygons)
            cx, cy = self.centers[idx]
            r = 10
            angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            poly = np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])
            polygons.append(poly)

        return polygons[:self.n_cells]

    def _compute_areas(self):
        """Compute area of each cell polygon."""
        areas = []
        for poly in self.cell_polygons:
            # Shoelace formula
            n = len(poly)
            if n < 3:
                areas.append(100.0)
                continue
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * abs(
                np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                + x[-1] * y[0]
                - x[0] * y[-1]
            )
            areas.append(max(area, 1.0))
        return np.array(areas)

    def _compute_centroids(self):
        """Compute centroid of each cell polygon."""
        centroids = []
        for poly in self.cell_polygons:
            centroids.append(poly.mean(axis=0))
        return np.array(centroids)

    def _compute_nucleus_radii(self):
        """Compute nucleus radius based on cell area."""
        equiv_radii = np.sqrt(self.cell_areas / np.pi)
        return equiv_radii * self.nucleus_fraction

    def set_marker_pattern(
        self,
        nucleus_positive_indices=None,
        membrane_positive_indices=None,
    ):
        """Set which cells express each marker.

        Args:
            nucleus_positive_indices: list of cell indices with nucleus marker.
                None means all cells.
            membrane_positive_indices: list of cell indices with membrane marker.
                None means all cells.
        """
        if nucleus_positive_indices is not None:
            self.has_nucleus_marker[:] = False
            for i in nucleus_positive_indices:
                if i < self.n_cells:
                    self.has_nucleus_marker[i] = True
        if membrane_positive_indices is not None:
            self.has_membrane_marker[:] = False
            for i in membrane_positive_indices:
                if i < self.n_cells:
                    self.has_membrane_marker[i] = True

    def render_brightfield(self, noise_level: float = 5.0) -> np.ndarray:
        """Render brightfield image of the tissue.

        Cells are filled with varying gray levels. Junctions appear as
        dark lines. Nuclei appear as darker spots within cells.
        """
        img = np.full((self.height, self.width), 128, dtype=np.uint8)

        # Draw cell interiors
        for i, poly in enumerate(self.cell_polygons):
            if len(poly) < 3:
                continue
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(img, [pts], int(self.cell_gray[i]))

        # Draw junctions (dark lines between cells)
        for i, poly in enumerate(self.cell_polygons):
            if len(poly) < 3:
                continue
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, 40, thickness=2, lineType=cv2.LINE_AA)

        # Draw nuclei (darker circles)
        for i in range(min(self.n_cells, len(self.cell_centroids))):
            cx, cy = self.cell_centroids[i]
            r = max(3, int(self.nucleus_radii[i]))
            if 0 <= cx < self.width and 0 <= cy < self.height:
                cv2.circle(img, (int(cx), int(cy)), r, 60, -1, cv2.LINE_AA)
                # Inner nucleus highlight
                cv2.circle(
                    img, (int(cx), int(cy)), max(1, r // 2), 50, -1, cv2.LINE_AA
                )

        # Add noise
        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def render_nuclei(self, noise_level: float = 3.0) -> np.ndarray:
        """Render nuclear fluorescence (DAPI-like).

        Only cells with has_nucleus_marker=True are visible.
        Bright round nuclei on dark background.
        """
        img = np.zeros((self.height, self.width), dtype=np.float32)

        for i in range(min(self.n_cells, len(self.cell_centroids))):
            if not self.has_nucleus_marker[i]:
                continue
            cx, cy = self.cell_centroids[i]
            r = max(3, int(self.nucleus_radii[i]))
            if 0 <= cx < self.width and 0 <= cy < self.height:
                intensity = self.nucleus_intensity[i] * 200
                # Gaussian-like nucleus
                cv2.circle(
                    img, (int(cx), int(cy)), r, float(intensity), -1, cv2.LINE_AA
                )
                # Brighter center
                cv2.circle(
                    img,
                    (int(cx), int(cy)),
                    max(1, r // 2),
                    float(intensity * 1.2),
                    -1,
                    cv2.LINE_AA,
                )

        # Slight blur for realism
        img = cv2.GaussianBlur(img, (5, 5), 1.0)

        # Add noise
        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 255)

        return img.astype(np.uint8)

    def render_membrane(self, noise_level: float = 3.0) -> np.ndarray:
        """Render membrane/junction fluorescence (E-cadherin-like).

        Cell boundaries appear as bright lines on dark background.
        Only cells with has_membrane_marker=True contribute.
        """
        img = np.zeros((self.height, self.width), dtype=np.float32)

        for i, poly in enumerate(self.cell_polygons):
            if len(poly) < 3:
                continue
            if not self.has_membrane_marker[i]:
                continue
            intensity = self.membrane_intensity[i] * 180
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                img, [pts], True, float(intensity), thickness=2, lineType=cv2.LINE_AA
            )

        # Slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0.8)

        # Add noise
        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 255)

        return img.astype(np.uint8)

    def get_cell_at_pixel(self, x: int, y: int) -> int:
        """Return index of cell containing pixel (x, y), or -1 if none."""
        point = np.array([x, y])
        for i, poly in enumerate(self.cell_polygons):
            if len(poly) < 3:
                continue
            if cv2.pointPolygonTest(
                poly.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False
            ) >= 0:
                return i
        return -1

    def get_ground_truth(self):
        """Return ground truth data for challenge grading."""
        cells = []
        for i in range(self.n_cells):
            cx, cy = self.cell_centroids[i]
            cells.append({
                "idx": i,
                "centroid_x": round(float(cx), 1),
                "centroid_y": round(float(cy), 1),
                "area": round(float(self.cell_areas[i]), 1),
                "nucleus_radius": round(float(self.nucleus_radii[i]), 1),
                "has_nucleus_marker": bool(self.has_nucleus_marker[i]),
                "has_membrane_marker": bool(self.has_membrane_marker[i]),
                "n_vertices": len(self.cell_polygons[i]),
            })

        return {
            "n_cells": self.n_cells,
            "cells": cells,
            "mean_area": round(float(self.cell_areas.mean()), 1),
            "n_nucleus_positive": int(self.has_nucleus_marker.sum()),
            "n_membrane_positive": int(self.has_membrane_marker.sum()),
        }
