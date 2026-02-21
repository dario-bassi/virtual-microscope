"""
Hemocytometer Simulator — Neubauer counting chamber.

Simulates a hemocytometer (cell counting chamber) for cell concentration
measurement. Renders a top-down view of the Neubauer improved grid with
cells scattered randomly.

Key concepts:
  - 3×3 mm total ruled area, viewed through microscope
  - Center 1×1 mm square divided into 25 groups (5×5 grid of group squares)
  - 4 corner squares (each 1×1 mm) used for counting (at 10x, these are
    the large L-shaped squares; here we use the 4 corner group squares
    of the center area)
  - Standard protocol: count cells in 4 corner squares of center grid
  - Concentration = (total count / 4) × dilution_factor × 10^4 cells/mL

Grid rendering:
  - World = 512×512 pixels representing ~3mm × 3mm field
  - Thick border lines around 1mm squares
  - Medium lines for 0.25mm subdivisions
  - Fine lines for 0.05mm subdivisions (in center)
  - Triple-ruled lines (characteristic of Neubauer chamber)

Channels:
  - BF (mode 0): phase contrast — grid lines dark, cells as bright/dark circles
  - Trypan blue (mode 1): dead cells bright (absorbed dye), live cells dim

Devices: Camera + Objective + Channel
"""

import numpy as np
import cv2
from typing import Optional


# ── Grid constants ──────────────────────────────────────────────────────
# Pixels per mm at our scale: 512px / 3mm ≈ 170.7 px/mm
PX_PER_MM = 512 / 3.0
GRID_ORIGIN = int(PX_PER_MM)  # 1mm from edge = start of center 1mm square
GRID_SIZE = int(PX_PER_MM)    # 1mm = ~171 pixels

# Line colors
LINE_THICK = 50      # Heavy border lines
LINE_MEDIUM = 80     # Medium subdivision lines
LINE_FINE = 110      # Fine subdivision lines
CHAMBER_BG = 185     # Glass background (slightly grayish)
CELL_LIVE = 220      # Live cell (bright, refractile)
CELL_DEAD_BF = 130   # Dead cell in BF (darker, trypan blue absorbed)
CELL_DEAD_TB = 200   # Dead cell in trypan blue channel (bright)


class HemocytometerSim:
    """Simulates a hemocytometer counting chamber.

    Parameters
    ----------
    n_cells : int
        Total number of cells in the chamber.
    viability : float
        Fraction of live cells (0.0-1.0).
    cell_radius_range : tuple
        (min, max) cell radius in pixels.
    dilution_factor : int
        Dilution factor used (e.g., 2 for 1:2 dilution).
    clump_fraction : float
        Fraction of cells that form clumps (0.0-1.0). When > 0, some
        cells are placed in tight clusters of 2-4 touching cells.
        Common artifact in real counting (under-trypsinized cultures).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_cells: int = 150,
        viability: float = 0.85,
        cell_radius_range: tuple = (3, 7),
        dilution_factor: int = 2,
        clump_fraction: float = 0.0,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.n_cells = n_cells
        self.viability = viability
        self.cell_radius_range = cell_radius_range
        self.dilution_factor = dilution_factor
        self.clump_fraction = clump_fraction

        # World dimensions
        self.width = 512
        self.height = 512
        self.viewport_width = 512
        self.viewport_height = 512

        # Camera / bridge attributes
        self.pixel_dtype = np.uint8
        self.camera_offset = [0, 0]
        self.objective_state = 0
        self.state_devices = {}
        self.mode = 0

        # Grid geometry: 3×3 grid of 1mm squares
        # The 4 corner 1mm squares are the standard counting areas
        # Center 1mm square has finer subdivisions (for platelet/RBC counting)
        self._mm_size = PX_PER_MM  # pixels per 1mm square

        # The 4 corner counting squares (each 1mm × 1mm = ~171 × 171 px)
        mm = self._mm_size
        self._corner_squares = [
            (0, 0, mm, mm),                        # top-left
            (2*mm, 0, 3*mm, mm),                   # top-right
            (0, 2*mm, mm, 3*mm),                   # bottom-left
            (2*mm, 2*mm, 3*mm, 3*mm),              # bottom-right
        ]

        # Generate cells
        self._cells = []
        self._generate_cells()

        # Pre-render
        self._bf_image = None
        self._tb_image = None  # trypan blue channel
        self._render_all()

    def _generate_cells(self):
        """Generate cell positions within the full ruled area."""
        self._cells = []
        min_r, max_r = self.cell_radius_range
        margin = 10

        # Determine how many cells go into clumps
        n_clumped = int(self.n_cells * self.clump_fraction)
        n_single = self.n_cells - n_clumped

        placed = 0

        def _place_cell(x, y, radius, is_clumped=False):
            nonlocal placed
            is_live = self.rng.random() < self.viability
            self._cells.append({
                "x": float(x), "y": float(y),
                "radius": float(radius),
                "is_live": is_live,
                "idx": placed,
                "brightness": self.rng.uniform(0.85, 1.0),
                "is_clumped": is_clumped,
            })
            placed += 1

        def _check_overlap(x, y, radius, min_gap=1.5):
            for c in self._cells:
                dist = np.sqrt((x - c["x"])**2 + (y - c["y"])**2)
                if dist < radius + c["radius"] + min_gap:
                    return True
            return False

        # Place single cells (with normal spacing)
        for _ in range(n_single * 30):
            if placed >= n_single:
                break
            x = self.rng.uniform(margin, self.width - margin)
            y = self.rng.uniform(margin, self.height - margin)
            radius = self.rng.uniform(min_r, max_r)
            if _check_overlap(x, y, radius):
                continue
            _place_cell(x, y, radius)

        # Place clumps (groups of 2-4 touching cells)
        clumped_so_far = 0
        for _ in range(n_clumped * 30):
            if clumped_so_far >= n_clumped:
                break
            # Place clump anchor
            x0 = self.rng.uniform(margin + 15, self.width - margin - 15)
            y0 = self.rng.uniform(margin + 15, self.height - margin - 15)
            r0 = self.rng.uniform(min_r, max_r)
            if _check_overlap(x0, y0, r0, min_gap=0.5):
                continue

            clump_size = self.rng.randint(2, 5)  # 2-4 cells per clump
            clump_size = min(clump_size, n_clumped - clumped_so_far)
            _place_cell(x0, y0, r0, is_clumped=True)
            clumped_so_far += 1

            # Add touching neighbors
            for _ in range(clump_size - 1):
                angle = self.rng.uniform(0, 2 * np.pi)
                ri = self.rng.uniform(min_r, max_r)
                # Place touching (gap = 0-1 px)
                gap = self.rng.uniform(0, 1.0)
                dist = r0 + ri + gap
                xi = x0 + dist * np.cos(angle)
                yi = y0 + dist * np.sin(angle)
                if (xi < margin or xi > self.width - margin or
                        yi < margin or yi > self.height - margin):
                    continue
                _place_cell(xi, yi, ri, is_clumped=True)
                clumped_so_far += 1

        self.n_cells = len(self._cells)

    def _render_all(self):
        """Pre-render both channels."""
        self._bf_image = self._render_bf()
        self._tb_image = self._render_trypan_blue()

    def _render_bf(self):
        """Render brightfield view with grid and cells.

        Rendering layers:
          1. Glass background with texture (scratches, dust specks, subtle gradient)
          2. Grid lines (heavy, medium, fine)
          3. Cells with phase-contrast appearance:
             - Live: bright interior, dark rim, outer bright halo, faint nucleus
             - Dead: darker fill with granular texture (trypan blue absorbed)
          4. Sensor noise
        """
        img = np.full((self.height, self.width), CHAMBER_BG, dtype=np.float32)

        # Glass texture: subtle Perlin-like gradient + micro-scratches
        rng_tex = np.random.RandomState(self.seed + 4000)
        # Low-frequency gradient (uneven illumination)
        Y, X = np.ogrid[:self.height, :self.width]
        ox, oy = rng_tex.uniform(-0.2, 0.2, 2)
        grad = 3.0 * np.sin(np.pi * (X / self.width + ox)) * \
               np.cos(np.pi * (Y / self.height + oy))
        img += grad.astype(np.float32)

        # Micro-scratches: thin faint lines on the glass
        n_scratches = rng_tex.randint(3, 8)
        scratch_layer = np.zeros((self.height, self.width), dtype=np.float32)
        for _ in range(n_scratches):
            sx = rng_tex.randint(0, self.width)
            sy = rng_tex.randint(0, self.height)
            angle = rng_tex.uniform(0, np.pi)
            length = rng_tex.randint(40, 200)
            ex = int(sx + length * np.cos(angle))
            ey = int(sy + length * np.sin(angle))
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.line(mask, (sx, sy), (ex, ey), 255, 1)
            scratch_layer -= mask.astype(np.float32) / 255.0 * rng_tex.uniform(1, 3)
        img += scratch_layer

        # Dust specks: tiny dark dots
        n_dust = rng_tex.randint(5, 15)
        dust_img = np.zeros((self.height, self.width), dtype=np.uint8)
        for _ in range(n_dust):
            dx, dy = rng_tex.randint(0, self.width), rng_tex.randint(0, self.height)
            dr = rng_tex.randint(1, 3)
            cv2.circle(dust_img, (dx, dy), dr, 255, -1)
        img -= dust_img.astype(np.float32) * rng_tex.uniform(3, 8)

        # Clip and draw grid
        img = np.clip(img, 0, 255).astype(np.uint8)
        self._draw_grid(img)

        # Draw cells with phase-contrast detail
        rng_render = np.random.RandomState(self.seed + 5000)
        for c in self._cells:
            x, y = int(round(c["x"])), int(round(c["y"]))
            r = max(2, int(round(c["radius"])))

            if c["is_live"]:
                # Phase contrast live cell:
                # Outer bright halo (phase ring artifact)
                cv2.circle(img, (x, y), r + 2,
                           min(255, int(CELL_LIVE * 1.08)), 1)
                # Dark trough between halo and cell body
                cv2.circle(img, (x, y), r + 1,
                           int(CELL_LIVE * 0.55), 1)
                # Cell body: bright center with radial gradient
                body_int = int(CELL_LIVE * c["brightness"])
                cv2.circle(img, (x, y), r, body_int, -1)
                # Slightly brighter center (cytoplasm refraction)
                if r >= 3:
                    cv2.circle(img, (x, y), max(1, r // 2),
                               min(255, body_int + 15), -1)
                # Faint nucleus dot (slightly darker)
                if r >= 4:
                    nuc_r = max(1, r // 3)
                    # Offset nucleus slightly
                    nx = x + rng_render.randint(-1, 2)
                    ny = y + rng_render.randint(-1, 2)
                    cv2.circle(img, (nx, ny), nuc_r,
                               max(0, body_int - 25), -1)
                # Dark rim (membrane)
                cv2.circle(img, (x, y), r, int(CELL_LIVE * 0.60), 1)
            else:
                # Dead cell: trypan blue absorbed — darker, granular fill
                base_int = int(CELL_DEAD_BF * c["brightness"])
                cv2.circle(img, (x, y), r, base_int, -1)
                # Granular texture inside dead cell
                if r >= 3:
                    n_granules = rng_render.randint(3, 8)
                    for _ in range(n_granules):
                        gx = x + rng_render.randint(-r + 1, r)
                        gy = y + rng_render.randint(-r + 1, r)
                        if (gx - x)**2 + (gy - y)**2 < r**2:
                            gint = base_int + rng_render.randint(-20, 10)
                            cv2.circle(img, (gx, gy), 1,
                                       max(0, min(255, gint)), -1)
                # Dark condensed nucleus
                if r >= 4:
                    cv2.circle(img, (x, y), max(1, r // 3),
                               max(0, base_int - 35), -1)
                # Darker rim
                cv2.circle(img, (x, y), r, int(CELL_DEAD_BF * 0.65), 1)

        # Sensor noise
        noise = rng_render.normal(0, 2.5, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def _render_trypan_blue(self):
        """Render trypan blue exclusion channel.

        Dead cells (trypan blue positive) appear bright with internal structure.
        Live cells are dim/invisible (dye excluded by intact membrane).
        Grid is faintly visible.
        Background has low-level blue-channel fluorescence.
        """
        # Dark background with subtle gradient
        rng_tb = np.random.RandomState(self.seed + 6000)
        Y, X = np.ogrid[:self.height, :self.width]
        bg_grad = 2.0 * np.sin(np.pi * X / self.width)
        img = np.full((self.height, self.width), 20, dtype=np.float32)
        img += bg_grad.astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Faint grid
        self._draw_grid(img, intensity_scale=0.3)

        for c in self._cells:
            x, y = int(round(c["x"])), int(round(c["y"]))
            r = max(2, int(round(c["radius"])))

            if not c["is_live"]:
                # Dead cell: bright (trypan blue accumulated inside)
                intensity = int(CELL_DEAD_TB * c["brightness"])
                # Outer ring slightly dimmer (membrane edge)
                cv2.circle(img, (x, y), r, int(intensity * 0.85), -1)
                # Brighter core (concentrated dye in cytoplasm)
                core_r = max(1, int(r * 0.7))
                cv2.circle(img, (x, y), core_r,
                           min(255, int(intensity * 1.05)), -1)
                # Very bright nucleus (condensed chromatin absorbs more)
                if r >= 4:
                    nuc_r = max(1, r // 3)
                    cv2.circle(img, (x, y), nuc_r,
                               min(255, int(intensity * 1.15)), -1)
            else:
                # Live cell: very faint outline (membrane intact, dye excluded)
                cv2.circle(img, (x, y), r, 32, -1)
                cv2.circle(img, (x, y), r, 38, 1)

        # Noise
        noise = rng_tb.normal(0, 3.0, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def _draw_grid(self, img, intensity_scale=1.0):
        """Draw the Neubauer counting grid on the image.

        Grid structure:
          - 3×3 mm ruled area = 3 large squares across
          - Heavy lines at 1mm boundaries
          - Corner squares: subdivided into 4×4 = 16 medium squares (0.25mm each)
          - Center square: subdivided into 5×5 = 25 group squares (0.2mm each),
            each further divided into 4×4 = 16 fine squares
          - Triple-ruled lines on 1mm boundaries (Neubauer feature)
        """
        mm = self._mm_size

        def _line(img, pt1, pt2, color, thickness):
            c = int(np.clip(color * intensity_scale + (1 - intensity_scale) * 20,
                            0, 255))
            cv2.line(img, pt1, pt2, c, thickness)

        # ── Heavy border lines: 3×3 mm grid ──
        for i in range(4):
            pos = int(i * mm)
            _line(img, (pos, 0), (pos, 511), LINE_THICK, 2)
            _line(img, (0, pos), (511, pos), LINE_THICK, 2)

        # Triple-ruled lines on 1mm boundaries
        for i in range(1, 3):
            pos = int(i * mm)
            for offset in [-2, 2]:
                _line(img, (pos + offset, 0), (pos + offset, 511), LINE_MEDIUM, 1)
                _line(img, (0, pos + offset), (511, pos + offset), LINE_MEDIUM, 1)

        # ── Corner squares: 4×4 subdivision (0.25mm each) ──
        sub_corner = mm / 4.0
        corners = [(0, 0), (2, 0), (0, 2), (2, 2)]  # grid positions
        for (gi, gj) in corners:
            ox = int(gi * mm)
            oy = int(gj * mm)
            for k in range(1, 4):
                sx = int(ox + k * sub_corner)
                sy = int(oy + k * sub_corner)
                _line(img, (sx, oy), (sx, int(oy + mm)), LINE_MEDIUM, 1)
                _line(img, (ox, sy), (int(ox + mm), sy), LINE_MEDIUM, 1)

        # ── Center square: 5×5 group squares (0.2mm each) ──
        cx0 = int(mm)
        cy0 = int(mm)
        group = mm / 5.0
        for i in range(1, 5):
            gx = int(cx0 + i * group)
            gy = int(cy0 + i * group)
            _line(img, (gx, cy0), (gx, int(cy0 + mm)), LINE_MEDIUM, 1)
            _line(img, (cx0, gy), (int(cx0 + mm), gy), LINE_MEDIUM, 1)
            # Triple-ruled on group boundaries
            for offset in [-1, 1]:
                _line(img, (gx + offset, cy0), (gx + offset, int(cy0 + mm)),
                      LINE_FINE, 1)
                _line(img, (cx0, gy + offset), (int(cx0 + mm), gy + offset),
                      LINE_FINE, 1)

        # ── Fine lines in center: each group → 4×4 ──
        small = group / 4.0
        for gi in range(5):
            for gj in range(5):
                gx0 = cx0 + gi * group
                gy0 = cy0 + gj * group
                for k in range(1, 4):
                    sx = int(gx0 + k * small)
                    sy = int(gy0 + k * small)
                    _line(img, (sx, int(gy0)), (sx, int(gy0 + group)),
                          LINE_FINE, 1)
                    _line(img, (int(gx0), sy), (int(gx0 + group), sy),
                          LINE_FINE, 1)

    def set_focal_plane(self, z):
        """No-op: hemocytometer is flat."""
        pass

    def _update_mode(self):
        """Update rendering mode from device state."""
        if "Channel" not in self.state_devices:
            return  # keep current mode when no devices are registered
        channel = self.state_devices["Channel"]
        label = channel.get("label", channel.get("Label", "brightfield"))
        if "trypan" in label.lower() or "blue" in label.lower():
            self.mode = 1
        else:
            self.mode = 0

    def snap_frame(self, **kwargs):
        """Return current frame based on mode."""
        self._update_mode()
        if self.mode == 1:
            return self._tb_image.copy()
        return self._bf_image.copy()

    def _cell_touches_line(self, cell, line_pos, axis):
        """Check if a cell's body overlaps a grid line.

        Parameters
        ----------
        cell : dict with x, y, radius
        line_pos : float — position of the grid line
        axis : 'x' or 'y' — which axis the line runs along

        Returns True if cell overlaps the line.
        """
        coord = cell["x"] if axis == "x" else cell["y"]
        return abs(coord - line_pos) < cell["radius"]

    def _count_in_square(self, sq, edge_rule=False):
        """Count cells within a counting square.

        Parameters
        ----------
        sq : tuple (x0, y0, x1, y1) — bounds of the square.
        edge_rule : bool
            If True, apply the hemocytometer edge exclusion rule:
            cells touching the TOP or LEFT boundary are included,
            cells touching the BOTTOM or RIGHT boundary are excluded.
            Cells fully inside are always counted.
        """
        x0, y0, x1, y1 = sq
        count = 0
        live = 0
        dead = 0
        for c in self._cells:
            cx, cy, r = c["x"], c["y"], c["radius"]

            # Check if cell center is within or near the square
            if edge_rule:
                # Cell must be within square or overlapping its edges
                if cx + r < x0 or cx - r > x1 or cy + r < y0 or cy - r > y1:
                    continue  # completely outside

                # Cell fully inside — always count
                inside = (cx - r >= x0 and cx + r <= x1 and
                          cy - r >= y0 and cy + r <= y1)
                if inside:
                    pass  # count it
                else:
                    # Cell overlaps a boundary — apply rule
                    touches_right = (cx + r > x1 and cx < x1)
                    touches_bottom = (cy + r > y1 and cy < y1)
                    touches_left = (cx - r < x0 and cx > x0)
                    touches_top = (cy - r < y0 and cy > y0)

                    # Exclude if touching bottom or right (and not rescued by top/left)
                    if touches_right or touches_bottom:
                        # Only exclude if NOT also touching top or left
                        if not (touches_top or touches_left):
                            continue
                    # Cell touching only top or left — include
                    # Cell completely outside on left/top side — skip
                    if cx < x0 - r or cy < y0 - r:
                        continue
            else:
                # Simple center-based counting
                if not (x0 <= cx < x1 and y0 <= cy < y1):
                    continue

            count += 1
            if c["is_live"]:
                live += 1
            else:
                dead += 1
        return count, live, dead

    def get_ground_truth(self):
        """Return cell counting data for grading.

        Includes both simple center-based counts and edge-rule counts.
        The edge exclusion rule is the standard hemocytometer protocol:
        cells touching the top or left boundary of a square are counted,
        cells touching the bottom or right boundary are excluded.
        """
        # Count cells in each corner square — both methods
        corner_counts = []
        corner_live = []
        corner_dead = []
        edge_rule_counts = []
        edge_rule_live = []
        edge_rule_dead = []

        for sq in self._corner_squares:
            total, live, dead = self._count_in_square(sq, edge_rule=False)
            corner_counts.append(total)
            corner_live.append(live)
            corner_dead.append(dead)

            total_e, live_e, dead_e = self._count_in_square(sq, edge_rule=True)
            edge_rule_counts.append(total_e)
            edge_rule_live.append(live_e)
            edge_rule_dead.append(dead_e)

        total_in_corners = sum(corner_counts)
        live_in_corners = sum(corner_live)
        dead_in_corners = sum(corner_dead)

        total_edge_rule = sum(edge_rule_counts)
        live_edge_rule = sum(edge_rule_live)
        dead_edge_rule = sum(edge_rule_dead)

        # Concentration calculation:
        # cells/mL = (total count / n_squares) × dilution × 10^4
        avg_per_square = total_in_corners / 4.0
        concentration = avg_per_square * self.dilution_factor * 1e4

        avg_edge = total_edge_rule / 4.0
        concentration_edge_rule = avg_edge * self.dilution_factor * 1e4

        # Viability from corner squares
        measured_viability = (live_in_corners / total_in_corners
                             if total_in_corners > 0 else 0)
        viability_edge = (live_edge_rule / total_edge_rule
                          if total_edge_rule > 0 else 0)

        n_clumped = sum(1 for c in self._cells if c.get("is_clumped"))

        # Identify cells on boundary lines for each square
        boundary_cells = []
        for sq in self._corner_squares:
            x0, y0, x1, y1 = sq
            for c in self._cells:
                cx, cy, r = c["x"], c["y"], c["radius"]
                touches = []
                if cx - r < x0 < cx + r:
                    touches.append("left")
                if cx - r < x1 < cx + r:
                    touches.append("right")
                if cy - r < y0 < cy + r:
                    touches.append("top")
                if cy - r < y1 < cy + r:
                    touches.append("bottom")
                if touches:
                    boundary_cells.append({
                        "idx": c["idx"], "x": round(c["x"], 1),
                        "y": round(c["y"], 1), "touches": touches,
                    })

        return {
            # Simple center-based counts (cell center in square)
            "corner_counts": corner_counts,
            "total_in_corners": total_in_corners,
            "live_in_corners": live_in_corners,
            "dead_in_corners": dead_in_corners,
            "concentration": round(concentration),
            "measured_viability": round(measured_viability, 3),
            # Edge-rule counts (standard hemocytometer protocol)
            "edge_rule_counts": edge_rule_counts,
            "total_edge_rule": total_edge_rule,
            "live_edge_rule": live_edge_rule,
            "dead_edge_rule": dead_edge_rule,
            "concentration_edge_rule": round(concentration_edge_rule),
            "viability_edge_rule": round(viability_edge, 3),
            # Metadata
            "dilution_factor": self.dilution_factor,
            "n_cells_total": self.n_cells,
            "n_clumped": n_clumped,
            "n_boundary_cells": len(boundary_cells),
            "boundary_cells": boundary_cells,
            "cell_positions": [
                {"x": round(c["x"], 1), "y": round(c["y"], 1),
                 "radius": round(c["radius"], 1), "is_live": c["is_live"],
                 "is_clumped": c.get("is_clumped", False)}
                for c in self._cells
            ],
            "corner_square_bounds": [
                {"x0": round(s[0], 1), "y0": round(s[1], 1),
                 "x1": round(s[2], 1), "y1": round(s[3], 1)}
                for s in self._corner_squares
            ],
        }
