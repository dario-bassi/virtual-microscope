"""
Colony Counter Simulator — Bacterial colonies on agar plates.

Renders a top-down view of a petri dish with bacterial colonies.
Fundamentally different from microscopy backends: macroscopic imaging,
no stage movement, colony-level features instead of cell-level.

Applications:
  - Colony counting (transformation efficiency)
  - Blue/white screening (X-gal, β-galactosidase)
  - Antibiotic susceptibility (Kirby-Bauer disk diffusion)
  - Colony morphology classification

Devices: Camera + Channel. Single zoom level, whole-plate view.
World: 512×512 pixel camera image of a circular petri dish.

Plate types:
  - "spread": isolated colonies (dilution plating / transformation)
  - "streak": quadrant streak (isolation from mixed culture)
  - "lawn": confluent bacterial lawn (for disk diffusion / susceptibility)
"""

import numpy as np
import cv2
from typing import Optional


# ── Agar plate color constants ──────────────────────────────────────────
AGAR_LB = 200           # LB agar: pale beige (grayscale)
AGAR_XGAL = 195         # X-gal agar: slightly bluer
COLONY_WHITE = 240       # White colony (E. coli on LB)
COLONY_CREAM = 225       # Cream colony (slightly off-white)
COLONY_BLUE = 140        # Blue colony (β-gal+ on X-gal)
LAWN_COLOR = 215         # Bacterial lawn: slightly brighter than agar
LAWN_EDGE = 210          # Lawn at zone boundaries
PETRI_EDGE_GRAY = 160    # Plastic rim of petri dish
DISC_COLOR = 230         # Antibiotic paper disc
DISC_BORDER = 185        # Disc edge


class ColonySim:
    """Simulates a bacterial colony plate for top-down imaging.

    Parameters
    ----------
    n_colonies : int
        Number of colonies on the plate.
    plate_diameter : int
        Diameter of the petri dish in pixels (default 460 out of 512).
    colony_size_range : tuple
        (min_radius, max_radius) in pixels.
    plate_type : str
        "spread" (isolated colonies), "streak" (quadrant streak),
        "lawn" (confluent bacterial lawn for disk diffusion testing).
    staining : str or None
        None (plain LB), "xgal" (blue/white screening), "gfp" (GFP expression).
    blue_fraction : float
        Fraction of colonies that are blue (for X-gal). 0.0-1.0.
    zone_discs : list or None
        List of dicts with {x, y, radius, antibiotic, disc_radius} for
        zones of inhibition. radius = zone of inhibition radius in pixels.
    satellite_fraction : float
        Fraction of colonies that spawn 1-3 tiny satellite colonies nearby.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_colonies: int = 200,
        plate_diameter: int = 460,
        colony_size_range: tuple = (3, 15),
        plate_type: str = "spread",
        distribution: str = "random",
        staining: Optional[str] = None,
        blue_fraction: float = 0.3,
        zone_discs: Optional[list] = None,
        satellite_fraction: float = 0.0,
        dynamic: bool = False,
        growth_rate: float = 0.1,
        max_colony_radius: float = 20.0,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.n_colonies = n_colonies
        self.plate_diameter = plate_diameter
        self.plate_radius = plate_diameter // 2
        self.colony_size_range = colony_size_range
        self.plate_type = plate_type
        self.distribution = distribution  # "random", "clustered", "regular"
        self.staining = staining
        self.blue_fraction = blue_fraction
        self.zone_discs = zone_discs or []
        self.satellite_fraction = satellite_fraction

        # Dynamic growth settings
        self.dynamic = dynamic
        self.growth_rate = growth_rate  # base growth rate (px per step)
        self.max_colony_radius = max_colony_radius  # carrying capacity
        self._time_step = 0
        self.auto_step = False  # if True, step() called automatically per snap
        self.snaps_per_step = 1  # how many snaps before a step

        # World dimensions
        self.width = 512
        self.height = 512
        self.viewport_width = 512
        self.viewport_height = 512
        self.cx = self.width // 2
        self.cy = self.height // 2

        # Mode: 0=BF, 1=blue channel (X-gal), 2=GFP fluorescence
        self.mode = 0

        # Camera properties (match other backends)
        self.pixel_dtype = np.uint8
        self.camera_offset = [0, 0]
        self.objective_state = 0  # single zoom
        self.state_devices = {}   # populated by bridge.update_state()

        # Colony data
        self._colonies = []  # list of dicts
        self._generate_colonies()
        if self.satellite_fraction > 0:
            self._add_satellite_colonies()

        # Store initial radii for growth tracking
        if self.dynamic:
            for c in self._colonies:
                c["initial_radius"] = c["radius"]
                # Per-colony growth rate variation (±30%)
                c["growth_rate"] = self.growth_rate * self.rng.uniform(0.7, 1.3)

        # Snap counter for auto_step
        self._snap_count = 0

        # Pre-render plates
        self._bf_image = None
        self._blue_image = None
        self._gfp_image = None
        self._dirty = True
        self._render_all()

    def _generate_colonies(self):
        """Generate colony positions, sizes, and properties."""
        self._colonies = []

        if self.plate_type == "streak":
            self._generate_streak_colonies()
            return

        if self.plate_type == "lawn":
            self._generate_lawn_edge_colonies()
            return

        if self.distribution == "clustered":
            self._generate_clustered_colonies()
        elif self.distribution == "regular":
            self._generate_regular_colonies()
        else:
            self._generate_random_colonies()

    def _place_colony(self, x, y, idx):
        """Create a colony dict at (x, y) with random properties."""
        min_r, max_r = self.colony_size_range
        radius = min_r + self.rng.lognormal(0.5, 0.6)
        radius = np.clip(radius, min_r, max_r)
        is_blue = self.staining == "xgal" and self.rng.random() < self.blue_fraction
        has_gfp = self.staining == "gfp" and self.rng.random() < 0.5
        morphology = self._assign_morphology(radius)
        brightness = self.rng.uniform(0.85, 1.0)
        return {
            "x": float(x), "y": float(y), "radius": float(radius),
            "is_blue": is_blue, "has_gfp": has_gfp,
            "morphology": morphology, "brightness": brightness, "idx": idx,
        }

    def _check_overlap(self, x, y, radius):
        """Return True if (x, y, radius) overlaps existing colonies or zones."""
        for c in self._colonies:
            dist = np.sqrt((x - c["x"]) ** 2 + (y - c["y"]) ** 2)
            if dist < radius + c["radius"] + 2:
                return True
        for disc in self.zone_discs:
            dist = np.sqrt((x - disc["x"]) ** 2 + (y - disc["y"]) ** 2)
            if dist < disc["radius"]:
                return True
        return False

    def _in_plate(self, x, y, margin=5):
        """Return True if (x, y) is within the plate boundary."""
        return np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) < self.plate_radius - margin

    def _generate_random_colonies(self):
        """Random (Poisson-like) placement with rejection sampling."""
        min_r, max_r = self.colony_size_range
        max_attempts = self.n_colonies * 50
        placed = 0

        for _ in range(max_attempts):
            if placed >= self.n_colonies:
                break
            angle = self.rng.uniform(0, 2 * np.pi)
            r_frac = self.rng.beta(1.5, 1.5)
            r_pos = r_frac * (self.plate_radius - max_r - 5)
            x = self.cx + r_pos * np.cos(angle)
            y = self.cy + r_pos * np.sin(angle)

            col = self._place_colony(x, y, placed)
            if not self._check_overlap(x, y, col["radius"]):
                self._colonies.append(col)
                placed += 1

        self.n_colonies = len(self._colonies)

    def _generate_clustered_colonies(self):
        """Clustered placement — colonies near a few parent positions.

        Models nutrient hotspots or localized contamination.  Produces a
        Thomas cluster process: Poisson parent points, each surrounded by
        a Gaussian daughter scatter.
        """
        min_r, max_r = self.colony_size_range
        n_clusters = max(3, self.n_colonies // 15)
        cluster_sigma = 30.0  # px spread around each parent

        # Place cluster parents within plate
        parents = []
        for _ in range(n_clusters):
            angle = self.rng.uniform(0, 2 * np.pi)
            r_frac = self.rng.beta(2.0, 2.0)
            r_pos = r_frac * (self.plate_radius - 50)
            px = self.cx + r_pos * np.cos(angle)
            py = self.cy + r_pos * np.sin(angle)
            parents.append((px, py))

        placed = 0
        max_attempts = self.n_colonies * 80
        for _ in range(max_attempts):
            if placed >= self.n_colonies:
                break
            # Pick a random parent
            px, py = parents[self.rng.randint(0, len(parents))]
            x = px + self.rng.normal(0, cluster_sigma)
            y = py + self.rng.normal(0, cluster_sigma)

            if not self._in_plate(x, y, margin=max_r + 5):
                continue
            col = self._place_colony(x, y, placed)
            if not self._check_overlap(x, y, col["radius"]):
                self._colonies.append(col)
                placed += 1

        self.n_colonies = len(self._colonies)

    def _generate_regular_colonies(self):
        """Regular (inhibitory) placement — quasi-hexagonal spacing.

        Models competitive exclusion where colonies suppress neighbours
        via nutrient depletion.  Uses a jittered hexagonal grid, then
        prunes to fit the target count and plate boundary.
        """
        min_r, max_r = self.colony_size_range
        # Compute grid spacing from target count and plate area
        plate_area = np.pi * self.plate_radius ** 2
        area_per_colony = plate_area / max(1, self.n_colonies)
        spacing = np.sqrt(area_per_colony / 0.9)  # hex packing factor

        # Generate hex grid positions
        candidates = []
        row = 0
        y = self.cy - self.plate_radius + spacing * 0.5
        while y < self.cy + self.plate_radius:
            offset = (spacing * 0.5) if row % 2 else 0
            x = self.cx - self.plate_radius + offset + spacing * 0.5
            while x < self.cx + self.plate_radius:
                if self._in_plate(x, y, margin=max_r + 5):
                    # Small jitter for realism (±15% of spacing)
                    jx = x + self.rng.uniform(-0.15, 0.15) * spacing
                    jy = y + self.rng.uniform(-0.15, 0.15) * spacing
                    if self._in_plate(jx, jy, margin=max_r + 5):
                        candidates.append((jx, jy))
                x += spacing
            y += spacing * 0.866  # hex row height = spacing * sqrt(3)/2
            row += 1

        # Randomly subsample to target count
        self.rng.shuffle(candidates)
        candidates = candidates[:self.n_colonies]

        for idx, (x, y) in enumerate(candidates):
            col = self._place_colony(x, y, idx)
            # Override: regular plates tend to have more uniform sizes
            col["radius"] = float(np.clip(
                self.rng.normal((min_r + max_r) / 2, (max_r - min_r) * 0.15),
                min_r, max_r))
            self._colonies.append(col)

        self.n_colonies = len(self._colonies)

    def _generate_streak_colonies(self):
        """Generate colonies in quadrant streak pattern."""
        min_r, max_r = self.colony_size_range

        # 4 quadrants with decreasing density
        quadrant_angles = [
            (0, np.pi / 2),
            (np.pi / 2, np.pi),
            (np.pi, 3 * np.pi / 2),
            (3 * np.pi / 2, 2 * np.pi),
        ]
        densities = [0.6, 0.3, 0.15, 0.05]  # colonies per attempt

        for qi, (a_start, a_end) in enumerate(quadrant_angles):
            n_target = max(3, int(self.n_colonies * densities[qi]))
            placed = 0
            for _ in range(n_target * 20):
                if placed >= n_target:
                    break

                angle = self.rng.uniform(a_start, a_end)
                r_frac = self.rng.uniform(0.2, 0.9)
                r_pos = r_frac * (self.plate_radius - max_r - 5)
                x = self.cx + r_pos * np.cos(angle)
                y = self.cy + r_pos * np.sin(angle)

                radius = min_r + self.rng.exponential(1.5)
                radius = np.clip(radius, min_r, max_r * (1.0 - qi * 0.15))

                # Check overlap
                ok = True
                for c in self._colonies:
                    dist = np.sqrt((x - c["x"]) ** 2 + (y - c["y"]) ** 2)
                    if dist < radius + c["radius"] + 2:
                        ok = False
                        break
                if not ok:
                    continue

                is_blue = self.staining == "xgal" and self.rng.random() < self.blue_fraction
                has_gfp = self.staining == "gfp" and self.rng.random() < 0.5

                self._colonies.append({
                    "x": float(x),
                    "y": float(y),
                    "radius": float(radius),
                    "is_blue": is_blue,
                    "has_gfp": has_gfp,
                    "morphology": self._assign_morphology(radius),
                    "brightness": self.rng.uniform(0.85, 1.0),
                    "idx": len(self._colonies),
                })
                placed += 1

        self.n_colonies = len(self._colonies)

    def _generate_lawn_edge_colonies(self):
        """Generate visible individual colonies at zone boundaries.

        On lawn plates, the confluent growth is rendered as texture.
        At the edge of inhibition zones, individual colonies may be visible
        where growth is just barely possible (resistant mutants or MIC boundary).
        These are sparse, small colonies that help define the zone edge.
        """
        for disc in self.zone_discs:
            zone_r = disc["radius"]
            dx, dy = disc["x"], disc["y"]
            # Place 5-15 small colonies at the zone boundary
            n_edge = self.rng.randint(5, 16)
            for _ in range(n_edge * 5):
                if len([c for c in self._colonies
                        if np.sqrt((c["x"]-dx)**2 + (c["y"]-dy)**2) > zone_r - 5]) >= n_edge:
                    break
                angle = self.rng.uniform(0, 2 * np.pi)
                # Place just outside the zone radius (within 5px of boundary)
                r_pos = zone_r + self.rng.uniform(-3, 8)
                x = dx + r_pos * np.cos(angle)
                y = dy + r_pos * np.sin(angle)

                # Must be within dish
                if np.sqrt((x - self.cx)**2 + (y - self.cy)**2) > self.plate_radius - 5:
                    continue

                # Must not be inside another zone
                in_other_zone = False
                for other in self.zone_discs:
                    if other is disc:
                        continue
                    d = np.sqrt((x - other["x"])**2 + (y - other["y"])**2)
                    if d < other["radius"] - 3:
                        in_other_zone = True
                        break
                if in_other_zone:
                    continue

                radius = self.rng.uniform(1.5, 4.0)
                self._colonies.append({
                    "x": float(x), "y": float(y),
                    "radius": float(radius),
                    "is_blue": False, "has_gfp": False,
                    "morphology": "punctiform",
                    "brightness": self.rng.uniform(0.85, 1.0),
                    "idx": len(self._colonies),
                })
        self.n_colonies = len(self._colonies)

    def _add_satellite_colonies(self):
        """Add tiny satellite colonies near some existing colonies.

        Satellites are very small (1-2px) and appear 2-6px from parent.
        Common in some species (e.g., Proteus, Clostridium).
        """
        parents = [c for c in self._colonies
                   if self.rng.random() < self.satellite_fraction and c["radius"] > 4]
        for parent in parents:
            n_sat = self.rng.randint(1, 4)  # 1-3 satellites
            for _ in range(n_sat):
                angle = self.rng.uniform(0, 2 * np.pi)
                gap = parent["radius"] + self.rng.uniform(2, 6)
                sx = parent["x"] + gap * np.cos(angle)
                sy = parent["y"] + gap * np.sin(angle)

                # Check within dish
                dist_center = np.sqrt((sx - self.cx)**2 + (sy - self.cy)**2)
                if dist_center > self.plate_radius - 5:
                    continue

                # Check not in zone
                in_zone = False
                for disc in self.zone_discs:
                    d = np.sqrt((sx - disc["x"])**2 + (sy - disc["y"])**2)
                    if d < disc["radius"]:
                        in_zone = True
                        break
                if in_zone:
                    continue

                self._colonies.append({
                    "x": float(sx), "y": float(sy),
                    "radius": float(self.rng.uniform(1, 2.5)),
                    "is_blue": parent["is_blue"],
                    "has_gfp": parent["has_gfp"],
                    "morphology": "punctiform",
                    "brightness": parent["brightness"] * self.rng.uniform(0.8, 1.0),
                    "idx": len(self._colonies),
                    "is_satellite": True,
                })
        self.n_colonies = len(self._colonies)

    def _assign_morphology(self, radius):
        """Assign colony morphology based on size."""
        if radius < 5:
            return "punctiform"
        elif radius < 10:
            return "circular" if self.rng.random() < 0.7 else "irregular"
        else:
            choices = ["circular", "irregular", "spreading"]
            return self.rng.choice(choices, p=[0.3, 0.4, 0.3])

    def _render_all(self):
        """Pre-render all channels."""
        self._bf_image = self._render_bf()
        if self.staining == "xgal":
            self._blue_image = self._render_xgal()
        if self.staining == "gfp":
            self._gfp_image = self._render_gfp()

    def _render_bf(self):
        """Render brightfield (transmitted light) view of plate."""
        img = np.full((self.height, self.width), AGAR_LB, dtype=np.uint8)

        # Petri dish mask
        dish_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(dish_mask, (self.cx, self.cy), self.plate_radius, 255, -1)

        # Agar texture: grain + radial gradient (uneven pour)
        rng_tex = np.random.RandomState(self.seed + 9000)
        texture = rng_tex.normal(0, 2.5, (self.height, self.width))
        # Slight radial gradient — agar slightly thicker in center (brighter)
        Y, X = np.ogrid[:self.height, :self.width]
        rdist = np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2).astype(np.float32)
        rdist /= max(1, self.plate_radius)
        # Off-center pour: gradient shifted by a random offset
        ox = rng_tex.uniform(-0.15, 0.15)
        oy = rng_tex.uniform(-0.15, 0.15)
        pour_dist = np.sqrt(((X / self.width - 0.5 - ox)) ** 2 +
                            ((Y / self.height - 0.5 - oy)) ** 2)
        pour_gradient = -5.0 * pour_dist  # center brighter by ~3-5 gray levels
        texture += pour_gradient
        img = np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)

        # Bacterial lawn for "lawn" plate type (Kirby-Bauer background)
        if self.plate_type == "lawn":
            self._render_lawn(img, dish_mask)

        # Petri dish edge (plastic rim)
        cv2.circle(img, (self.cx, self.cy), self.plate_radius + 4, PETRI_EDGE_GRAY, 8)

        # Outside dish = dark (bench surface)
        outside = dish_mask == 0
        img[outside] = 40

        # Zone of inhibition discs (antibiotic paper)
        for disc in self.zone_discs:
            self._draw_disc(img, disc)

        # Draw individual colonies (for spread/streak plates)
        for c in self._colonies:
            self._draw_colony_bf(img, c)

        # Light vignetting
        Y, X = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((X - self.cx) ** 2 + (Y - self.cy) ** 2) / self.plate_radius
        vignette = 1.0 - 0.08 * dist ** 2
        img = np.clip(img.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

        return img

    def _render_lawn(self, img, dish_mask):
        """Render a confluent bacterial lawn over the agar.

        The lawn is a slightly brighter, textured layer over the agar.
        Clear zones around antibiotic discs show agar underneath.
        The zone boundary has a gradient (growth density fades near zone edge).
        """
        # Create lawn texture: stippled pattern
        rng_lawn = np.random.RandomState(self.seed + 9500)
        lawn = np.full_like(img, LAWN_COLOR, dtype=np.float32)
        # Fine stipple (colony-scale texture)
        stipple = rng_lawn.normal(0, 4.0, img.shape)
        lawn += stipple
        # Coarser growth variation (patches of denser/thinner growth)
        coarse = rng_lawn.normal(0, 1.0, (64, 64))
        coarse = cv2.resize(coarse, (self.width, self.height),
                            interpolation=cv2.INTER_LINEAR)
        lawn += coarse * 3.0

        # Build zone mask: 0 inside zone (clear), 1 outside (lawn grows)
        zone_mask = np.ones((self.height, self.width), dtype=np.float32)
        for disc in self.zone_discs:
            dx, dy = int(disc["x"]), int(disc["y"])
            zone_r = int(disc["radius"])
            # Create distance field from disc center
            Y, X = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((X - dx)**2 + (Y - dy)**2).astype(np.float32)
            # Smooth gradient at zone boundary (10px transition)
            transition = np.clip((dist - zone_r + 10) / 10.0, 0, 1)
            zone_mask = np.minimum(zone_mask, transition)

        # Apply lawn only inside dish, respecting zone clearings
        dish_f = (dish_mask / 255.0).astype(np.float32)
        lawn_layer = np.clip(lawn, 0, 255)
        agar_layer = img.astype(np.float32)
        # Blend: inside zone → agar, outside zone → lawn
        blended = agar_layer * (1 - zone_mask * dish_f) + lawn_layer * (zone_mask * dish_f)
        np.copyto(img, np.clip(blended, 0, 255).astype(np.uint8))

    def _draw_disc(self, img, disc):
        """Draw an antibiotic paper disc with realistic appearance."""
        dx, dy = int(disc["x"]), int(disc["y"])
        dr = int(disc.get("disc_radius", 6))
        # White paper disc with slight texture
        cv2.circle(img, (dx, dy), dr, DISC_COLOR, -1)
        cv2.circle(img, (dx, dy), dr, DISC_BORDER, 1)
        # Subtle center mark (where antibiotic was applied)
        cv2.circle(img, (dx, dy), max(1, dr // 3), DISC_COLOR - 10, -1)

    def _draw_colony_bf(self, img, colony):
        """Draw a single colony on the brightfield image."""
        x, y = int(colony["x"]), int(colony["y"])
        r = max(1, int(colony["radius"]))
        brightness = colony["brightness"]
        morphology = colony["morphology"]

        if colony["is_blue"]:
            base_val = int(COLONY_BLUE * brightness)
        else:
            # Ensure minimum contrast above agar background (~200)
            base_val = max(AGAR_LB + 15, int(COLONY_WHITE * brightness))

        # Shadow/edge ring: real colonies sit above agar and cast a slight shadow
        # This makes even small colonies visible against noisy background
        edge_val = max(0, AGAR_LB - 25)  # darker than agar
        edge_thick = 1 if r <= 4 else max(1, r // 5)
        cv2.circle(img, (x, y), r + edge_thick, edge_val, edge_thick)

        if morphology == "punctiform":
            # Tiny circular colony
            cv2.circle(img, (x, y), r, base_val, -1)
        elif morphology == "circular":
            # Smooth round colony with slight dome effect
            self._draw_dome_colony(img, x, y, r, base_val)
        elif morphology == "irregular":
            # Slightly irregular edges
            self._draw_irregular_colony(img, x, y, r, base_val, colony["idx"])
        elif morphology == "spreading":
            # Large with fuzzy edges
            self._draw_spreading_colony(img, x, y, r, base_val, colony["idx"])

    def _draw_dome_colony(self, img, x, y, r, base_val):
        """Draw a circular colony with dome (brighter center) effect."""
        # Use cv2 for fast rendering: draw filled circle then add bright center
        cv2.circle(img, (x, y), r, int(base_val * 0.88), -1)
        inner_r = max(1, int(r * 0.5))
        cv2.circle(img, (x, y), inner_r, min(255, int(base_val * 0.98)), -1)

    def _draw_irregular_colony(self, img, x, y, r, base_val, seed):
        """Draw a colony with slightly irregular edges."""
        rng = np.random.RandomState(seed + 70000)
        n_pts = 20
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        radii = r * (1.0 + rng.normal(0, 0.15, n_pts))
        radii = np.clip(radii, r * 0.7, r * 1.3)

        pts = np.array([
            [int(x + ri * np.cos(a)), int(y + ri * np.sin(a))]
            for a, ri in zip(angles, radii)
        ], dtype=np.int32)

        cv2.fillPoly(img, [pts], base_val)
        # Dome highlight in center
        cv2.circle(img, (x, y), max(1, r // 2), min(255, base_val + 10), -1)

    def _draw_spreading_colony(self, img, x, y, r, base_val, seed):
        """Draw a large spreading colony with fuzzy edges."""
        rng = np.random.RandomState(seed + 80000)

        # Dense center
        cv2.circle(img, (x, y), max(1, int(r * 0.6)), base_val, -1)

        # Radiating tendrils
        n_tendrils = rng.randint(5, 12)
        for _ in range(n_tendrils):
            angle = rng.uniform(0, 2 * np.pi)
            length = r * rng.uniform(0.5, 1.0)
            end_x = int(x + length * np.cos(angle))
            end_y = int(y + length * np.sin(angle))
            thickness = max(1, int(r * 0.15))
            cv2.line(img, (x, y), (end_x, end_y), base_val - 15, thickness)

    def _render_xgal(self):
        """Render X-gal blue/white screening channel.

        Returns image where blue colonies are bright, white colonies are dim.
        This is the "blue channel" — useful for distinguishing insert+ from insert-.
        """
        img = np.zeros((self.height, self.width), dtype=np.uint8)

        for c in self._colonies:
            x, y = int(c["x"]), int(c["y"])
            r = max(1, int(c["radius"]))
            if c["is_blue"]:
                # Blue colony: bright in blue channel
                intensity = int(200 * c["brightness"])
                cv2.circle(img, (x, y), r, intensity, -1)
            else:
                # White colony: dim in blue channel
                intensity = int(40 * c["brightness"])
                cv2.circle(img, (x, y), r, intensity, -1)

        return img

    def _render_gfp(self):
        """Render GFP fluorescence channel.

        Returns image where GFP+ colonies are bright.
        """
        img = np.zeros((self.height, self.width), dtype=np.uint8)

        for c in self._colonies:
            if not c["has_gfp"]:
                continue
            x, y = int(c["x"]), int(c["y"])
            r = max(1, int(c["radius"]))
            intensity = int(180 * c["brightness"])
            # Bright center, dimmer edge (simple two-circle approach)
            cv2.circle(img, (x, y), r + 1, max(0, intensity // 3), -1)
            cv2.circle(img, (x, y), r, intensity, -1)
            cv2.circle(img, (x, y), max(1, r // 2), min(255, int(intensity * 1.1)), -1)

        # Slight Gaussian blur for natural falloff
        img = cv2.GaussianBlur(img, (3, 3), 0.8)

        # Add faint read noise
        noise = self.rng.normal(0, 3, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def step(self, dt: float = 1.0):
        """Advance colony growth by one time step.

        Uses logistic growth: dr/dt = rate * r * (1 - r/r_max)
        Each colony grows independently toward max_colony_radius.
        After reaching max radius, colony edges fluctuate (morphology jitter).
        Only has effect when dynamic=True.
        """
        if not self.dynamic:
            return

        self._time_step += 1
        for c in self._colonies:
            r = c["radius"]
            r_max = self.max_colony_radius
            rate = c.get("growth_rate", self.growth_rate)

            if r < r_max * 0.98:
                # Active growth: logistic model
                dr = rate * r * (1.0 - r / r_max) * dt
                dr += self.rng.normal(0, abs(dr) * 0.05)
                c["radius"] = max(c.get("initial_radius", 1.0),
                                  min(r_max, r + dr))
            else:
                # Post-plateau: edge jitter (irregular growth front)
                jitter = self.rng.normal(0, 0.3) * dt
                c["radius"] = np.clip(r + jitter, r_max * 0.95, r_max * 1.05)

        self._dirty = True  # lazy re-render on next snap_frame()

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step (no SLM effects)."""
        self.step(dt)

    def get_growth_data(self):
        """Return colony size data for growth tracking.

        Returns dict with per-colony radii and time step.
        """
        return {
            "time_step": self._time_step,
            "radii": [round(c["radius"], 2) for c in self._colonies],
            "initial_radii": [round(c.get("initial_radius", c["radius"]), 2)
                              for c in self._colonies],
            "growth_rates": [round(c.get("growth_rate", self.growth_rate), 4)
                             for c in self._colonies],
        }

    def set_focal_plane(self, z):
        """No-op: colony counter has fixed focus."""
        pass

    def _update_mode(self):
        """Update rendering mode from device state (Channel device)."""
        channel = self.state_devices.get("Channel", {})
        label = channel.get("label", channel.get("Label", "transmitted"))
        if "transmitted" in label:
            self.mode = 0
        elif "blue" in label:
            self.mode = 1
        elif "GFP" in label:
            self.mode = 2

    def snap_frame(self, **kwargs):
        """Return current frame based on mode.

        If auto_step=True and dynamic=True, automatically calls step()
        every snaps_per_step snaps.
        """
        # Auto-step logic
        if self.auto_step and self.dynamic:
            if (self._snap_count > 0 and
                    self._snap_count % self.snaps_per_step == 0):
                self.step()
            self._snap_count += 1

        # Lazy re-render if dirty (dynamics changed state)
        if self._dirty:
            self._render_all()
            self._dirty = False

        self._update_mode()
        if self.mode == 0:
            return self._bf_image.copy()
        elif self.mode == 1:
            if self._blue_image is not None:
                return self._blue_image.copy()
            return np.zeros((self.height, self.width), dtype=np.uint8)
        elif self.mode == 2:
            if self._gfp_image is not None:
                return self._gfp_image.copy()
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return self._bf_image.copy()

    def get_ground_truth(self):
        """Return colony data for grading."""
        n_total = len(self._colonies)
        n_blue = sum(1 for c in self._colonies if c["is_blue"])
        n_white = n_total - n_blue
        n_gfp = sum(1 for c in self._colonies if c["has_gfp"])

        morphology_counts = {}
        for c in self._colonies:
            m = c["morphology"]
            morphology_counts[m] = morphology_counts.get(m, 0) + 1

        mean_radius = np.mean([c["radius"] for c in self._colonies]) if self._colonies else 0
        size_std = np.std([c["radius"] for c in self._colonies]) if self._colonies else 0

        # Zone data
        zone_data = []
        for disc in self.zone_discs:
            zone_data.append({
                "x": disc["x"],
                "y": disc["y"],
                "zone_diameter": disc["radius"] * 2,
                "antibiotic": disc.get("antibiotic", "unknown"),
            })

        return {
            "n_colonies": n_total,
            "n_blue": n_blue,
            "n_white": n_white,
            "n_gfp": n_gfp,
            "blue_fraction": round(n_blue / n_total, 3) if n_total > 0 else 0,
            "morphology_counts": morphology_counts,
            "mean_colony_radius": round(float(mean_radius), 1),
            "size_std": round(float(size_std), 1),
            "plate_type": self.plate_type,
            "distribution": getattr(self, "distribution", "random"),
            "staining": self.staining,
            "zones_of_inhibition": zone_data,
            "colony_positions": [
                {"x": round(c["x"], 1), "y": round(c["y"], 1),
                 "radius": round(c["radius"], 1), "is_blue": c["is_blue"],
                 "morphology": c["morphology"]}
                for c in self._colonies
            ],
        }
