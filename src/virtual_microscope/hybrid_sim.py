"""
Hybrid Simulator — Composite tissue + scattered overlays
=========================================================

Layers a VoronoiSim (or DynamicVoronoiSim) tissue background with
scattered overlay objects (floating cells, debris) at different Z depths.

The tissue is the base layer. Overlays are rendered on top with
depth-dependent defocus blur.

Usage
-----
    from voronoi_sim import VoronoiSim
    from hybrid_sim import HybridSim

    tissue = VoronoiSim(...)
    hybrid = HybridSim(tissue)

    # Add floating cells above the tissue
    hybrid.add_particles(
        positions=[(100, 200), (300, 400)],
        radii=[8, 12],
        z_depths=[5.0, 10.0],       # µm above tissue
        intensities=[0.8, 0.6],
        channels=["nucleus"],         # which channels they appear in
    )

    # Use hybrid like any other sim
    bridge = SimulationBridge(hybrid)
    _init_devices(core)
    ...
"""

import numpy as np
import cv2
from typing import Optional


class HybridSim:
    """Composites a tissue background with scattered overlay particles."""

    def __init__(self, tissue_sim):
        """
        Parameters
        ----------
        tissue_sim : VoronoiSim or DynamicVoronoiSim
            The tissue layer (base).
        """
        self._tissue = tissue_sim
        self._particles = []  # list of particle dicts
        self._next_particle_id = 0

        # Particle dynamics settings
        self.particle_motility = 0.0       # random walk magnitude (px/step)
        self.particle_chemotaxis = 0.0     # directed migration strength
        self.particle_target = None        # (x, y) chemotaxis target
        self.boundary_bias = 0.0           # attraction toward cell-cell junctions
        self._boundary_map = None          # cached gradient map from membrane
        self._particle_rng = np.random.default_rng(42)

        # Auto-stepping: particles advance automatically during snaps
        self.auto_step_particles = False
        self.particle_snaps_per_step = 2   # advance every N snaps
        self._particle_snap_count = 0

        # Proxy all standard properties from tissue sim
        self.width = tissue_sim.width
        self.height = tissue_sim.height
        self.viewport_width = tissue_sim.viewport_width
        self.viewport_height = tissue_sim.viewport_height

    # -- Particle management --

    def add_particles(self, positions, radii, z_depths=None,
                      intensities=None, channels=None, colors=None):
        """Add scattered particles as overlay.

        Parameters
        ----------
        positions : list of (x, y) — world coordinates
        radii : list of float — particle radii in pixels
        z_depths : list of float — µm above tissue (0 = in-focus with tissue)
        intensities : list of float — brightness 0..1
        channels : list of str or str — which channels particles appear in
            Options: "all", "brightfield", "nucleus", "membrane", or list per particle
        colors : list of (B, G, R) — color in each channel (default: white)
        """
        n = len(positions)
        if z_depths is None:
            z_depths = [0.0] * n
        if intensities is None:
            intensities = [0.8] * n
        if channels is None:
            channels = ["all"] * n
        elif isinstance(channels, str):
            channels = [channels] * n
        if colors is None:
            colors = [None] * n

        for i in range(n):
            self._particles.append({
                "id": self._next_particle_id,
                "pos": np.array(positions[i], dtype=float),
                "radius": float(radii[i]),
                "z": float(z_depths[i]),
                "intensity": float(intensities[i]),
                "channels": channels[i],
                "color": colors[i],
            })
            self._next_particle_id += 1

    def clear_particles(self):
        """Remove all overlay particles."""
        self._particles.clear()

    def get_particle(self, particle_id):
        """Get a particle by its ID."""
        for p in self._particles:
            if p["id"] == particle_id:
                return p
        return None

    def get_particle_positions(self):
        """Return array of current particle (x, y) positions."""
        return np.array([p["pos"].copy() for p in self._particles])

    def update_particle_positions(self, new_positions):
        """Update positions of existing particles.

        Parameters
        ----------
        new_positions : list of (x, y) or np.ndarray of shape (N, 2)
            New positions for each particle (same order as self._particles).
        """
        for i, pos in enumerate(new_positions):
            if i < len(self._particles):
                self._particles[i]["pos"] = np.array(pos, dtype=float)

    def _compute_boundary_map(self):
        """Compute gradient map from membrane image for boundary-guided migration.

        Uses the rendered membrane channel as a "road map" — bright pixels
        (cell boundaries) attract immune cells. Returns (grad_x, grad_y)
        pointing toward the nearest bright boundary.
        """
        mem = getattr(self._tissue, '_mem_full', None)
        if mem is None:
            return None

        # Convert to grayscale if needed
        if mem.ndim == 3:
            gray = cv2.cvtColor(mem, cv2.COLOR_BGR2GRAY)
        else:
            gray = mem.copy()

        # Blur to create smooth gradient field
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 8.0)

        # Compute spatial gradients (pointing toward bright = boundaries)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=5)

        # Normalize gradient magnitude
        mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        grad_x /= mag
        grad_y /= mag

        return grad_x, grad_y

    def step_particles(self, dt=1.0):
        """Advance particle positions by one timestep.

        Applies random motility, chemotaxis toward a target, and optional
        boundary-guided migration (particles navigate along cell-cell junctions).
        Clamps positions within world bounds.
        """
        margin = 10
        rng = self._particle_rng

        # Compute boundary gradient map if needed
        if self.boundary_bias > 0 and self._boundary_map is None:
            self._boundary_map = self._compute_boundary_map()

        for p in self._particles:
            # Random motility
            if self.particle_motility > 0:
                p["pos"] += rng.normal(0, self.particle_motility, 2) * dt

            # Chemotaxis toward target
            if self.particle_target is not None and self.particle_chemotaxis > 0:
                target = np.array(self.particle_target, dtype=float)
                direction = target - p["pos"]
                dist = np.linalg.norm(direction)
                if dist > 1:
                    direction /= dist
                    p["pos"] += direction * self.particle_chemotaxis * dt

            # Boundary-guided migration (attracted to cell-cell junctions)
            if self.boundary_bias > 0 and self._boundary_map is not None:
                gx, gy = self._boundary_map
                px, py = int(np.clip(p["pos"][0], 0, self.width - 1)), \
                         int(np.clip(p["pos"][1], 0, self.height - 1))
                p["pos"][0] += gx[py, px] * self.boundary_bias * dt
                p["pos"][1] += gy[py, px] * self.boundary_bias * dt

            # Clamp within bounds
            p["pos"][0] = np.clip(p["pos"][0], margin, self.width - margin)
            p["pos"][1] = np.clip(p["pos"][1], margin, self.height - margin)

    def add_random_debris(self, n, rng=None, channels=None,
                          radius_range=(2, 7), z_range=(3, 15),
                          intensity_range=(0.3, 0.9), margin=30):
        """Add n random debris particles.

        Convenience method for generating random floating debris.

        Parameters
        ----------
        n : int — number of debris particles
        rng : np.random.Generator — random number generator
        channels : str or list — which channels debris appears in (default: all)
        radius_range : tuple — (min, max) radius in pixels
        z_range : tuple — (min, max) Z depth in µm
        intensity_range : tuple — (min, max) brightness
        margin : int — pixels from edge to avoid
        """
        if rng is None:
            rng = np.random.default_rng()
        if channels is None:
            channels = "all"

        positions = [
            (rng.uniform(margin, self.width - margin),
             rng.uniform(margin, self.height - margin))
            for _ in range(n)
        ]
        radii = [rng.uniform(*radius_range) for _ in range(n)]
        z_depths = [rng.uniform(*z_range) for _ in range(n)]
        intensities = [rng.uniform(*intensity_range) for _ in range(n)]

        self.add_particles(positions, radii, z_depths, intensities,
                           channels=[channels] * n)
        return len(self._particles)

    # -- Interface proxied from tissue sim --

    @property
    def camera_offset(self):
        return self._tissue.camera_offset

    @camera_offset.setter
    def camera_offset(self, val):
        self._tissue.camera_offset = val

    @property
    def focal_plane(self):
        return self._tissue.focal_plane

    @focal_plane.setter
    def focal_plane(self, val):
        self._tissue.focal_plane = val

    @property
    def state_devices(self):
        return self._tissue.state_devices

    @state_devices.setter
    def state_devices(self, val):
        self._tissue.state_devices = val

    @property
    def mode(self):
        return self._tissue.mode

    @mode.setter
    def mode(self, val):
        self._tissue.mode = val

    @property
    def current_objectiv(self):
        return self._tissue.current_objectiv

    @current_objectiv.setter
    def current_objectiv(self, val):
        self._tissue.current_objectiv = val

    @property
    def _cells(self):
        return self._tissue._cells

    @property
    def centers(self):
        return self._tissue.centers

    @property
    def cell_centroids(self):
        return self._tissue.cell_centroids

    @property
    def nb_cells(self):
        return self._tissue.nb_cells

    @property
    def cell_areas(self):
        return self._tissue.cell_areas

    @property
    def has_nucleus_marker(self):
        return self._tissue.has_nucleus_marker

    @property
    def has_membrane_marker(self):
        return self._tissue.has_membrane_marker

    @property
    def tissue_z(self):
        return getattr(self._tissue, 'tissue_z', 0.0)

    def set_focal_plane(self, z):
        self._tissue.set_focal_plane(z)

    def get_ground_truth(self):
        gt = self._tissue.get_ground_truth()
        if self._particles:
            gt["overlay_particles"] = [
                {
                    "id": p["id"],
                    "x": round(float(p["pos"][0]), 1),
                    "y": round(float(p["pos"][1]), 1),
                    "radius": p["radius"],
                    "z": p["z"],
                    "channels": p["channels"],
                }
                for p in self._particles
            ]
            gt["n_overlay_particles"] = len(self._particles)
        return gt

    # -- Rendering --

    def _channel_matches(self, particle_channels, mode):
        """Check if a particle should appear in the current rendering mode."""
        if particle_channels == "all":
            return True
        mode_names = {0: "brightfield", 1: "nucleus", 2: "membrane"}
        current_name = mode_names.get(mode, f"extra_{mode}")
        if isinstance(particle_channels, list):
            return current_name in particle_channels
        return particle_channels == current_name

    def _defocus_sigma(self, z_offset):
        """Compute defocus blur sigma based on Z offset from focal plane.

        Uses depth-of-field model: sigma proportional to |z_offset| / DOF.
        """
        # DOF depends on objective
        obj = self.current_objectiv
        if obj == 40:
            dof = 1.5
        elif obj == 20:
            dof = 4.0
        else:
            dof = 6.0

        if abs(z_offset) < 0.1:
            return 0.0
        return abs(z_offset) / dof * 3.0  # 3px blur per DOF unit

    def _render_particles_on_viewport(self, viewport_img):
        """Render particles onto a viewport-space BGR image."""
        mode = self.mode
        cam_x, cam_y = self.camera_offset

        # Magnification crop params
        obj = self.current_objectiv
        vw, vh = self.viewport_width, self.viewport_height

        if obj == 20:
            scale = 2
        elif obj == 40:
            scale = 4
        else:
            scale = 1

        # Visible region in world coordinates
        if scale > 1:
            crop_w = vw // scale
            crop_h = vh // scale
            x0 = cam_x + (vw - crop_w) // 2
            y0 = cam_y + (vh - crop_h) // 2
        else:
            crop_w = vw
            crop_h = vh
            x0 = cam_x
            y0 = cam_y

        result = viewport_img.copy()
        tissue_z = getattr(self._tissue, 'tissue_z', 0.0)

        for p in self._particles:
            if not self._channel_matches(p["channels"], mode):
                continue

            # World position to viewport position
            wx, wy = p["pos"]
            vx = (wx - x0) * scale
            vy = (wy - y0) * scale

            # Compute defocus for this particle
            z_offset = p["z"] - (self.focal_plane - tissue_z)
            sigma = self._defocus_sigma(z_offset)

            # Effective radius includes defocus spread
            r_scaled = p["radius"] * scale
            effective_r = r_scaled + sigma * 3

            # Skip if outside viewport
            if (vx + effective_r < 0 or vx - effective_r >= vw or
                    vy + effective_r < 0 or vy - effective_r >= vh):
                continue

            # Particle intensity
            pix_intensity = int(p["intensity"] * 200)
            if p["color"] is not None:
                color = p["color"]
            else:
                color = (pix_intensity, pix_intensity, pix_intensity)

            # Render particle into a small local patch, then blur, then composite
            # This gives proper per-particle defocus
            margin = int(effective_r + 10)
            px, py = int(vx), int(vy)

            # Local patch bounds (in viewport coords)
            lx0 = max(0, px - margin)
            ly0 = max(0, py - margin)
            lx1 = min(vw, px + margin + 1)
            ly1 = min(vh, py + margin + 1)

            if lx1 <= lx0 or ly1 <= ly0:
                continue

            # Draw particle on local patch
            patch = np.zeros((ly1 - ly0, lx1 - lx0, 3), dtype=np.uint8)
            local_cx = px - lx0
            local_cy = py - ly0
            cv2.circle(patch, (local_cx, local_cy), max(1, int(r_scaled)),
                       color, -1, cv2.LINE_AA)

            # Apply per-particle defocus blur
            if sigma > 0.5:
                ksize = int(sigma * 4) | 1
                ksize = max(3, min(ksize, 31))
                patch = cv2.GaussianBlur(patch, (ksize, ksize), sigma)

            # Additive composite onto result
            result[ly0:ly1, lx0:lx1] = cv2.add(
                result[ly0:ly1, lx0:lx1], patch)

        return result

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs):
        """Render composite frame: tissue + overlay particles.

        Delegates to tissue sim's snap_frame, then overlays particles.
        Auto-steps particles if auto_step_particles is enabled.
        Returns grayscale uint8 like both backend sims.
        """
        # Auto-step particles
        if self.auto_step_particles and self._particles:
            self._particle_snap_count += 1
            if self._particle_snap_count % self.particle_snaps_per_step == 0:
                self.step_particles(dt=1.0)

        # Get tissue frame (grayscale)
        tissue_gray = self._tissue.snap_frame(
            mask=mask, exposure=exposure, intensity=intensity, **kwargs
        )

        if not self._particles:
            return tissue_gray

        # Convert to BGR for particle rendering
        if tissue_gray.ndim == 2:
            tissue_bgr = cv2.merge([tissue_gray, tissue_gray, tissue_gray])
        else:
            tissue_bgr = tissue_gray.copy()

        # Render particles on top
        composite_bgr = self._render_particles_on_viewport(tissue_bgr)

        # Scale by exposure/intensity (already done in tissue snap_frame,
        # so only apply to particle overlay)
        # Convert back to grayscale
        composite_gray = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2GRAY)
        return composite_gray

    def __getattr__(self, name):
        """Proxy any attribute not found here to the tissue sim."""
        if name.startswith('_') and name != '_cells':
            raise AttributeError(name)
        return getattr(self._tissue, name)
