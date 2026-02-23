"""Mitochondrial dynamics simulator.

A tubular network inside a single cell, with fission/fusion events.
World scale: 1 pixel = 0.1 µm (so 512 px = 51.2 µm).

At 10x: see the whole cell with mitochondrial network as bright mesh
At 20x: see half the cell, individual tubules start to resolve
At 40x: see detailed tubule structure, branch points, fragmented puncta

Channels:
  - BF (mode 0): cell outline, faint cytoplasm
  - Nucleus (mode 1): DAPI-stained nucleus (bright oval)
  - MitoTracker (mode 2): mitochondrial network (bright tubules)

Dynamics: fission (tubule splits) and fusion (tubules merge) events.
"""

import math
import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class MitoSim(SimBase):
    """Virtual mitochondrial network with fission/fusion dynamics."""

    continuous = True

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_tubules: int = 40,
        fragmentation: float = 0.0,
        fission_rate: float = 0.03,
        fusion_rate: float = 0.03,
        seed: int = 42,
        fixed_dt: float = 5.0,
        internal_scale: int = 4,
    ):
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=fixed_dt,
            auto_step=True, snaps_per_step=2,
            mode_map={
                ("SCFP2(434/474)", "UV"): 1,           # DAPI
                ("mScarlet3(569/582)", "ORANGE"): 2,   # MitoTracker
            },
        )

        self._noise_rng = np.random.default_rng(seed + 6666)

        # Optical pipelines per channel (higher PSF for organelle-level detail)
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 400),
            1: OpticalPipeline(
                psf_sigma=1.5, noise={"photon_scale": 3.0, "read_std": 3.5},
                vignette=0.12, rng_seed=seed + 401),
            2: OpticalPipeline(
                psf_sigma=1.8, noise={"photon_scale": 2.5, "read_std": 4.0},
                vignette=0.15, rng_seed=seed + 402),
        }

        # Dynamics
        self.fission_rate = fission_rate  # probability per tubule per step
        self.fusion_rate = fusion_rate    # probability per close pair per step
        self._step_count = 0
        self._cleanup_accum = 0.0  # time-based dead tubule cleanup
        self._event_log = []  # [(time, "fission"|"fusion", details)]

        # Cell geometry: circular cell centered in world
        self._cell_cx = world_size / 2
        self._cell_cy = world_size / 2
        self._cell_radius = world_size * 0.40  # cell fills ~80% of world
        self._nuc_cx = self._cell_cx
        self._nuc_cy = self._cell_cy
        self._nuc_radius = world_size * 0.10  # nucleus is ~20% of cell

        # Generate mitochondrial network
        self._tubules = []
        self._generate_network(n_tubules, fragmentation)

        # Membrane potential and drug response
        self._membrane_potential = 1.0  # normalized Δψ (1.0 = healthy, 0 = depolarized)
        self._base_fission_rate = fission_rate
        self._base_fusion_rate = fusion_rate
        self._drug_active = False
        self._drug_name = None
        self._drug_target_psi = 1.0  # target Δψ under drug
        self._drug_rate = 0.0  # how fast Δψ changes per step
        self._drug_washing_out = False
        self._drug_profiles = {
            "cccp": {"target_psi": 0.05, "onset_rate": 0.12, "washout_rate": 0.04},
            "fccp": {"target_psi": 0.02, "onset_rate": 0.15, "washout_rate": 0.03},
            "oligomycin": {"target_psi": 0.6, "onset_rate": 0.06, "washout_rate": 0.05},
            "rotenone": {"target_psi": 0.3, "onset_rate": 0.08, "washout_rate": 0.03},
        }

        # Cache rendered images
        self._dirty = True

    def _generate_network(self, n_tubules: int, fragmentation: float):
        """Generate mitochondrial network with branching junctions.

        First creates independent seed tubules, then adds branches that
        sprout from midpoints of existing tubules (Y-junctions). This
        produces a connected graph-like network rather than isolated lines.
        """
        self._tubules = []
        margin = self._nuc_radius + 10

        # Phase 1: seed tubules (60% of total)
        n_seeds = max(10, int(n_tubules * 0.6))
        for _ in range(n_seeds):
            for _attempt in range(50):
                tubule = self._random_tubule(margin, fragmentation)
                if tubule is not None:
                    self._tubules.append(tubule)
                    break

        # Phase 2: branch tubules from existing (40% of total)
        n_branches = n_tubules - n_seeds
        for _ in range(n_branches):
            if not self._tubules:
                break
            branch = self._branch_from_existing(fragmentation)
            if branch is not None:
                self._tubules.append(branch)

    def _random_tubule(self, margin: float, fragmentation: float):
        """Generate a single random tubule inside the cell, outside the nucleus."""
        cx, cy = self._cell_cx, self._cell_cy
        r_cell = self._cell_radius
        r_nuc = self._nuc_radius

        # Random start point in cytoplasm — biased toward perinuclear region
        # (real mito density is 2-4x higher near nucleus)
        for _ in range(20):
            angle = self.rng.uniform(0, 2 * math.pi)
            # Beta distribution peaks near nucleus (a=1.5, b=4.0)
            frac = self.rng.beta(1.5, 4.0)
            dist = r_nuc + 15 + frac * (r_cell - r_nuc - 25)
            x1 = cx + dist * math.cos(angle)
            y1 = cy + dist * math.sin(angle)
            if self._in_cytoplasm(x1, y1):
                break
        else:
            return None

        # Tubule length (shorter when fragmented)
        if self.rng.random() < fragmentation:
            length = self.rng.uniform(8, 25)  # short fragment
        else:
            length = self.rng.uniform(25, 90)  # normal tubule (up to 9 µm)

        # Direction: perinuclear tubules tend tangential, peripheral tend radial
        d_from_nuc = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        radial_angle = math.atan2(y1 - cy, x1 - cx)
        peri_frac = max(0, 1.0 - (d_from_nuc - r_nuc) / (r_cell - r_nuc))
        if peri_frac > 0.5:
            # Near nucleus: tangential
            direction = radial_angle + math.pi / 2 + self.rng.normal(0, 0.4)
        else:
            # Peripheral: radial or random
            direction = self.rng.uniform(0, 2 * math.pi)

        # Generate path with slight curvature
        n_points = max(3, int(length / 5))
        points = [(x1, y1)]
        x, y = x1, y1
        for i in range(1, n_points):
            direction += self.rng.normal(0, 0.3)  # slight turns
            step = length / n_points
            x += step * math.cos(direction)
            y += step * math.sin(direction)

            # Clip to cytoplasm
            dx, dy = x - cx, y - cy
            dist_c = math.sqrt(dx * dx + dy * dy)
            if dist_c > r_cell - 5:
                # Deflect inward
                x = cx + (r_cell - 8) * dx / dist_c
                y = cy + (r_cell - 8) * dy / dist_c
            elif dist_c < r_nuc + 8:
                # Deflect outward
                x = cx + (r_nuc + 12) * dx / max(dist_c, 1)
                y = cy + (r_nuc + 12) * dy / max(dist_c, 1)

            points.append((x, y))

        # Tubule properties
        width = self.rng.uniform(2.5, 6.0)  # 0.25-0.6 µm
        # Heterogeneous membrane potential: ~15% dim/depolarized
        if self.rng.random() < 0.15:
            brightness = self.rng.uniform(50, 100)  # dim (low Δψ)
        else:
            brightness = self.rng.uniform(130, 240)  # healthy (high Δψ)

        return {
            "points": points,
            "width": width,
            "brightness": brightness,
            "alive": True,
        }

    def _branch_from_existing(self, fragmentation: float):
        """Create a branch that sprouts from a midpoint of an existing tubule.

        Produces Y-junction topology: the branch starts at a point along
        an existing tubule and extends in a roughly perpendicular direction.
        """
        # Pick a random alive tubule with enough points
        alive = [t for t in self._tubules if t["alive"] and len(t["points"]) >= 4]
        if not alive:
            return None
        parent = self.rng.choice(alive)
        pts = parent["points"]

        # Pick a junction point (not at the ends)
        junc_idx = self.rng.integers(1, len(pts) - 1)
        jx, jy = pts[junc_idx]

        # Branch direction: roughly perpendicular to parent's local direction
        dx = pts[min(junc_idx + 1, len(pts) - 1)][0] - pts[max(0, junc_idx - 1)][0]
        dy = pts[min(junc_idx + 1, len(pts) - 1)][1] - pts[max(0, junc_idx - 1)][1]
        parent_angle = math.atan2(dy, dx)
        # Perpendicular ± some randomness
        branch_angle = parent_angle + self.rng.choice([-1, 1]) * (math.pi / 2 + self.rng.normal(0, 0.4))

        # Branch length (shorter than parent, 30-70% of typical)
        if self.rng.random() < fragmentation:
            length = self.rng.uniform(5, 15)
        else:
            length = self.rng.uniform(15, 50)

        n_points = max(2, int(length / 5))
        points = [(jx, jy)]
        x, y = jx, jy
        cx, cy = self._cell_cx, self._cell_cy
        r_cell = self._cell_radius
        r_nuc = self._nuc_radius
        direction = branch_angle

        for _ in range(1, n_points):
            direction += self.rng.normal(0, 0.3)
            step = length / n_points
            x += step * math.cos(direction)
            y += step * math.sin(direction)
            # Clip to cytoplasm
            dx_c, dy_c = x - cx, y - cy
            dist_c = math.sqrt(dx_c * dx_c + dy_c * dy_c)
            if dist_c > r_cell - 5:
                x = cx + (r_cell - 8) * dx_c / dist_c
                y = cy + (r_cell - 8) * dy_c / dist_c
            elif dist_c < r_nuc + 8:
                x = cx + (r_nuc + 12) * dx_c / max(dist_c, 1)
                y = cy + (r_nuc + 12) * dy_c / max(dist_c, 1)
            points.append((x, y))

        # Inherit parent's brightness ± some variation
        brightness = parent["brightness"] * self.rng.uniform(0.85, 1.15)
        brightness = np.clip(brightness, 40, 250)
        width = parent["width"] * self.rng.uniform(0.8, 1.1)

        return {
            "points": points,
            "width": width,
            "brightness": float(brightness),
            "alive": True,
        }

    def _in_cytoplasm(self, x, y):
        """Check if point is inside cell but outside nucleus."""
        dx_c = x - self._cell_cx
        dy_c = y - self._cell_cy
        dist_cell = math.sqrt(dx_c * dx_c + dy_c * dy_c)

        dx_n = x - self._nuc_cx
        dy_n = y - self._nuc_cy
        dist_nuc = math.sqrt(dx_n * dx_n + dy_n * dy_n)

        return dist_cell < self._cell_radius - 5 and dist_nuc > self._nuc_radius + 5

    # ----------------------------------------------------------------
    # Dynamics
    # ----------------------------------------------------------------

    # ── Membrane potential & drug response ──

    def apply_drug(self, drug_name: str, onset_rate: float = None):
        """Apply a mitochondrial drug that depolarizes membrane potential.

        Drugs reduce Δψ, which dims MitoTracker and increases fission rate:
          - CCCP/FCCP: protonophore, rapid severe depolarization (Δψ → ~0)
          - oligomycin: ATP synthase inhibitor, partial depolarization (Δψ → 0.6)
          - rotenone: Complex I inhibitor, moderate depolarization (Δψ → 0.3)
        """
        name = drug_name.lower()
        if name not in self._drug_profiles:
            raise ValueError(f"Unknown drug: {drug_name}. "
                             f"Available: {list(self._drug_profiles.keys())}")
        profile = self._drug_profiles[name]
        self._drug_active = True
        self._drug_name = name
        self._drug_target_psi = profile["target_psi"]
        self._drug_rate = onset_rate if onset_rate is not None else profile["onset_rate"]
        self._drug_washing_out = False

    def remove_drug(self, washout_rate: float = None):
        """Remove drug — membrane potential recovers gradually."""
        if not self._drug_active:
            return
        self._drug_washing_out = True
        if washout_rate is not None:
            self._drug_rate = washout_rate
        else:
            profile = self._drug_profiles.get(self._drug_name, {})
            self._drug_rate = profile.get("washout_rate", 0.04)
        self._drug_target_psi = 1.0  # recover to healthy

    def _update_membrane_potential(self):
        """Update Δψ toward target and adjust fission/fusion rates."""
        if not self._drug_active:
            return

        # Move Δψ toward target
        diff = self._drug_target_psi - self._membrane_potential
        self._membrane_potential += diff * self._drug_rate

        # Check if washout is complete
        if self._drug_washing_out and abs(self._membrane_potential - 1.0) < 0.01:
            self._membrane_potential = 1.0
            self._drug_active = False
            self._drug_name = None
            self._drug_washing_out = False
            self.fission_rate = self._base_fission_rate
            self.fusion_rate = self._base_fusion_rate
            return

        # Depolarization increases fission, decreases fusion
        psi = self._membrane_potential
        # Fission: up to 4x at full depolarization
        self.fission_rate = self._base_fission_rate * (1.0 + 3.0 * (1.0 - psi))
        # Fusion: drops to ~20% at full depolarization
        self.fusion_rate = self._base_fusion_rate * (0.2 + 0.8 * psi)

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 37.0
        return float(self.state_devices["Temperature"].get("label", "37"))

    def _temp_rate_factor(self) -> float:
        """Temperature-dependent rate factor for mitochondrial dynamics.

        Q10 ~ 2.0 for fission/fusion machinery. Optimal at 37°C.
        Cold arrest below 10°C, heat denaturation above 42°C.
        """
        temp = self._get_temperature()
        if temp < 10:
            return 0.05
        factor = 2.0 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.05, 1.0 - (temp - 42) * 0.3)
        return factor

    def _check_perfusion_drug(self):
        """Check Perfusion device state and toggle drug delivery accordingly.

        Only active when "Perfusion" is registered (real sim with bridge).
        Shadow sims (no bridge, empty state_devices) use apply_drug() directly.

        Perfusion state 4 ("Drug") → apply FCCP (mitochondrial uncoupler).
        Other states → wash out drug if active.
        """
        if "Perfusion" not in self.state_devices:
            return  # no device registered — respect direct apply_drug() calls
        perf = self.state_devices.get("Perfusion", {})
        if isinstance(perf, dict):
            label = perf.get("label", "Off")
        else:
            label = "Off"
        if label == "Drug":
            if not self._drug_active:
                self.apply_drug("cccp")
        else:
            if self._drug_active and not self._drug_washing_out:
                self.remove_drug()

    def get_network_state(self) -> dict:
        """Return current mitochondrial network fragmentation state.

        Returns metrics useful for ground truth and agent feedback:
          - n_fragments: number of alive tubule segments
          - mean_length: mean number of nodes per fragment
          - total_nodes: total connected nodes across all fragments
          - fragmentation_index: n_fragments / max(1, total_nodes/5)
              low → interconnected network; high → fragmented puncta
          - membrane_potential: current Δψ (1.0=healthy, 0=depolarized)
          - drug_active: whether mitochondrial drug is applied
        """
        alive = [t for t in self._tubules if t["alive"]]
        n_frags = len(alive)
        lengths = [len(t["points"]) for t in alive] if alive else [0]
        total_nodes = sum(lengths)
        mean_len = float(np.mean(lengths)) if alive else 0.0
        frag_idx = n_frags / max(1, total_nodes / 5)
        return {
            "n_fragments": n_frags,
            "mean_length": round(mean_len, 2),
            "total_nodes": total_nodes,
            "fragmentation_index": round(frag_idx, 3),
            "membrane_potential": round(self._membrane_potential, 3),
            "drug_active": self._drug_active,
            "drug_name": self._drug_name,
        }

    def step(self, dt: float = 1.0):
        """Advance dynamics by one time step: fission and fusion events."""
        if self.fixed_dt > 0:
            dt = self.fixed_dt
        self._step_count += 1
        self._time += dt

        # Z-drift accumulation (mechanical — not temperature-dependent)
        if self.z_drift_rate != 0 or self.z_drift_noise != 0:
            dz = self.z_drift_rate * dt + self.rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        # Temperature scaling for fission/fusion rates
        temp_factor = self._temp_rate_factor()

        # Perfusion-driven drug delivery (FCCP depolarizes Δψ → fission)
        self._check_perfusion_drug()

        # Update membrane potential and fission/fusion rates
        self._update_membrane_potential()

        alive = [t for t in self._tubules if t["alive"]]

        # Motility: subtle node jitter (thermal + cytoskeletal oscillation)
        jitter_scale = 0.3 * temp_factor * dt  # ~0.03 µm per step at 37°C
        for t in alive:
            for j in range(len(t["points"])):
                px, py = t["points"][j]
                px += self.rng.normal(0, jitter_scale)
                py += self.rng.normal(0, jitter_scale)
                # Keep inside cytoplasm
                if self._in_cytoplasm(px, py):
                    t["points"][j] = (px, py)

        # Fission: random tubule splits (rate scaled by temperature and dt)
        for t in alive:
            if len(t["points"]) < 4:
                continue
            if self.rng.random() < self.fission_rate * temp_factor * dt:
                self._do_fission(t)

        # Fusion: nearby endpoints merge (rate scaled by temperature and dt)
        alive = [t for t in self._tubules if t["alive"]]
        self._try_fusion(alive, rate_scale=temp_factor * dt)

        # Periodic cleanup: remove dead tubules to cap array growth
        self._cleanup_accum += dt
        if self._cleanup_accum >= 20.0:
            self._cleanup_accum = 0.0
            self._tubules = [t for t in self._tubules if t["alive"]]

        self._dirty = True

    def _do_fission(self, tubule):
        """Split a tubule at a random point, creating a gap."""
        pts = tubule["points"]
        if len(pts) < 4:
            return

        split_idx = self.rng.integers(2, len(pts) - 1)

        # Kill the original
        tubule["alive"] = False

        # Create two fragments
        frag1 = {
            "points": pts[:split_idx],
            "width": tubule["width"],
            "brightness": tubule["brightness"],
            "alive": True,
        }
        frag2 = {
            "points": pts[split_idx:],
            "width": tubule["width"],
            "brightness": tubule["brightness"],
            "alive": True,
        }

        if len(frag1["points"]) >= 2:
            self._tubules.append(frag1)
        if len(frag2["points"]) >= 2:
            self._tubules.append(frag2)

        self._event_log.append((self._time, "fission", {
            "position": list(pts[split_idx]),
        }))

    def _try_fusion(self, alive_tubules, rate_scale: float = 1.0):
        """Try to fuse nearby tubule endpoints."""
        if len(alive_tubules) < 2:
            return

        endpoints = []
        for i, t in enumerate(alive_tubules):
            pts = t["points"]
            endpoints.append((i, 0, pts[0]))   # start
            endpoints.append((i, -1, pts[-1]))  # end

        # Check all pairs for proximity
        fusion_dist = 12.0  # pixels (~1.2 µm)
        fused = set()

        for a in range(len(endpoints)):
            if endpoints[a][0] in fused:
                continue
            for b in range(a + 1, len(endpoints)):
                if endpoints[b][0] in fused:
                    continue
                if endpoints[a][0] == endpoints[b][0]:
                    continue

                x1, y1 = endpoints[a][2]
                x2, y2 = endpoints[b][2]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if dist < fusion_dist and self.rng.random() < self.fusion_rate * rate_scale:
                    # Merge the two tubules
                    i_a = endpoints[a][0]
                    i_b = endpoints[b][0]
                    t_a = alive_tubules[i_a]
                    t_b = alive_tubules[i_b]

                    # Orient points for merging
                    pts_a = list(t_a["points"])
                    pts_b = list(t_b["points"])

                    if endpoints[a][1] == 0:
                        pts_a = pts_a[::-1]
                    if endpoints[b][1] == -1:
                        pts_b = pts_b[::-1]

                    merged = {
                        "points": pts_a + pts_b,
                        "width": (t_a["width"] + t_b["width"]) / 2,
                        "brightness": (t_a["brightness"] + t_b["brightness"]) / 2,
                        "alive": True,
                    }

                    t_a["alive"] = False
                    t_b["alive"] = False
                    self._tubules.append(merged)
                    fused.add(i_a)
                    fused.add(i_b)

                    self._event_log.append((self._time, "fusion", {
                        "position": [(x1 + x2) / 2, (y1 + y2) / 2],
                    }))
                    break

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------

    def _render_bf_full(self) -> np.ndarray:
        """Render brightfield with phase-contrast-like appearance.

        Cell body is slightly darker than background with phase halo at edges.
        Nucleus appears as dark/refractile region. Mitochondria visible as
        faint dark filaments (refractive index difference).  Cytoplasmic
        texture from organelles.
        """
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        img = np.full((ih, iw), 200.0, dtype=np.float32)

        ccx, ccy = self._s(self._cell_cx), self._s(self._cell_cy)
        cr = self._s(self._cell_radius)
        ncx, ncy = self._s(self._nuc_cx), self._s(self._nuc_cy)
        nr = self._s(self._nuc_radius)

        # Cell body — slight darkening
        cv2.circle(img, (ccx, ccy), cr, 175.0, -1, lineType=cv2.LINE_AA)

        # Phase-contrast halo: bright ring outside cell edge
        cv2.circle(img, (ccx, ccy), cr + max(2, s * 2),
                   220.0, max(2, s * 2), lineType=cv2.LINE_AA)
        # Dark shade-off just inside cell edge
        cv2.circle(img, (ccx, ccy), cr - max(1, s),
                   165.0, max(2, s * 2), lineType=cv2.LINE_AA)

        # Cytoplasmic texture (granular organelle noise)
        yy, xx = np.mgrid[0:ih, 0:iw]
        dist_cell = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2).astype(
            np.float32
        )
        cyto_mask = dist_cell < cr - s * 2
        dist_nuc = np.sqrt((xx - ncx) ** 2 + (yy - ncy) ** 2).astype(
            np.float32
        )
        cyto_mask &= dist_nuc > nr + s * 2

        # Fine granular noise in cytoplasm
        texture = self._noise_rng.normal(0, 3, (ih // s, iw // s)).astype(
            np.float32
        )
        texture = cv2.GaussianBlur(texture, (5, 5), 1.0)
        if s > 1:
            texture = cv2.resize(texture, (iw, ih),
                                 interpolation=cv2.INTER_LINEAR)
        img[cyto_mask] += texture[cyto_mask]

        # Nucleus (dark, refractile with nucleolus-like features)
        cv2.circle(img, (ncx, ncy), nr, 135.0, -1, lineType=cv2.LINE_AA)
        # Nuclear envelope (slightly darker ring)
        cv2.circle(img, (ncx, ncy), nr, 120.0, max(1, s), lineType=cv2.LINE_AA)
        # Nucleolus (very dark spot)
        nuc_off = int(nr * 0.3)
        cv2.circle(img, (ncx + nuc_off, ncy - nuc_off),
                   max(2, nr // 4), 95.0, -1, lineType=cv2.LINE_AA)

        # Mitochondrial filaments — dark lines (phase-dense organelles)
        for t in self._tubules:
            if not t["alive"]:
                continue
            pts = np.array([(self._s(x), self._s(y)) for x, y in t["points"]],
                           dtype=np.int32)
            cv2.polylines(img, [pts], False, 155.0,
                          max(1, self._s(t["width"] * 0.4)),
                          lineType=cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_nuc_full(self) -> np.ndarray:
        """Render nucleus channel: bright DAPI nucleus with chromatin texture."""
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        img = np.full((ih, iw), 3.0, dtype=np.float32)

        ncx, ncy = self._s(self._nuc_cx), self._s(self._nuc_cy)
        nr = self._s(self._nuc_radius)

        # Nuclear body
        cv2.circle(img, (ncx, ncy), nr, 190.0, -1, lineType=cv2.LINE_AA)

        # Chromatin texture (heterochromatin foci = brighter DAPI spots)
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        n_foci = rng.integers(4, 10)
        for _ in range(n_foci):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0, nr * 0.7)
            fx = int(ncx + dist * math.cos(angle))
            fy = int(ncy + dist * math.sin(angle))
            fr = max(2, int(nr * rng.uniform(0.08, 0.18)))
            cv2.circle(img, (fx, fy), fr,
                       float(rng.uniform(210, 240)),
                       -1, lineType=cv2.LINE_AA)

        # Nuclear envelope (brighter rim)
        cv2.circle(img, (ncx, ncy), nr, 210.0,
                   max(1, s), lineType=cv2.LINE_AA)

        # Nucleolus (dark void in DAPI)
        nuc_off = int(nr * 0.3)
        cv2.circle(img, (ncx + nuc_off, ncy - nuc_off),
                   max(2, nr // 4), 50.0, -1, lineType=cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_mito_full(self) -> np.ndarray:
        """Render MitoTracker channel: bright tubular network.

        Brightness scales with membrane potential squared (Nernst equation:
        cationic dye accumulation proportional to delta-psi squared).

        Includes:
         - Perinuclear brightness gradient (mito density higher near nucleus)
         - Cytoplasmic background with granular texture (unbound dye + organelles)
         - Along-tubule intensity variation (cristae density gradients)
         - Punctate morphology for short fragments (post-fission swelling)
         - Higher OOF haze for widefield simulation
        """
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        img = np.zeros((ih, iw), dtype=np.float32)
        psi_scale = self._membrane_potential ** 2

        ccx, ccy = self._s(self._cell_cx), self._s(self._cell_cy)
        cr = self._s(self._cell_radius)
        ncx, ncy = self._s(self._nuc_cx), self._s(self._nuc_cy)
        nr = self._s(self._nuc_radius)

        # Cytoplasmic background with granular texture
        yy, xx = np.mgrid[0:ih, 0:iw]
        dist_cell = np.sqrt((xx - ccx) ** 2 + (yy - ccy) ** 2).astype(
            np.float32
        )
        dist_nuc = np.sqrt((xx - ncx) ** 2 + (yy - ncy) ** 2).astype(
            np.float32
        )
        cyto = (dist_cell < cr) & (dist_nuc > nr)
        # Perinuclear gradient: brighter background closer to nucleus
        peri_frac = np.clip(1.0 - (dist_nuc - nr) / (cr - nr), 0, 1)
        bg_level = 5 + 12 * peri_frac * psi_scale
        img[cyto] = bg_level[cyto]

        # Granular cytoplasmic texture (ER, vesicles, out-of-plane organelles)
        if not hasattr(self, '_cyto_texture'):
            rng_tex = np.random.default_rng(self.rng.integers(0, 2**31))
            raw = rng_tex.normal(0, 1.0, (ih // 4, iw // 4)).astype(np.float32)
            self._cyto_texture = cv2.resize(
                raw, (iw, ih), interpolation=cv2.INTER_LINEAR) * 6.0
        img[cyto] += self._cyto_texture[cyto]

        # Draw tubules
        for t in self._tubules:
            if not t["alive"]:
                continue

            pts = [(self._s(x), self._s(y)) for x, y in t["points"]]
            brightness = t["brightness"] * psi_scale

            # Thinner rendering — rely on PSF blur for apparent width
            raw_w = t["width"]
            # Swelling factor for low membrane potential (pre-fission)
            swell = 1.0 + 0.5 * (1.0 - self._membrane_potential)
            thickness = max(1, int(self._s(raw_w * 0.7 * swell)))

            # Perinuclear brightness boost for this tubule
            mid_idx = len(pts) // 2
            if mid_idx < len(pts):
                mx, my = pts[mid_idx]
                d_nuc = math.sqrt((mx - ncx) ** 2 + (my - ncy) ** 2)
                peri_boost = 1.0 + 0.4 * max(0, 1.0 - d_nuc / cr)
            else:
                peri_boost = 1.0

            # Short fragments → render as puncta (round spots)
            total_len = sum(
                math.sqrt((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)
                for i in range(len(pts)-1)
            ) if len(pts) > 1 else 0

            if len(pts) <= 2 and total_len < self._s(10):
                # Punctum: bright round spot (swollen fragment)
                px = int(np.mean([p[0] for p in pts]))
                py = int(np.mean([p[1] for p in pts]))
                punctum_r = max(2, int(thickness * 1.5 * swell))
                punctum_bright = min(255, int(brightness * peri_boost * 1.2))
                cv2.circle(img, (px, py), punctum_r,
                           float(punctum_bright), -1, cv2.LINE_AA)
            else:
                # Along-tubule intensity variation (cristae density)
                n_seg = len(pts) - 1
                # Random walk brightness modulation (smooth, ~20% amplitude)
                seg_mods = np.ones(n_seg)
                if n_seg > 1:
                    walk = np.cumsum(self._noise_rng.normal(0, 0.08, n_seg))
                    walk -= walk.mean()
                    seg_mods = np.clip(1.0 + walk, 0.6, 1.3)

                for i in range(n_seg):
                    seg_bright = int(
                        brightness * peri_boost * seg_mods[i]
                        * self._noise_rng.uniform(0.85, 1.0)
                    )
                    seg_bright = min(255, seg_bright)
                    cv2.line(img, pts[i], pts[i + 1], float(seg_bright),
                             thickness, lineType=cv2.LINE_AA)

        # PSF-like blur (tighter sigma for sub-resolution tubules)
        blur_k = max(3, s * 3) | 1
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0.9 * s)

        # Out-of-focus haze: diffuse glow from mito in other Z-planes
        # Higher for widefield (~25%) to match real OOF contribution
        haze = cv2.GaussianBlur(img, (0, 0), 8.0 * s)
        img = img + haze * 0.25

        return np.clip(img, 0, 255).astype(np.uint8)

    # ----------------------------------------------------------------
    # SimulationBridge interface
    # ----------------------------------------------------------------

    def _get_pad_bg(self) -> int:
        """Background value for out-of-bounds padding in ``_crop_fov``."""
        return 200 if self.mode == 0 else 0

    def _render_for_mode(self, mode: int) -> np.ndarray:
        """Return full-resolution image for the active channel."""
        if mode == 0:
            return self._render_bf_full()
        elif mode == 1:
            return self._render_nuc_full()
        elif mode == 2:
            return self._render_mito_full()
        elif mode in self._extra_channels:
            return self._extra_channels[mode]["image"].copy()
        else:
            return self._render_bf_full()

    def _finalize_output(self, viewport: np.ndarray) -> np.ndarray:
        """Convert viewport to grayscale; handle both 2D and 3D inputs."""
        if viewport.ndim == 3:
            return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)
        return viewport

    # ----------------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------------

    def get_ground_truth(self) -> dict:
        """Return current mitochondrial network state."""
        alive = [t for t in self._tubules if t["alive"]]

        # Total network length
        total_length = 0.0
        for t in alive:
            for i in range(len(t["points"]) - 1):
                x1, y1 = t["points"][i]
                x2, y2 = t["points"][i + 1]
                total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Fragmentation: ratio of short tubules
        short_count = sum(1 for t in alive if len(t["points"]) <= 3)
        frag_ratio = short_count / max(len(alive), 1)

        # Mean tubule length
        lengths = []
        for t in alive:
            tl = 0.0
            for i in range(len(t["points"]) - 1):
                x1, y1 = t["points"][i]
                x2, y2 = t["points"][i + 1]
                tl += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lengths.append(tl)

        gt = {
            "n_tubules": len(alive),
            "total_network_length_px": round(total_length, 1),
            "mean_tubule_length_px": round(float(np.mean(lengths)) if lengths else 0, 1),
            "fragmentation_ratio": round(frag_ratio, 3),
            "n_fission_events": sum(1 for e in self._event_log if e[1] == "fission"),
            "n_fusion_events": sum(1 for e in self._event_log if e[1] == "fusion"),
            "event_log": self._event_log.copy(),
            "time": self._time,
            "membrane_potential": round(self._membrane_potential, 3),
        }
        if self._drug_active:
            gt["drug"] = {
                "name": self._drug_name,
                "membrane_potential": round(self._membrane_potential, 3),
                "fission_rate": round(self.fission_rate, 4),
                "fusion_rate": round(self.fusion_rate, 4),
                "washing_out": self._drug_washing_out,
            }
        return gt

    def add_channel(self, name, image, mode_id):
        self._extra_channels[mode_id] = {"name": name, "image": image}

    def reset(self, seed=None):
        super().reset(seed)
        if seed is not None:
            self._noise_rng = np.random.default_rng(seed + 6666)
        self._step_count = 0
        self._event_log = []
