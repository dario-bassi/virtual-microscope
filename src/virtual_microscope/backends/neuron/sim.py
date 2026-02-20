"""
NeuronSim — Cultured neuron simulation backend.

Simulates dissociated cortical neurons growing on a coverslip.
Neurons have branching dendritic arbors, axons, and synaptic puncta —
a fundamentally different morphology from compact cells or tissues.

Cell types:
  - Pyramidal: large soma, apical dendrite + basal arbor, long axon
  - Stellate: medium soma, radial dendrite arbor, shorter axon
  - Bipolar: small soma, two primary processes

Channels:
  - mode 0 (brightfield): Phase contrast — soma + thick processes visible
  - mode 1 (nucleus): MAP2 immunostaining (dendrite marker, green)
  - mode 2 (membrane): Synaptophysin puncta (synaptic marker, red)

Ground truth:
  - Neuron count, positions, types
  - Total neurite length per neuron
  - Branch points per neuron
  - Synapse (puncta) count and positions

Usage:
    sim = NeuronSim(n_neurons=8, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.optical_pipeline import OpticalPipeline


class NeuronSim(SimBase):
    """Cultured neuron simulation with branching morphology."""

    continuous = True

    def __init__(self, n_neurons=8, world_size=512, seed=42,
                 viewport_width=512, viewport_height=512,
                 neuron_types=None, density="medium",
                 internal_scale: int = 4):
        """
        Parameters:
            n_neurons: number of neurons in the field
            world_size: full image size
            seed: random seed
            neuron_types: list of types per neuron, or None for random mix
            density: "sparse" (3-5), "medium" (6-10), "dense" (12-20)
            internal_scale: render at world_size * scale internally (default 4)
        """
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=1.0,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
                ("mScarlet3(569/582)", "ORANGE"): 2,
            },
        )

        self.n_neurons = n_neurons
        self._seed = seed

        # Override DOF table entry for 20x
        self._dof_table[20] = 3.0

        # Optical pipelines per channel
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 600),
            1: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.5},
                vignette=0.10, rng_seed=seed + 601),
            2: OpticalPipeline(
                psf_sigma=0.5, noise={"photon_scale": 6.0, "read_std": 2.0},
                vignette=0.10, rng_seed=seed + 602),
        }

        self._rng = np.random.default_rng(seed)
        self._generate_neurons(neuron_types)

        # Pre-render all channels at internal resolution
        self._bf_full = self._render_brightfield()
        self._map2_full = self._render_map2()
        self._syn_full = self._render_synaptophysin()

        # Calcium activity (opt-in)
        self._calcium_enabled = False
        self._ca_base_dim = 1.0  # no dimming until calcium enabled
        self._ca_levels = np.zeros(self.n_neurons, dtype=np.float64)
        self._ca_time = 0
        self._ca_fire_log = []  # [(time, neuron_idx), ...]

        # Synaptic connectivity (opt-in via enable_synaptic_connections)
        self._synaptic_enabled = False
        self._synaptic_adj = {}       # {pre_idx: [(post_idx, weight), ...]}
        self._synaptic_delay = 0      # steps delay before post receives input
        self._synaptic_queue = []     # [(fire_time, post_idx, weight), ...]
        self._synaptic_fire_threshold = 0.5  # calcium level to trigger cascade
        self._synaptic_refractory = 5.0  # steps after firing before neuron can fire again
        self._ca_last_fire = np.full(self.n_neurons, -100.0)  # last fire time per neuron

        # SLM optogenetics (ChR2: light → neuron depolarization → calcium)
        self._slm_mask = None  # world-coordinate bool mask
        self._snap_counter_auto = 0

        # Drug response
        self._drug_active = False
        self._drug_name = None
        self._drug_effect = 0.0   # 0 = no effect, 1 = full effect
        self._drug_washing_out = False
        self._drug_profiles = {
            "ttx": {
                "target_effect": 1.0, "onset_rate": 0.12,
                "washout_rate": 0.06, "firing_mult": 0.0,
                "description": "sodium channel blocker — silences activity",
            },
            "glutamate": {
                "target_effect": 1.0, "onset_rate": 0.15,
                "washout_rate": 0.08, "firing_mult": 4.0,
                "description": "excitatory neurotransmitter — increases firing",
            },
            "gabazine": {
                "target_effect": 1.0, "onset_rate": 0.10,
                "washout_rate": 0.05, "firing_mult": 3.0,
                "description": "GABA antagonist — disinhibition, increases firing",
            },
            "kcl": {
                "target_effect": 1.0, "onset_rate": 0.20,
                "washout_rate": 0.10, "firing_mult": 8.0,
                "description": "depolarization — strong synchronized firing",
            },
        }

    def _pear_contour(self, cx, cy, r, apical_angle, n_pts=48):
        """Generate a pear-shaped contour for pyramidal neuron soma.

        The soma is wider at the basal end (opposite the apical dendrite)
        and narrower at the apical end, creating the characteristic
        triangular/pear shape of pyramidal neurons.

        Returns array of (x, y) points at internal-scale resolution.
        """
        s = self.internal_scale
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        # Pear modulation: narrower toward apical, wider at base
        # cos(θ - apical) = 1 at apical direction, -1 opposite
        # Subtract at apical end (narrow), add at basal end (wide)
        modulation = 1.0 - 0.2 * np.cos(angles - apical_angle)
        rx = r * s * modulation
        ry = r * s * 0.92 * modulation  # slight overall elongation
        pts = np.stack([
            cx * s + rx * np.cos(angles),
            cy * s + ry * np.sin(angles),
        ], axis=-1).astype(np.int32)
        return pts

    def _generate_neurons(self, neuron_types=None):
        """Generate neurons with branching morphology."""
        rng = self._rng
        margin = 60

        # Place soma positions (avoid overlap)
        self.neurons = []
        positions = []
        min_dist = 50  # minimum distance between somata

        for i in range(self.n_neurons):
            for _ in range(200):  # rejection sampling
                x = rng.uniform(margin, self.width - margin)
                y = rng.uniform(margin, self.height - margin)
                if all(np.hypot(x - px, y - py) > min_dist
                       for px, py in positions):
                    positions.append((x, y))
                    break
            else:
                # Fallback: place anyway
                x = rng.uniform(margin, self.width - margin)
                y = rng.uniform(margin, self.height - margin)
                positions.append((x, y))

        # Assign types
        type_options = ["pyramidal", "stellate", "bipolar"]
        type_weights = [0.5, 0.35, 0.15]

        for i, (sx, sy) in enumerate(positions):
            if neuron_types and i < len(neuron_types):
                ntype = neuron_types[i]
            else:
                ntype = rng.choice(type_options, p=type_weights)

            neuron = self._build_neuron(sx, sy, ntype, rng)
            self.neurons.append(neuron)

        # Generate dendritic spines (tiny protrusions along dendrites)
        self._generate_spines(rng)

        # Generate synaptic puncta along dendrites
        self._generate_puncta(rng)

    def _build_neuron(self, sx, sy, ntype, rng):
        """Build a single neuron with soma, dendrites, and axon."""
        neuron = {
            "type": ntype,
            "soma_x": sx,
            "soma_y": sy,
            "segments": [],  # list of (x0,y0,x1,y1,width,is_dendrite)
            "branch_points": [],
        }

        if ntype == "pyramidal":
            soma_r = rng.uniform(7, 10)
            neuron["soma_r"] = soma_r
            # Apical dendrite: upward, long, branches
            apical_angle = -np.pi/2 + rng.normal(0, 0.15)
            neuron["apical_angle"] = apical_angle  # for pear-shaped soma
            self._grow_dendrite(neuron, sx, sy, apical_angle,
                                length=rng.uniform(80, 140), width=2.5,
                                branch_prob=0.35, depth=0, max_depth=4, rng=rng,
                                soma_r=soma_r)
            # Basal dendrites: 3-5 radiating downward
            n_basal = rng.integers(3, 6)
            for j in range(n_basal):
                angle = np.pi/2 + rng.uniform(-0.8, 0.8) * np.pi / n_basal + j * np.pi / n_basal - np.pi/2
                self._grow_dendrite(neuron, sx, sy, angle,
                                    length=rng.uniform(40, 80), width=2.0,
                                    branch_prob=0.30, depth=0, max_depth=3, rng=rng,
                                    soma_r=soma_r)
            # Axon: one long thin process
            axon_angle = np.pi/2 + rng.normal(0, 0.3)
            self._grow_axon(neuron, sx, sy, axon_angle,
                            length=rng.uniform(100, 200), rng=rng,
                            soma_r=soma_r)

        elif ntype == "stellate":
            soma_r = rng.uniform(5, 8)
            neuron["soma_r"] = soma_r
            # Radial dendrites: 4-7 processes
            n_dend = rng.integers(4, 8)
            for j in range(n_dend):
                angle = 2 * np.pi * j / n_dend + rng.normal(0, 0.2)
                self._grow_dendrite(neuron, sx, sy, angle,
                                    length=rng.uniform(50, 100), width=2.0,
                                    branch_prob=0.25, depth=0, max_depth=3, rng=rng,
                                    soma_r=soma_r)
            # Axon
            axon_angle = rng.uniform(0, 2 * np.pi)
            self._grow_axon(neuron, sx, sy, axon_angle,
                            length=rng.uniform(80, 160), rng=rng,
                            soma_r=soma_r)

        elif ntype == "bipolar":
            soma_r = rng.uniform(4, 6)
            neuron["soma_r"] = soma_r
            # Two opposing primary processes
            angle1 = rng.uniform(0, np.pi)
            self._grow_dendrite(neuron, sx, sy, angle1,
                                length=rng.uniform(60, 120), width=2.0,
                                branch_prob=0.15, depth=0, max_depth=2, rng=rng,
                                soma_r=soma_r)
            self._grow_dendrite(neuron, sx, sy, angle1 + np.pi + rng.normal(0, 0.2),
                                length=rng.uniform(60, 120), width=2.0,
                                branch_prob=0.15, depth=0, max_depth=2, rng=rng,
                                soma_r=soma_r)
            # Short axon from soma
            axon_angle = angle1 + np.pi/2 + rng.normal(0, 0.3)
            self._grow_axon(neuron, sx, sy, axon_angle,
                            length=rng.uniform(40, 80), rng=rng,
                            soma_r=soma_r)

        return neuron

    def _grow_dendrite(self, neuron, x0, y0, angle, length, width,
                       branch_prob, depth, max_depth, rng,
                       soma_r=None):
        """Recursively grow a dendrite with random branching.

        For primary dendrites (depth=0), the start point is offset to the
        soma surface and the proximal segment is thicker (cone-shaped
        hillock), matching real neuron morphology where thick primary
        dendrites taper away from the soma.
        """
        if depth > max_depth or length < 8:
            return

        # Primary dendrites start at soma surface, not center
        if depth == 0 and soma_r is not None:
            x0 = x0 + soma_r * np.cos(angle)
            y0 = y0 + soma_r * np.sin(angle)
            # Proximal cone: 1.6x wider at soma junction, tapers over
            # first ~20px to normal width
            width *= 1.6

        # Grow in segments with slight curvature
        seg_len = min(length, rng.uniform(15, 30))
        remaining = length
        cx, cy = x0, y0
        # Track distance from soma for proximal taper
        dist_from_start = 0.0
        proximal_zone = 20.0  # world px over which hillock tapers

        while remaining > 5:
            seg = min(seg_len, remaining)
            # Slight random curvature
            angle += rng.normal(0, 0.15)
            nx = cx + seg * np.cos(angle)
            ny = cy + seg * np.sin(angle)

            # Clip to world bounds
            nx = np.clip(nx, 2, self.width - 2)
            ny = np.clip(ny, 2, self.height - 2)

            # Subtle varicosities: random thickness modulation ±15%
            seg_width = max(0.5, width * rng.uniform(0.85, 1.15))
            neuron["segments"].append(
                (cx, cy, nx, ny, seg_width, True))
            remaining -= seg
            dist_from_start += seg

            # Branch?
            if remaining > 15 and rng.random() < branch_prob:
                neuron["branch_points"].append((nx, ny))
                # Branch off
                branch_angle = angle + rng.choice([-1, 1]) * rng.uniform(0.3, 1.3)
                self._grow_dendrite(
                    neuron, nx, ny, branch_angle,
                    length=remaining * rng.uniform(0.4, 0.7),
                    width=width * 0.75,
                    branch_prob=branch_prob * 0.7,
                    depth=depth + 1, max_depth=max_depth, rng=rng)

            cx, cy = nx, ny

            # Taper: faster in proximal cone zone, normal elsewhere
            if depth == 0 and dist_from_start < proximal_zone:
                width *= 0.90  # fast taper from hillock
            else:
                width *= 0.95

    def _grow_axon(self, neuron, x0, y0, angle, length, rng,
                   soma_r=None):
        """Grow a thin axon with en passant varicosities (bouton swellings).

        Axons originate from the axon hillock — a slight cone-shaped
        thickening at the soma surface that tapers to normal axon width
        over the first ~15 world px.
        """
        # Start from soma surface
        if soma_r is not None:
            x0 = x0 + soma_r * np.cos(angle)
            y0 = y0 + soma_r * np.sin(angle)
        cx, cy = x0, y0
        remaining = length
        width = 2.0 if soma_r else 1.2  # hillock starts wider
        varicosity_interval = rng.uniform(12, 25)  # world px between boutons
        dist_since_varicosity = 0

        dist_from_start = 0.0
        hillock_zone = 15.0  # world px for axon hillock taper

        while remaining > 5:
            seg = min(rng.uniform(20, 40), remaining)
            angle += rng.normal(0, 0.2)  # more wandering than dendrites
            nx = cx + seg * np.cos(angle)
            ny = cy + seg * np.sin(angle)
            nx = np.clip(nx, 2, self.width - 2)
            ny = np.clip(ny, 2, self.height - 2)

            neuron["segments"].append(
                (cx, cy, nx, ny, width, False))  # is_dendrite=False

            # Track distance for varicosity placement
            dist_since_varicosity += seg
            if dist_since_varicosity >= varicosity_interval:
                if "varicosities" not in neuron:
                    neuron["varicosities"] = []
                neuron["varicosities"].append((nx, ny))
                dist_since_varicosity = 0
                varicosity_interval = rng.uniform(12, 25)

            remaining -= seg
            dist_from_start += seg
            cx, cy = nx, ny

            # Axon hillock taper: quickly narrow from 2.0 to normal 1.2
            if dist_from_start < hillock_zone and width > 1.2:
                width = max(1.2, width * 0.85)
            # Normal axon doesn't taper further

    def _generate_spines(self, rng):
        """Generate dendritic spines along dendrite segments.

        Real neurons have 1-10 spines per 10µm of dendrite. Spines are
        tiny protrusions (0.5-2µm) perpendicular to the shaft, with
        mushroom, thin, or stubby morphology. Visible at 40x/100x in
        both phase contrast and MAP2 fluorescence.
        """
        self.spines = []  # list of (x, y, head_r, type) in world coords

        for neuron in self.neurons:
            for (x0, y0, x1, y1, w, is_dend) in neuron["segments"]:
                if not is_dend:
                    continue
                seg_len = np.hypot(x1 - x0, y1 - y0)
                if seg_len < 3:
                    continue
                # Spine density: ~0.5-1.0 per world px of dendrite
                # (at our scale, world px ≈ µm at 10x)
                n_spines = rng.poisson(max(1, seg_len * 0.6))
                dx, dy = x1 - x0, y1 - y0
                norm = np.hypot(dx, dy) + 1e-6
                # Perpendicular direction
                px, py = -dy / norm, dx / norm
                for _ in range(n_spines):
                    t = rng.uniform(0.05, 0.95)
                    sx = x0 + t * dx + rng.choice([-1, 1]) * px * rng.uniform(1.0, 2.5)
                    sy = y0 + t * dy + rng.choice([-1, 1]) * py * rng.uniform(1.0, 2.5)
                    # Spine head radius (0.3-0.8 world px)
                    head_r = rng.uniform(0.3, 0.8)
                    # Morphology type
                    stype = rng.choice(["mushroom", "thin", "stubby"],
                                       p=[0.3, 0.5, 0.2])
                    self.spines.append((sx, sy, head_r, stype))

    def _generate_puncta(self, rng):
        """Generate synaptic puncta along dendrites."""
        self.puncta = []  # list of (x, y, brightness)

        for neuron in self.neurons:
            for (x0, y0, x1, y1, w, is_dendrite) in neuron["segments"]:
                if not is_dendrite:
                    continue
                # Puncta density: ~1 per 15-25px of dendrite
                seg_len = np.hypot(x1 - x0, y1 - y0)
                n_puncta = rng.poisson(seg_len / 20.0)
                for _ in range(n_puncta):
                    t = rng.uniform(0, 1)
                    px = x0 + t * (x1 - x0) + rng.normal(0, 1.5)
                    py = y0 + t * (y1 - y0) + rng.normal(0, 1.5)
                    brightness = rng.uniform(120, 240)
                    self.puncta.append((px, py, brightness))

    # ── Rendering ──

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs):
        """Capture a frame — compatible with SimulationBridge."""
        self._update_mode()
        self._update_objectif()

        # Map SLM mask to world coordinates (ChR2 optogenetics)
        if mask is not None and np.any(mask):
            self._slm_mask = self._map_slm_to_world(mask)
        else:
            self._slm_mask = None

        # Auto-step dynamics
        if self.auto_step and self._calcium_enabled:
            self._snap_counter_auto += 1
            if self._snap_counter_auto >= self.snaps_per_step:
                self._snap_counter_auto = 0
                self.step()

        # SLM optogenetic stimulation (ChR2) — observation-coupled effect.
        # Applied during snap (laser on during exposure), not during step().
        if self._slm_mask is not None and self._calcium_enabled:
            self._apply_slm_stimulation()

        self._snap_count += 1

        if self.mode == 0:
            full = self._bf_full
        elif self.mode == 1:
            full = self._map2_full
            # In calcium-imaging mode, dim the MAP2 base to model resting
            # GCaMP fluorescence (low baseline → high dynamic range)
            if self._calcium_enabled and self._ca_base_dim < 1.0:
                full = (full.astype(np.float32) * self._ca_base_dim).astype(np.uint8)
            # Overlay calcium transients on MAP2 channel
            if self._calcium_enabled and np.any(self._ca_levels > 0.05):
                full = self._overlay_calcium(full)
        elif self.mode == 2:
            full = self._syn_full
        elif self.mode in self._extra_channels:
            ch = self._extra_channels[self.mode]
            if ch.get("image") is not None:
                full = ch["image"]
            elif ch.get("render_fn"):
                full = ch["render_fn"]()
            else:
                full = self._bf_full
        else:
            full = self._bf_full

        crop = self._crop_fov(full)
        crop = self._apply_defocus(crop)
        crop = self._apply_pipeline(crop, exposure)

        return crop

    def _overlay_calcium(self, base_map2):
        """Add calcium transient glow to MAP2 image (non-destructive).

        GCaMP is cytoplasmic — excluded from the nucleus. Active neurons
        show a bright ring-like soma flash (dark nuclear hole) plus calcium
        that propagates along proximal dendrite paths with distance-dependent
        attenuation. This matches real GCaMP6f imaging where back-propagating
        APs produce visible dendritic calcium signals.
        """
        s = self.internal_scale
        img = base_map2.astype(np.float32)

        for i, neuron in enumerate(self.neurons):
            level = self._ca_levels[i]
            if level < 0.05:
                continue

            sx = int(round(self._sf(neuron["soma_x"])))
            sy = int(round(self._sf(neuron["soma_y"])))
            sr = int(round(self._sf(neuron["soma_r"])))

            added = self._ca_peak * level

            # ── Soma flash: cytoplasmic ring with nuclear exclusion ──
            glow_r = sr + max(2, int(3 * s))
            y0 = max(0, sy - glow_r)
            y1 = min(self._ih, sy + glow_r + 1)
            x0 = max(0, sx - glow_r)
            x1 = min(self._iw, sx + glow_r + 1)
            if y1 > y0 and x1 > x0:
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
                # Sharp-edged soma fill (not Gaussian — cytoplasm is uniform)
                soma_mask = dist_sq <= sr ** 2
                glow_patch = np.zeros_like(dist_sq, dtype=np.float32)
                glow_patch[soma_mask] = added
                # Nuclear exclusion
                nuc_r = sr * 0.5
                nuc_mask = dist_sq < (nuc_r ** 2)
                glow_patch[nuc_mask] *= 0.12
                # Slight glow just outside soma (cytoplasmic haze)
                outer = (dist_sq > sr ** 2) & (dist_sq < (sr * 1.4) ** 2)
                falloff = np.exp(-(dist_sq[outer] - sr ** 2) / (2 * (sr * 0.3) ** 2 + 1))
                glow_patch[outer] = added * 0.3 * falloff
                img[y0:y1, x0:x1] += glow_patch

            # ── Dendritic calcium propagation ──
            # Back-propagating APs spread calcium into proximal dendrites.
            # Intensity attenuates exponentially with path distance from soma.
            soma_wx = neuron["soma_x"]
            soma_wy = neuron["soma_y"]
            decay_length = 60.0  # world px — calcium fades over ~60px

            for (x0s, y0s, x1s, y1s, w, is_dend) in neuron["segments"]:
                if not is_dend:
                    continue
                # Distance of segment midpoint from soma
                mid_x = (x0s + x1s) / 2
                mid_y = (y0s + y1s) / 2
                dist = np.hypot(mid_x - soma_wx, mid_y - soma_wy)
                atten = np.exp(-dist / decay_length)
                if atten < 0.05:
                    continue
                seg_bright = added * atten * 0.7  # 70% of soma intensity
                thickness = max(2, int(w * 1.3 * s))
                ix0, iy0 = self._s(x0s), self._s(y0s)
                ix1, iy1 = self._s(x1s), self._s(y1s)
                cv2.line(img, (ix0, iy0), (ix1, iy1),
                         float(seg_bright), thickness, cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_brightfield(self):
        """Render phase contrast at internal resolution.

        Real phase contrast of cultured neurons shows:
        - Dark somata (phase-dense) with bright halo ring
        - Lighter nuclear region inside dark soma (shade-off)
        - Tiny dark nucleolus dot inside nuclear region
        - Dark processes with subtle bright edge halos
        - Faint flat glial patches in background
        - Small dark debris particles scattered across field
        """
        s = self.internal_scale
        lt = max(1, s // 2)
        rng = np.random.default_rng(self._seed + 777)
        img = np.full((self._ih, self._iw), 175, dtype=np.float32)

        # ── Faint glial cell patches (astrocytes under neurons) ──
        n_glia = rng.integers(3, 8)
        for _ in range(n_glia):
            gx = rng.integers(20 * s, self._iw - 20 * s)
            gy = rng.integers(20 * s, self._ih - 20 * s)
            gr = rng.integers(15 * s, 40 * s)
            # Irregular elliptical shape
            axes = (int(gr * rng.uniform(0.6, 1.0)),
                    int(gr * rng.uniform(0.6, 1.0)))
            angle = rng.uniform(0, 360)
            cv2.ellipse(img, (int(gx), int(gy)), axes, angle,
                        0, 360, 168.0, -1, cv2.LINE_AA)

        # ── Debris particles ──
        n_debris = rng.integers(20, 50)
        for _ in range(n_debris):
            dx = rng.integers(0, self._iw)
            dy = rng.integers(0, self._ih)
            dr = rng.integers(1, max(2, s))
            cv2.circle(img, (int(dx), int(dy)), dr, 130.0, -1)

        # ── Neurite processes with phase contrast halos ──
        for neuron in self.neurons:
            for (x0, y0, x1, y1, w, is_dend) in neuron["segments"]:
                ix0, iy0 = self._s(x0), self._s(y0)
                ix1, iy1 = self._s(x1), self._s(y1)
                thickness = max(lt, int(w * s))

                # Bright halo edges (draw wider bright line first)
                halo_t = thickness + max(2, s)
                cv2.line(img, (ix0, iy0), (ix1, iy1),
                         195.0, halo_t, cv2.LINE_AA)

                # Dark process core
                dark = 130.0 if is_dend else 148.0
                cv2.line(img, (ix0, iy0), (ix1, iy1),
                         dark, thickness, cv2.LINE_AA)

        # ── Axon varicosities: small dark swellings (en passant boutons) ──
        for neuron in self.neurons:
            for (vx, vy) in neuron.get("varicosities", []):
                ix, iy = self._s(vx), self._s(vy)
                vr = max(2, int(1.5 * s))
                if 0 <= ix < self._iw and 0 <= iy < self._ih:
                    cv2.circle(img, (ix, iy), vr + max(1, s // 2),
                               192.0, -1, cv2.LINE_AA)  # halo
                    cv2.circle(img, (ix, iy), vr,
                               135.0, -1, cv2.LINE_AA)  # dark bouton

        # ── Dendritic spines: tiny dark dots with halo at high magnification ──
        for (sx, sy, head_r, stype) in self.spines:
            ix, iy = self._s(sx), self._s(sy)
            ir = max(1, int(round(head_r * s)))
            if 0 <= ix < self._iw and 0 <= iy < self._ih:
                # Bright halo around spine head
                cv2.circle(img, (ix, iy), ir + max(1, s // 2), 190.0,
                           -1, cv2.LINE_AA)
                # Dark spine head (phase-dense)
                cv2.circle(img, (ix, iy), ir, 120.0, -1, cv2.LINE_AA)

        # ── Soma rendering with proper phase contrast ──
        for ni, neuron in enumerate(self.neurons):
            sx, sy = neuron["soma_x"], neuron["soma_y"]
            sr = neuron["soma_r"]
            isx = int(round(sx * s))
            isy = int(round(sy * s))
            isr = max(1, int(round(sr * s)))
            soma_rng = np.random.default_rng(self._seed + 3000 + ni)
            is_pyramidal = neuron["type"] == "pyramidal"

            if is_pyramidal and "apical_angle" in neuron:
                # Pear-shaped soma for pyramidal neurons
                apical = neuron["apical_angle"]
                contour = self._pear_contour(sx, sy, sr, apical)
                # Bright halo (draw enlarged contour)
                halo_contour = self._pear_contour(
                    sx, sy, sr + 1.5, apical)
                cv2.polylines(img, [halo_contour], True, 210.0,
                              max(2, int(1.2 * s)), cv2.LINE_AA)
                # Dark cytoplasm fill
                cv2.fillPoly(img, [contour], 85.0, cv2.LINE_AA)
                # Organellar texture
                mask_y0 = max(0, isy - isr - s)
                mask_y1 = min(self._ih, isy + isr + s + 1)
                mask_x0 = max(0, isx - isr - s)
                mask_x1 = min(self._iw, isx + isr + s + 1)
                if mask_y1 > mask_y0 and mask_x1 > mask_x0:
                    ph, pw = mask_y1 - mask_y0, mask_x1 - mask_x0
                    sm = np.zeros((ph, pw), dtype=np.uint8)
                    shifted = contour.copy()
                    shifted[:, 0] -= mask_x0
                    shifted[:, 1] -= mask_y0
                    cv2.fillPoly(sm, [shifted], 255)
                    soma_px = sm > 0
                    noise = soma_rng.normal(0, 3.5, (ph, pw)).astype(np.float32)
                    noise = cv2.GaussianBlur(noise, (0, 0), 0.6 * s)
                    img[mask_y0:mask_y1, mask_x0:mask_x1][soma_px] += noise[soma_px]
            else:
                # Elliptical soma for stellate/bipolar
                ecc = soma_rng.uniform(0.85, 1.0)
                angle = soma_rng.uniform(0, 360)
                axes_a = isr
                axes_b = max(1, int(isr * ecc))
                # Bright halo ring
                halo_a = axes_a + max(2, int(1.5 * s))
                halo_b = axes_b + max(2, int(1.5 * s))
                halo_w = max(2, int(1.2 * s))
                cv2.ellipse(img, (isx, isy), (halo_a, halo_b), angle,
                            0, 360, 210.0, halo_w, cv2.LINE_AA)
                # Dark cytoplasm fill
                cv2.ellipse(img, (isx, isy), (axes_a, axes_b), angle,
                            0, 360, 85.0, -1, cv2.LINE_AA)
                # Organellar texture
                mask_y0 = max(0, isy - isr - 1)
                mask_y1 = min(self._ih, isy + isr + 2)
                mask_x0 = max(0, isx - isr - 1)
                mask_x1 = min(self._iw, isx + isr + 2)
                if mask_y1 > mask_y0 and mask_x1 > mask_x0:
                    ph, pw = mask_y1 - mask_y0, mask_x1 - mask_x0
                    sm = np.zeros((ph, pw), dtype=np.uint8)
                    cv2.ellipse(sm, (isx - mask_x0, isy - mask_y0),
                                (axes_a, axes_b), angle, 0, 360, 255, -1)
                    soma_px = sm > 0
                    noise = soma_rng.normal(0, 3.5, (ph, pw)).astype(np.float32)
                    noise = cv2.GaussianBlur(noise, (0, 0), 0.6 * s)
                    img[mask_y0:mask_y1, mask_x0:mask_x1][soma_px] += noise[soma_px]

            # Nuclear region and nucleolus (same for all types)
            nuc_r = max(1, int(sr * s * 0.55))
            cv2.circle(img, (isx, isy), nuc_r, 105.0, -1, cv2.LINE_AA)

            nucl_r = max(1, int(sr * s * 0.15))
            nucl_off = max(1, int(nuc_r * 0.25))
            nucl_angle = soma_rng.uniform(0, 2 * np.pi)
            nucl_dx = int(nucl_off * np.cos(nucl_angle))
            nucl_dy = int(nucl_off * np.sin(nucl_angle))
            cv2.circle(img, (isx + nucl_dx, isy + nucl_dy),
                       nucl_r, 55.0, -1, cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_map2(self):
        """Render MAP2 immunostaining at internal resolution.

        MAP2 is a dendritic marker: bright soma + dendrites, dim axons.
        Proximal dendrites brighter than distal (MAP2 accumulates near soma).
        Branch points show bright hotspots (microtubule intersections).
        Faint out-of-focus haze from processes in other Z-planes.
        """
        s = self.internal_scale
        lt = max(1, s // 2)
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        img += 4.0

        for neuron in self.neurons:
            sx, sy = self._sf(neuron["soma_x"]), self._sf(neuron["soma_y"])
            sr = self._sf(neuron["soma_r"])
            isx, isy = int(round(sx)), int(round(sy))

            for (x0, y0, x1, y1, w, is_dend) in neuron["segments"]:
                if is_dend:
                    thickness = max(lt, int(w * 1.2 * s))
                    # Proximal-distal gradient: brighter near soma
                    midx = (x0 + x1) / 2
                    midy = (y0 + y1) / 2
                    dist = np.hypot(midx - neuron["soma_x"],
                                    midy - neuron["soma_y"])
                    prox_frac = max(0.15, 1.0 - dist / 120.0)
                    brightness = (140.0 + 60.0 * min(w / 2.5, 1.0)) * prox_frac
                    cv2.line(img, (self._s(x0), self._s(y0)),
                             (self._s(x1), self._s(y1)),
                             brightness, thickness, cv2.LINE_AA)
                else:
                    thickness = max(lt, int(w * 0.8 * s))
                    cv2.line(img, (self._s(x0), self._s(y0)),
                             (self._s(x1), self._s(y1)),
                             20.0, thickness, cv2.LINE_AA)

            # Soma: bright fill with dark nuclear hole (MAP2 is cytoplasmic)
            is_pyramidal = neuron["type"] == "pyramidal"
            if is_pyramidal and "apical_angle" in neuron:
                contour = self._pear_contour(
                    neuron["soma_x"], neuron["soma_y"],
                    neuron["soma_r"], neuron["apical_angle"])
                cv2.fillPoly(img, [contour], 190.0, cv2.LINE_AA)
            else:
                cv2.circle(img, (isx, isy),
                           max(1, int(round(sr))), 190.0, -1)
            # Dark nuclear hole (MAP2 is excluded from nucleus)
            nuc_r = max(1, int(sr * 0.5))
            cv2.circle(img, (isx, isy), nuc_r, 35.0, -1)

            # Branch point hotspots
            for bx, by in neuron["branch_points"]:
                bp_x, bp_y = self._s(bx), self._s(by)
                bp_r = max(2, int(2.5 * s))
                cv2.circle(img, (bp_x, bp_y), bp_r, 220.0, -1, cv2.LINE_AA)

        # ── Dendritic spines: bright dots along dendrites ──
        for (sx, sy, head_r, stype) in self.spines:
            ix, iy = self._s(sx), self._s(sy)
            ir = max(1, int(round(head_r * s)))
            if 0 <= ix < self._iw and 0 <= iy < self._ih:
                brightness = 160.0 if stype == "mushroom" else 110.0
                cv2.circle(img, (ix, iy), ir, brightness, -1, cv2.LINE_AA)

        # Dead cell debris: bright round pyknotic cells (common in culture)
        rng_debris = np.random.default_rng(self._seed + 888)
        n_dead = rng_debris.integers(2, 6)
        for _ in range(n_dead):
            dx = rng_debris.integers(10 * s, self._iw - 10 * s)
            dy = rng_debris.integers(10 * s, self._ih - 10 * s)
            dr = rng_debris.integers(3 * s, 7 * s)
            cv2.circle(img, (int(dx), int(dy)), int(dr),
                       rng_debris.uniform(80, 180), -1, cv2.LINE_AA)

        # Out-of-focus haze from neuropil in other Z-planes
        haze = cv2.GaussianBlur(img, (0, 0), 6.0 * s)
        img = np.clip(img + haze * 0.20, 0, 255)

        # Subtle autofluorescence background texture (culture substrate)
        rng_bg = np.random.default_rng(self._seed + 777)
        bg_noise = rng_bg.normal(0, 2.0, size=img.shape).astype(np.float32)
        bg_noise = cv2.GaussianBlur(bg_noise, (0, 0), 3.0 * s)
        img = np.clip(img + bg_noise, 0, 255)

        return img.astype(np.uint8)

    def _render_synaptophysin(self):
        """Render synaptophysin puncta at internal resolution."""
        s = self.internal_scale
        lt = max(1, s // 2)
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        img += 3.0

        for (px, py, brightness) in self.puncta:
            x, y = self._s(px), self._s(py)
            if 0 <= x < self._iw and 0 <= y < self._ih:
                cv2.circle(img, (x, y), lt, float(brightness), -1)

        for neuron in self.neurons:
            for (x0, y0, x1, y1, w, is_dend) in neuron["segments"]:
                if is_dend:
                    cv2.line(img, (self._s(x0), self._s(y0)),
                             (self._s(x1), self._s(y1)),
                             8.0, lt, cv2.LINE_AA)

        # Axon varicosities: bright en passant boutons in synaptophysin
        for neuron in self.neurons:
            for (vx, vy) in neuron.get("varicosities", []):
                ix, iy = self._s(vx), self._s(vy)
                vr = max(2, int(1.5 * s))
                if 0 <= ix < self._iw and 0 <= iy < self._ih:
                    cv2.circle(img, (ix, iy), vr, 200.0, -1, cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def enable_calcium_activity(self, frequency=0.08, decay_rate=0.25,
                                peak_intensity=220, spread_radius=25,
                                base_dim=0.15):
        """Enable GCaMP-like calcium transients (visible in MAP2 channel).

        Each neuron fires independently with Poisson probability per step.
        Firing produces a bright calcium flash at the soma that decays
        exponentially over subsequent steps and spreads into proximal dendrites.

        Parameters:
            frequency: probability that an idle neuron fires per step
            decay_rate: exponential decay per step (0.25 = ~3 step half-life)
            peak_intensity: max added brightness at soma on firing (0-255)
            spread_radius: how far calcium glow extends from soma (world px)
            base_dim: dim the MAP2 baseline by this factor (0-1). Lower values
                model resting GCaMP (dim baseline → high dynamic range for
                transients). Default 0.15 = very dim baseline, matching real
                GCaMP6f (>50-fold dynamic range).
        """
        self._calcium_enabled = True
        self._ca_frequency = frequency
        self._ca_decay = decay_rate
        self._ca_peak = peak_intensity
        self._ca_spread = spread_radius
        self._ca_base_dim = max(0.05, min(1.0, base_dim))
        self._ca_levels = np.zeros(self.n_neurons, dtype=np.float64)
        self._ca_time = 0
        self._ca_fire_log = []
        self._ca_rng = np.random.default_rng(self._seed + 5000)

    def enable_synaptic_connections(self, connection_prob=0.3, max_distance=200,
                                       weight=0.6, delay=1,
                                       inhibitory_fraction=0.0,
                                       refractory=3.0):
        """Enable synaptic connectivity between neurons.

        Creates a distance-dependent random connectivity graph. When a
        presynaptic neuron fires, connected postsynaptic neurons receive
        a calcium boost after a configurable delay. Strong enough weights
        trigger propagation; weaker weights increase firing probability.

        A refractory period prevents re-firing: after firing, a neuron
        cannot be synaptically activated again until refractory steps
        have passed. This prevents infinite cycling in recurrent networks.
        ChR2 stimulation bypasses the refractory period (direct ion channel).

        Args:
            connection_prob: Probability of connection between neuron pairs
                             within max_distance. At 0.3, ~30% of in-range
                             pairs connect.
            max_distance: Maximum distance (world px) for possible connections.
            weight: Calcium boost delivered to postsynaptic neuron (0-1).
                    At 0.6, a single input triggers firing (threshold ~0.15).
                    At 0.3, needs ~2 concurrent inputs to trigger.
            delay: Steps of delay before signal arrives (models axonal
                   conduction). At delay=1, propagation takes 1 step per synapse.
            inhibitory_fraction: Fraction of connections that are inhibitory
                                 (negative weight, reduces calcium). Default 0.
            refractory: Steps after firing before neuron can fire synaptically
                        again. Models the biological refractory period. Default 3.
        """
        self._synaptic_refractory = refractory
        self._synaptic_enabled = True
        self._synaptic_delay = max(0, delay)
        self._synaptic_adj = {i: [] for i in range(self.n_neurons)}
        self._synaptic_queue = []

        conn_rng = np.random.default_rng(self._seed + 9000)

        for i in range(self.n_neurons):
            si = self.neurons[i]
            for j in range(self.n_neurons):
                if i == j:
                    continue
                sj = self.neurons[j]
                dist = np.hypot(si["soma_x"] - sj["soma_x"],
                                si["soma_y"] - sj["soma_y"])
                if dist > max_distance:
                    continue
                # Distance-weighted probability (closer = more likely)
                dist_factor = 1.0 - (dist / max_distance) ** 2
                if conn_rng.random() < connection_prob * dist_factor:
                    w = weight
                    if inhibitory_fraction > 0 and conn_rng.random() < inhibitory_fraction:
                        w = -weight * 0.5  # inhibitory weaker than excitatory
                    self._synaptic_adj[i].append((j, w))

        # Count connections for ground truth
        total = sum(len(v) for v in self._synaptic_adj.values())
        self._n_synapses = total

    def get_connectivity_ground_truth(self):
        """Return connectivity graph for ground truth.

        Returns dict with adjacency list and connection statistics.
        """
        if not self._synaptic_enabled:
            return {"enabled": False}

        adj = {}
        for pre, posts in self._synaptic_adj.items():
            if posts:
                adj[pre] = [(post, round(w, 2)) for post, w in posts]

        return {
            "enabled": True,
            "n_neurons": self.n_neurons,
            "n_synapses": self._n_synapses,
            "delay_steps": self._synaptic_delay,
            "adjacency": adj,
        }

    def apply_drug(self, drug_name, onset_rate=None):
        """Apply a neuroactive drug that modifies calcium activity."""
        drug_name = drug_name.lower()
        if drug_name not in self._drug_profiles:
            raise ValueError(f"Unknown drug: {drug_name}. "
                             f"Available: {list(self._drug_profiles.keys())}")
        self._drug_active = True
        self._drug_name = drug_name
        self._drug_washing_out = False
        if onset_rate is not None:
            self._drug_profiles[drug_name]["onset_rate"] = onset_rate

    def remove_drug(self, washout_rate=None):
        """Remove drug (washout)."""
        if not self._drug_active:
            return
        self._drug_washing_out = True
        if washout_rate is not None and self._drug_name:
            self._drug_profiles[self._drug_name]["washout_rate"] = washout_rate

    def _update_drug_effect(self, dt: float = 1.0):
        """Update drug effect level (gradual onset/washout, dt-scaled)."""
        if not self._drug_active:
            return
        profile = self._drug_profiles[self._drug_name]
        if self._drug_washing_out:
            self._drug_effect -= profile["washout_rate"] * dt
            if self._drug_effect <= 0:
                self._drug_effect = 0.0
                self._drug_active = False
                self._drug_name = None
                self._drug_washing_out = False
        else:
            target = profile["target_effect"]
            self._drug_effect += profile["onset_rate"] * (target - self._drug_effect) * dt

    def _get_temperature(self) -> float:
        """Read temperature from the Temperature state device (°C)."""
        if "Temperature" not in self.state_devices:
            return 37.0
        return float(self.state_devices["Temperature"].get("label", "37"))

    def _temp_rate_factor(self) -> float:
        """Temperature-dependent rate factor for neural dynamics.

        Q10 ~ 2.5 for ion channel kinetics and synaptic transmission.
        Optimal at 37°C. Cold slows firing, heat accelerates then denatures.
        """
        temp = self._get_temperature()
        if temp < 10:
            return 0.05
        factor = 2.5 ** ((temp - 37) / 10.0)
        if temp > 42:
            factor *= max(0.05, 1.0 - (temp - 42) * 0.3)
        return factor

    # ── SLM optogenetics (ChR2) ──

    def _map_slm_to_world(self, mask: np.ndarray) -> np.ndarray:
        """Map viewport-space SLM mask to world-coordinate bool array."""
        obj = self.current_objectiv
        fov_map = {100: 64, 40: 128, 20: 256}
        fov_world = fov_map.get(obj, min(512, self.width))

        cx = int(self.camera_offset[0]) + self.viewport_width // 2
        cy = int(self.camera_offset[1]) + self.viewport_height // 2

        mask_fov = cv2.resize(
            mask.astype(np.uint8), (fov_world, fov_world),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        world_mask = np.zeros((self.height, self.width), dtype=bool)
        half = fov_world // 2
        x0 = max(0, min(cx - half, self.width - fov_world))
        y0 = max(0, min(cy - half, self.height - fov_world))

        wx1 = min(self.width, x0 + fov_world)
        wy1 = min(self.height, y0 + fov_world)
        mw = wx1 - x0
        mh = wy1 - y0
        world_mask[y0:y0 + mh, x0:x0 + mw] = mask_fov[:mh, :mw]
        return world_mask

    def _apply_slm_stimulation(self):
        """ChR2 optogenetic stimulation: illuminated neurons fire.

        Blue light activates channelrhodopsin-2 → depolarization → calcium
        transient. ChR2 directly opens cation channels, bypassing the
        refractory period — illuminated neurons maintain high calcium.
        """
        if self._slm_mask is None or not self._calcium_enabled:
            return

        for i in range(self.n_neurons):
            n = self.neurons[i]
            sx = int(n["soma_x"])
            sy = int(n["soma_y"])
            if (0 <= sx < self.width and 0 <= sy < self.height
                    and self._slm_mask[sy, sx]):
                # ChR2 forces depolarization — bypasses refractory period
                was_low = self._ca_levels[i] < 0.5
                if was_low:
                    self._ca_fire_log.append((self._ca_time, i))
                self._ca_levels[i] = 1.0
                self._ca_last_fire[i] = self._ca_time  # record fire time
                # Propagate through synapses
                if was_low and self._synaptic_enabled:
                    for (post, w) in self._synaptic_adj.get(i, []):
                        delivery_t = self._ca_time + self._synaptic_delay
                        self._synaptic_queue.append((delivery_t, post, w))

    def step(self, dt: float = 1.0):
        """Advance dynamics by one timestep."""
        # Z-drift (mechanical — not temperature-dependent)
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self._rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

        if not self._calcium_enabled:
            return

        temp_factor = self._temp_rate_factor()

        self._ca_time += dt
        self._update_drug_effect(dt)

        # Decay existing calcium levels (dt-scaled exponential, faster at higher temp)
        effective_decay = min(1.0, self._ca_decay * temp_factor)
        self._ca_levels *= (1.0 - effective_decay) ** dt

        # Compute effective firing probability (drug + temperature modulated, dt-scaled)
        base_freq = self._ca_frequency * temp_factor
        if self._drug_active and self._drug_name:
            profile = self._drug_profiles[self._drug_name]
            mult = profile["firing_mult"]
            eff_freq = base_freq * (1.0 + (mult - 1.0) * self._drug_effect)
            eff_freq = max(0.0, min(1.0, eff_freq))
        else:
            eff_freq = min(1.0, base_freq)  # clamp to valid probability

        # Process synaptic queue: deliver delayed inputs
        if self._synaptic_enabled:
            pending = []
            for (fire_t, post_idx, w) in self._synaptic_queue:
                if self._ca_time >= fire_t:
                    # Check refractory period — skip if neuron fired recently
                    time_since_fire = self._ca_time - self._ca_last_fire[post_idx]
                    if time_since_fire <= self._synaptic_refractory:
                        continue  # neuron is refractory, discard input
                    self._ca_levels[post_idx] = np.clip(
                        self._ca_levels[post_idx] + w, 0.0, 1.0)
                    if self._ca_levels[post_idx] >= self._synaptic_fire_threshold:
                        # Synaptic input strong enough to fire
                        self._ca_levels[post_idx] = 1.0
                        self._ca_last_fire[post_idx] = self._ca_time
                        self._ca_fire_log.append((self._ca_time, post_idx))
                        # Propagate further (chain reaction)
                        for (post2, w2) in self._synaptic_adj.get(post_idx, []):
                            delivery_t = self._ca_time + self._synaptic_delay
                            pending.append((delivery_t, post2, w2))
                else:
                    pending.append((fire_t, post_idx, w))
            self._synaptic_queue = pending

        # Fire idle neurons with Poisson probability (dt-scaled)
        fire_prob = min(1.0, eff_freq * dt)
        for i in range(self.n_neurons):
            if self._ca_levels[i] < 0.15:  # only fire if calcium is low
                time_since_fire = self._ca_time - self._ca_last_fire[i]
                if time_since_fire < self._synaptic_refractory:
                    continue  # still in refractory period
                if self._ca_rng.random() < fire_prob:
                    self._ca_levels[i] = 1.0
                    self._ca_last_fire[i] = self._ca_time
                    self._ca_fire_log.append((self._ca_time, i))
                    # Queue synaptic outputs
                    if self._synaptic_enabled:
                        for (post, w) in self._synaptic_adj.get(i, []):
                            delivery_t = self._ca_time + self._synaptic_delay
                            self._synaptic_queue.append((delivery_t, post, w))

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics — same as step() (SLM is snap-coupled)."""
        self.step(dt)

    # ── Extra channels ──

    def add_channel(self, mode_id, name, led, filt, render_fn):
        """Register an extra fluorescence channel."""
        self._extra_channels[mode_id] = {
            "name": name, "led": led, "filter": filt,
            "render_fn": render_fn, "image": None,
        }

    # ── Ground truth ──

    def get_ground_truth(self):
        """Return ground truth for grading."""
        neuron_data = []
        total_length = 0.0
        total_branches = 0
        total_puncta = len(self.puncta)

        for neuron in self.neurons:
            # Total neurite length
            n_length = sum(
                np.hypot(x1 - x0, y1 - y0)
                for (x0, y0, x1, y1, w, is_d) in neuron["segments"]
            )
            n_dend_length = sum(
                np.hypot(x1 - x0, y1 - y0)
                for (x0, y0, x1, y1, w, is_d) in neuron["segments"] if is_d
            )
            n_branches = len(neuron["branch_points"])
            total_length += n_length
            total_branches += n_branches

            neuron_data.append({
                "type": neuron["type"],
                "soma_x": round(neuron["soma_x"], 1),
                "soma_y": round(neuron["soma_y"], 1),
                "soma_r": round(neuron["soma_r"], 1),
                "total_neurite_length": round(n_length, 1),
                "dendrite_length": round(n_dend_length, 1),
                "n_branch_points": n_branches,
                "n_segments": len(neuron["segments"]),
            })

        total_spines = len(self.spines)
        total_varicosities = sum(len(n.get("varicosities", []))
                                 for n in self.neurons)

        gt = {
            "n_neurons": self.n_neurons,
            "neurons": neuron_data,
            "total_neurite_length": round(total_length, 1),
            "total_branch_points": total_branches,
            "total_puncta": total_puncta,
            "total_spines": total_spines,
            "total_varicosities": total_varicosities,
            "type_counts": {
                "pyramidal": sum(1 for n in self.neurons if n["type"] == "pyramidal"),
                "stellate": sum(1 for n in self.neurons if n["type"] == "stellate"),
                "bipolar": sum(1 for n in self.neurons if n["type"] == "bipolar"),
            },
        }

        if self._calcium_enabled:
            gt["calcium"] = {
                "time": self._ca_time,
                "levels": [round(float(v), 3) for v in self._ca_levels],
                "n_active": int(np.sum(self._ca_levels > 0.15)),
                "total_fires": len(self._ca_fire_log),
                "fire_log": self._ca_fire_log[-50:],  # last 50 events
            }

        if self._drug_active:
            profile = self._drug_profiles[self._drug_name]
            gt["drug"] = {
                "name": self._drug_name,
                "effect": round(self._drug_effect, 3),
                "firing_mult": profile["firing_mult"],
                "washing_out": self._drug_washing_out,
            }

        return gt
