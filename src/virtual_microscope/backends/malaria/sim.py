"""
MalariaSmearSim — Giemsa-stained thin blood smear with Plasmodium falciparum.

Simulates a Giemsa-stained thin blood smear containing malaria parasites:
  - Normal RBCs: biconcave discs, ~7µm diameter (same as blood_smear_sim)
  - Infected RBCs: contain P. falciparum at various stages
  - WBCs: neutrophils, lymphocytes (sparse — thin smear has few)
  - Parasitemia: fraction of RBCs containing parasites

Parasite stages (P. falciparum thin smear):
  - Ring (early trophozoite): small ~1-2µm, thin ring with 1-2 chromatin dots
    Most common stage in peripheral blood. Delicate ring shape, pale blue
    cytoplasm, 1-2 red-purple chromatin dots. Multiple rings per RBC possible.
  - Trophozoite: larger ~3-4µm, irregular shape with hemozoin pigment
    Less common in P. falciparum (sequestered in capillaries). Dense blue
    cytoplasm, single large chromatin mass, hemozoin granules.
  - Schizont: fills most of RBC, contains 8-32 merozoites
    Rare in P. falciparum peripheral blood. Large structure with multiple
    small nuclei (merozoites) arranged around central pigment.
  - Gametocyte: banana/crescent shaped, diagnostic for P. falciparum
    Distinctive curved shape, red-purple chromatin, blue cytoplasm.
    P. falciparum gametocytes are UNIQUELY crescent-shaped (banana).

Channels:
  - mode 0: Brightfield (Giemsa stain)
  - mode 1: "nucleus channel" — parasite chromatin dots (training aid)
  - mode 2: "membrane channel" — infected RBC outlines (training aid)

Usage via SimulationBridge:
    sim = MalariaSmearSim(parasitemia=0.05, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class MalariaSmearSim(SimBase):
    """Giemsa-stained thin blood smear with P. falciparum parasites.

    Primary workflow: scan at 100x oil immersion, identify infected RBCs,
    classify parasite stages, calculate parasitemia.
    """

    # Parasite stage parameters (RGB Giemsa colors)
    PARASITE_STAGES = {
        "ring": {
            "fraction": 0.70,        # most common in peripheral blood
            "size_ratio": 0.25,      # ~1/5 of RBC diameter (CDC guideline)
            "chromatin_dots": (1, 2), # number of chromatin dots
            "color_cyto": (160, 175, 210),     # pale blue cytoplasm
            "color_chromatin": (170, 35, 65),   # vivid red-magenta chromatin
        },
        "trophozoite": {
            "fraction": 0.10,        # less common (sequestered)
            "size_ratio": 0.60,
            "chromatin_dots": (1, 1),
            "color_cyto": (120, 140, 195),     # denser blue
            "color_chromatin": (160, 30, 60),
        },
        "schizont": {
            "fraction": 0.05,        # rare in peripheral blood
            "size_ratio": 0.85,
            "chromatin_dots": (8, 24),  # merozoites (real: 8-32, typically 8-24)
            "color_cyto": (140, 155, 200),
            "color_chromatin": (165, 35, 65),
        },
        "gametocyte": {
            "fraction": 0.15,        # distinctive banana shape
            "size_ratio": 1.1,       # fills/extends beyond RBC
            "chromatin_dots": (1, 1),
            "color_cyto": (100, 115, 180),     # deep blue-purple
            "color_chromatin": (160, 30, 60),
        },
    }

    # Stage progression thresholds (hours of parasite age)
    STAGE_THRESHOLDS = {
        "ring": (0, 16),
        "trophozoite": (16, 36),
        "schizont": (36, 48),
    }

    # Antimalarial drug profiles
    DRUG_PROFILES = {
        "artemisinin": {
            "kill_rates": {"ring": 0.15, "trophozoite": 0.08,
                           "schizont": 0.05, "gametocyte": 0.02},
            "onset_rate": 0.15,
            "washout_rate": 0.08,
        },
        "chloroquine": {
            "kill_rates": {"ring": 0.02, "trophozoite": 0.12,
                           "schizont": 0.10, "gametocyte": 0.01},
            "onset_rate": 0.10,
            "washout_rate": 0.05,
        },
    }

    def __init__(
        self,
        world_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        n_rbc: int = 2000,
        n_wbc: int = 8,
        parasitemia: float = 0.05,
        stage_distribution: dict = None,
        multi_infection_rate: float = 0.10,
        n_platelets: int = 0,
        applique_rate: float = 0.35,
        seed: int = 42,
        internal_scale: int = 4,
    ):
        """
        Args:
            parasitemia: fraction of RBCs infected (0.0 to 1.0)
            stage_distribution: override default stage fractions
            multi_infection_rate: fraction of infected RBCs with 2+ parasites
            n_platelets: number of platelet particles (small purple dots, 0=auto)
            applique_rate: fraction of rings placed at RBC periphery (P. falciparum)
            internal_scale: render at world_size * scale internally (default 4)
        """
        super().__init__(
            width=world_size, height=world_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale,
            auto_step=False, snaps_per_step=2,
            mode_map={
                ("mScarlet3(569/582)", "ORANGE"): 1,   # chromatin-aid
                ("miRFP670(642/670)", "RED"): 2,       # RBC-overlay
            },
        )

        # Override DOF table (malaria uses 0.5 for 100x instead of default 0.6)
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.5}

        self._noise_rng = np.random.default_rng(seed + 7777)
        self._render_rng = np.random.default_rng(seed + 5555)

        # Dynamic lifecycle
        self._hours_per_step = 2.0  # simulated hours per step()
        self._destroyed = None      # set after _generate_cells
        self._dirty = False
        self._step_count = 0
        self._max_parasitemia = 0.25  # cap reinfection

        # Drug state
        self._drug_active = False
        self._drug_name = None
        self._drug_effect = 0.0
        self._drug_washing_out = False

        # Cell parameters
        self.n_rbc = n_rbc
        self.n_wbc = n_wbc
        self.parasitemia = parasitemia
        self.multi_infection_rate = multi_infection_rate
        self._applique_rate = applique_rate

        # Platelets — auto-compute if 0 (~10-20 per 100x FOV in a normal smear)
        if n_platelets == 0:
            # ~150-400k/µL vs ~4-5M/µL RBC → ~1 platelet per 15 RBCs
            n_platelets = max(20, n_rbc // 15)
        self.n_platelets = n_platelets

        # Stage distribution
        if stage_distribution is not None:
            self._stage_dist = stage_distribution
        else:
            self._stage_dist = {k: v["fraction"] for k, v in self.PARASITE_STAGES.items()}

        # RGB Giemsa stain colors — warmer salmon-pink (eosin on hemoglobin)
        self.rgb_mode = True
        self._bg_color = np.array([240, 236, 233], dtype=np.float32)  # near-white, faint warm
        self._rbc_body = np.array([225, 170, 155], dtype=np.float32)  # salmon-pink
        self._rbc_pallor = np.array([240, 225, 218], dtype=np.float32)  # almost white center
        self._rbc_edge = np.array([205, 148, 135], dtype=np.float32)  # darker salmon rim
        self._infected_rbc = np.array([200, 158, 148], dtype=np.float32)  # duller, slightly grey
        self._infected_pallor = np.array([215, 195, 185], dtype=np.float32)
        self._hemozoin_color = (60, 45, 30)      # dark brown-black
        self._ghost_fill = np.array([235, 228, 225], dtype=np.float32)
        self._ghost_outline = (215, 208, 205)
        self._wbc_nuc = (48, 30, 85)             # dark blue-violet
        self._wbc_cyto_neut = (210, 195, 205)    # pale pink
        self._wbc_cyto_lymph = (175, 185, 210)   # pale blue

        # Optical pipeline (single instance, shared across all channels)
        self._malaria_pipeline = OpticalPipeline()
        self._malaria_pipeline.noise = {"photon_scale": 500, "read_noise": 1.5}
        self._malaria_pipeline.vignette = 0.05
        self._pipeline = {0: self._malaria_pipeline, 1: self._malaria_pipeline,
                          2: self._malaria_pipeline}

        # Generate and render
        self._generate_cells()
        self._bf_full = None
        self._nuc_full = None
        self._mem_full = None
        self._render_full()

    def _generate_cells(self):
        """Generate RBCs, WBCs, and assign parasite infections."""
        margin = 30
        rng = self.rng

        # ── RBCs ──
        self._rbc_x = rng.uniform(margin, self.width - margin, self.n_rbc)
        self._rbc_y = rng.uniform(margin, self.height - margin, self.n_rbc)
        self._rbc_radius = rng.normal(3.5, 0.3, self.n_rbc).clip(2.5, 4.5)

        # ── Infection assignment ──
        n_infected = int(round(self.parasitemia * self.n_rbc))
        infected_indices = rng.choice(self.n_rbc, min(n_infected, self.n_rbc),
                                      replace=False)
        self._infected = np.zeros(self.n_rbc, dtype=bool)
        self._infected[infected_indices] = True
        self.n_infected = int(self._infected.sum())

        # Assign parasite stages
        self._parasites = []  # list of dicts per RBC (empty if uninfected)
        stage_names = list(self._stage_dist.keys())
        stage_probs = np.array([self._stage_dist[s] for s in stage_names])
        stage_probs /= stage_probs.sum()

        stage_counts = {s: 0 for s in stage_names}

        for i in range(self.n_rbc):
            if not self._infected[i]:
                self._parasites.append([])
                continue

            # How many parasites in this RBC?
            if rng.random() < self.multi_infection_rate:
                n_parasites = rng.integers(2, 4)  # 2-3 parasites
            else:
                n_parasites = 1

            cell_parasites = []
            for _ in range(n_parasites):
                stage = rng.choice(stage_names, p=stage_probs)
                # Random position within RBC (offset from center)
                r = self._rbc_radius[i]
                max_offset = r * 0.4
                px = rng.uniform(-max_offset, max_offset)
                py = rng.uniform(-max_offset, max_offset)
                # Random orientation for gametocytes
                angle = rng.uniform(0, 2 * np.pi)
                dot_lo, dot_hi = self.PARASITE_STAGES[stage]["chromatin_dots"]
                n_dots = int(rng.integers(dot_lo, dot_hi + 1))
                # Gametocyte sex dimorphism: macro (female) = darker, micro (male) = paler
                gam_sex = None
                if stage == "gametocyte":
                    gam_sex = "macro" if rng.random() < 0.6 else "micro"
                cell_parasites.append({
                    "stage": stage,
                    "offset_x": px,
                    "offset_y": py,
                    "angle": angle,
                    "n_dots": n_dots,
                    "gam_sex": gam_sex,
                })
                stage_counts[stage] += 1
            self._parasites.append(cell_parasites)

        self._stage_counts = stage_counts
        self._dominant_stage = max(stage_counts, key=stage_counts.get) if any(
            v > 0 for v in stage_counts.values()) else "ring"

        # ── Initialize parasite ages ──
        for i in range(self.n_rbc):
            for p in self._parasites[i]:
                stage = p["stage"]
                if stage == "gametocyte":
                    p["age"] = -1.0  # stable, doesn't progress
                elif stage in self.STAGE_THRESHOLDS:
                    lo, hi = self.STAGE_THRESHOLDS[stage]
                    p["age"] = float(rng.uniform(lo, hi))
                else:
                    p["age"] = 0.0

        # ── Destroyed RBC tracking ──
        self._destroyed = np.zeros(self.n_rbc, dtype=bool)

        # ── WBCs (sparse in thin smear) ──
        self._wbc_x = rng.uniform(margin + 10, self.width - margin - 10, self.n_wbc)
        self._wbc_y = rng.uniform(margin + 10, self.height - margin - 10, self.n_wbc)
        wbc_types = ["neutrophil"] * max(1, self.n_wbc // 2) + \
                    ["lymphocyte"] * (self.n_wbc - max(1, self.n_wbc // 2))
        rng.shuffle(wbc_types)
        self._wbc_types = wbc_types[:self.n_wbc]
        self._wbc_radius = np.array([6.5 if t == "neutrophil" else 4.5
                                     for t in self._wbc_types])

        # ── Platelets (small purple-blue bodies, 1-3µm) ──
        self._plt_x = rng.uniform(margin, self.width - margin, self.n_platelets)
        self._plt_y = rng.uniform(margin, self.height - margin, self.n_platelets)
        self._plt_radius = rng.uniform(0.5, 1.5, self.n_platelets)  # µm → world px

        # ── Appliqué flags for ring-stage parasites ──
        for i in range(self.n_rbc):
            for p in self._parasites[i]:
                if p["stage"] == "ring":
                    p["applique"] = bool(rng.random() < self._applique_rate)
                    if p["applique"]:
                        # Place ring on RBC periphery
                        edge_angle = float(rng.uniform(0, 2 * np.pi))
                        edge_r = self._rbc_radius[i] * 0.75
                        p["offset_x"] = edge_r * np.cos(edge_angle)
                        p["offset_y"] = edge_r * np.sin(edge_angle)
                else:
                    p["applique"] = False

    def _render_full(self):
        """Pre-render all channels at full resolution."""
        self._bf_base = self._render_bf_base()
        self._bf_full = self._render_bf_infections()
        self._nuc_full = self._render_nuc()
        self._mem_full = self._render_mem()

    def _rerender_dynamic(self):
        """Re-render only infection-dependent channels (after step)."""
        self._render_rng = np.random.default_rng(self._step_count + 5555)
        self._bf_full = self._render_bf_infections()
        self._nuc_full = self._render_nuc()
        self._mem_full = self._render_mem()
        self._dirty = False

    def _make_rbc_stamp(self, radius: float) -> np.ndarray:
        """Create biconcave disc stamp at internal resolution (RGB)."""
        s = self.internal_scale
        scaled_r = radius * s
        r = int(np.ceil(scaled_r)) + 1
        size = 2 * r + 1
        stamp = np.zeros((size, size, 3), dtype=np.float32)
        stamp[:] = self._bg_color

        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= scaled_r:
                    t = dist / scaled_r
                    ring_factor = np.sin(np.pi * t) ** 0.7
                    # Blend body → pallor based on ring profile
                    color = self._rbc_pallor * (1 - ring_factor) + self._rbc_body * ring_factor
                    if t > 0.82:
                        edge_fade = (1.0 - t) / 0.18
                        color = self._rbc_edge * edge_fade + self._bg_color * (1 - edge_fade)
                    stamp[cy + dy, cx + dx] = color
        return stamp

    def _render_bf_base(self) -> np.ndarray:
        """Render RBC stamps + WBCs at internal resolution (RGB, no infection overlay)."""
        s = self.internal_scale
        iw, ih = self._iw, self._ih
        img = np.zeros((ih, iw, 3), dtype=np.float32)
        img[:] = self._bg_color

        # Subtle background texture (per-channel noise)
        bg_noise = self._noise_rng.normal(0, 1.5, (self.height, self.width)).astype(np.float32)
        if s > 1:
            bg_noise = cv2.resize(bg_noise, (iw, ih), interpolation=cv2.INTER_LINEAR)
        img += bg_noise[:, :, None]

        # ── RBC stamps (uninfected appearance for all) ──
        stamp_cache = {}
        for i in range(self.n_rbc):
            r_key = round(self._rbc_radius[i] * 2) / 2
            if r_key not in stamp_cache:
                stamp_cache[r_key] = self._make_rbc_stamp(r_key)

        for i in range(self.n_rbc):
            cx, cy = self._s(self._rbc_x[i]), self._s(self._rbc_y[i])
            r_key = round(self._rbc_radius[i] * 2) / 2
            stamp = stamp_cache[r_key]
            sh, sw = stamp.shape[:2]
            hr, wr = sh // 2, sw // 2
            y0, y1 = cy - hr, cy - hr + sh
            x0, x1 = cx - wr, cx - wr + sw
            sy0, sx0 = 0, 0
            if y0 < 0:
                sy0 = -y0; y0 = 0
            if x0 < 0:
                sx0 = -x0; x0 = 0
            if y1 > ih:
                y1 = ih
            if x1 > iw:
                x1 = iw
            if y1 <= y0 or x1 <= x0:
                continue
            region = stamp[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
            # Per-channel minimum compositing (darker cell occludes background)
            img[y0:y1, x0:x1] = np.minimum(img[y0:y1, x0:x1], region)

        # ── WBCs ──
        for i in range(self.n_wbc):
            self._render_wbc_bf(img, i)

        # ── Platelets (small purple-blue bodies) ──
        plt_color_dark = (100, 70, 140)   # dark purple core
        plt_color_light = (160, 140, 180)  # lighter halo
        for i in range(self.n_platelets):
            px = self._s(self._plt_x[i])
            py = self._s(self._plt_y[i])
            pr = max(1, self._s(self._plt_radius[i]))
            # Halo (lighter)
            if pr > 1:
                cv2.circle(img, (px, py), pr + 1, plt_color_light, -1, cv2.LINE_AA)
            # Core (darker purple)
            cv2.circle(img, (px, py), pr, plt_color_dark, -1, cv2.LINE_AA)
            # Some platelets form small clusters (2-3 touching)
            if i < self.n_platelets - 1 and self._noise_rng.random() < 0.08:
                dx = self._noise_rng.integers(-pr * 2, pr * 2 + 1)
                dy = self._noise_rng.integers(-pr * 2, pr * 2 + 1)
                cv2.circle(img, (px + dx, py + dy), max(1, pr - 1),
                           plt_color_dark, -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_bf_infections(self) -> np.ndarray:
        """Overlay infection state (infected RBCs, parasites, ghosts) onto RGB base."""
        s = self.internal_scale
        lt = max(1, s // 2)
        img = self._bf_base.astype(np.float32)
        rng = self._render_rng

        ghost_fill = tuple(int(c) for c in self._ghost_fill)
        ghost_out = self._ghost_outline
        hemozoin = self._hemozoin_color
        inf_body = tuple(int(c) for c in self._infected_rbc)
        inf_pallor = tuple(int(c) for c in self._infected_pallor)
        inf_edge = tuple(int(max(0, c - 15)) for c in self._infected_rbc)

        for i in range(self.n_rbc):
            cx, cy = self._s(self._rbc_x[i]), self._s(self._rbc_y[i])
            r = self._s(self._rbc_radius[i])

            if self._destroyed[i]:
                # Ghost cell: clear area, faint outline + hemozoin debris
                cv2.circle(img, (cx, cy), r + 2, ghost_fill, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), r, ghost_out, lt, cv2.LINE_AA)
                n_pigment = int(rng.integers(3, 8))
                for _ in range(n_pigment):
                    dx = int(rng.integers(-r, r + 1))
                    dy = int(rng.integers(-r, r + 1))
                    if dx * dx + dy * dy < r * r:
                        cv2.circle(img, (cx + dx, cy + dy), lt, hemozoin, -1)
                continue

            if not self._infected[i]:
                continue

            # Darken infected RBC (duller pink)
            cv2.circle(img, (cx, cy), r, inf_body, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), max(1, int(r * 0.35)), inf_pallor, -1)
            cv2.circle(img, (cx, cy), r, inf_edge, lt, cv2.LINE_AA)

            # Hemozoin granules for mature stages
            for p in self._parasites[i]:
                if p["stage"] in ("trophozoite", "schizont"):
                    n_hem = int(rng.integers(3, 7))
                    for _ in range(n_hem):
                        dx = int(rng.integers(-r, r + 1))
                        dy = int(rng.integers(-r, r + 1))
                        if dx * dx + dy * dy < r * r:
                            cv2.circle(img, (cx + dx, cy + dy),
                                       max(1, lt - 1), hemozoin, -1)

            # Maurer's clefts — diagnostic P. falciparum feature
            # Small dark stipples in infected RBC cytoplasm (not rings, older stages)
            has_mature = any(p["stage"] in ("trophozoite", "schizont") for p in self._parasites[i])
            if has_mature or rng.random() < 0.3:
                n_clefts = int(rng.integers(2, 6))
                cleft_color = (80, 65, 130)  # dark blue-purple stipples
                for _ in range(n_clefts):
                    dx = int(rng.integers(-r + 2, r - 1))
                    dy = int(rng.integers(-r + 2, r - 1))
                    if dx * dx + dy * dy < (r - 2) ** 2:
                        cleft_len = max(1, int(rng.integers(lt, lt * 3)))
                        ca = float(rng.uniform(0, np.pi))
                        x1 = int(cx + dx - cleft_len * np.cos(ca))
                        y1 = int(cy + dy - cleft_len * np.sin(ca))
                        x2 = int(cx + dx + cleft_len * np.cos(ca))
                        y2 = int(cy + dy + cleft_len * np.sin(ca))
                        cv2.line(img, (x1, y1), (x2, y2), cleft_color, max(1, lt - 1))

            # Render each parasite
            for p in self._parasites[i]:
                self._render_parasite_bf(img, cx, cy, self._rbc_radius[i], p)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_parasite_bf(self, img, rbc_cx, rbc_cy, rbc_r_world, parasite):
        """Render a single parasite inside an RBC on the RGB BF image.

        rbc_cx, rbc_cy are already in internal coords.
        rbc_r_world is the radius in world coords (will be scaled).
        """
        s = self.internal_scale
        lt = max(1, s // 2)
        stage = parasite["stage"]
        params = self.PARASITE_STAGES[stage]
        px = int(round(rbc_cx + parasite["offset_x"] * s))
        py = int(round(rbc_cy + parasite["offset_y"] * s))
        angle = parasite["angle"]

        rbc_r = rbc_r_world * s
        size = max(2 * s, int(rbc_r * params["size_ratio"]))
        color_cyto = params["color_cyto"]
        color_chrom = params["color_chromatin"]
        # Ring interior: slightly lighter than infected pallor
        ring_interior = tuple(int(c) for c in self._infected_pallor)

        if stage == "ring":
            ring_r = max(2 * s, size)
            # P. falciparum rings are extremely delicate — barely visible arc
            # with prominent chromatin dots (the dots are the main visual feature)
            ring_thick = max(1, min(lt - 1, ring_r // 4))  # thinner than before
            # Draw only a partial arc (signet ring — not a full circle)
            arc_start = np.degrees(angle + 0.4)
            arc_end = np.degrees(angle + 2 * np.pi - 0.4)
            cv2.ellipse(img, (px, py), (ring_r, ring_r), 0,
                        arc_start, arc_end, color_cyto, ring_thick, cv2.LINE_AA)
            # Clear interior for ring transparency
            inner_r = max(1, ring_r - ring_thick - 1)
            if inner_r > 1:
                cv2.circle(img, (px, py), inner_r, ring_interior, -1)
            # Chromatin dots — vivid red-magenta, on ring perimeter
            # These are the MOST prominent feature of the ring
            for d in range(parasite["n_dots"]):
                dot_angle = angle + d * np.pi * 0.7
                dx = int(ring_r * np.cos(dot_angle))
                dy = int(ring_r * np.sin(dot_angle))
                dot_r = max(lt, int(ring_r * 0.50))  # large relative to ring
                cv2.circle(img, (px + dx, py + dy), dot_r, color_chrom, -1)

        elif stage == "trophozoite":
            ax1 = max(2 * s, int(size * 1.3))
            ax2 = max(2 * s, int(size * 0.9))
            ang_deg = np.degrees(angle)
            cv2.ellipse(img, (px, py), (ax1, ax2), ang_deg, 0, 360,
                        color_cyto, -1, cv2.LINE_AA)
            chrom_r = max(lt, int(size * 0.35))
            cv2.circle(img, (px, py), chrom_r, color_chrom, -1)
            hemozoin = self._hemozoin_color
            for _ in range(4):
                gx = px + self._render_rng.integers(-size, size + 1)
                gy = py + self._render_rng.integers(-size, size + 1)
                cv2.circle(img, (int(gx), int(gy)), lt, hemozoin, -1)

        elif stage == "schizont":
            sch_r = max(3 * s, size)
            cv2.circle(img, (px, py), sch_r, color_cyto, -1, cv2.LINE_AA)
            n_mero = parasite["n_dots"]
            mero_r = max(lt, int(sch_r * 0.18))
            # Irregularly packed merozoite cluster (not evenly-spaced rosette)
            rng = self._render_rng
            for m in range(n_mero):
                # Pack merozoites in a roughly circular cluster with jitter
                ma = 2 * np.pi * m / n_mero + float(rng.uniform(-0.3, 0.3))
                mr = float(rng.uniform(0.25, 0.65)) * sch_r  # variable radius
                mx = int(px + mr * np.cos(ma))
                my = int(py + mr * np.sin(ma))
                cv2.circle(img, (mx, my), mero_r, color_chrom, -1)
            # Central hemozoin pigment mass — dense clump
            cv2.circle(img, (px, py), max(lt, int(sch_r * 0.22)),
                       self._hemozoin_color, -1)

        elif stage == "gametocyte":
            gam_len = max(4 * s, int(rbc_r * 1.1))
            gam_w = max(2 * s, int(rbc_r * 0.45))

            # Macro/micro dimorphism: macro (female) darker, micro (male) paler
            gam_sex = parasite.get("gam_sex", "macro")
            if gam_sex == "macro":
                # Macrogametocyte: darker blue-purple, concentrated pigment
                fill_color = color_cyto  # deep blue-purple (100, 115, 180)
                chrom_color = color_chrom
            else:
                # Microgametocyte: paler, more diffuse
                fill_color = (140, 150, 200)  # lighter blue
                chrom_color = (185, 55, 85)   # paler red-pink

            # Smooth banana crescent with rounded ends (Stage V mature form)
            n_pts = 40  # more points for smoother curve
            t = np.linspace(-0.90 * np.pi, 0.90 * np.pi, n_pts)

            ox = gam_len * np.sin(t)
            oy = -gam_w * np.cos(t)
            ix = (gam_len * 0.60) * np.sin(t[::-1])
            iy = -(gam_w * 0.20) * np.cos(t[::-1])

            crescent_x = np.concatenate([ox, ix])
            crescent_y = np.concatenate([oy, iy])

            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rx = crescent_x * cos_a - crescent_y * sin_a + px
            ry = crescent_x * sin_a + crescent_y * cos_a + py

            pts = np.column_stack([rx, ry]).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], fill_color)
            # Outline slightly darker
            outline = tuple(max(0, c - 20) for c in fill_color)
            cv2.polylines(img, [pts], True, outline, lt, cv2.LINE_AA)

            # Hemozoin pigment granules along crescent body (concentrated in macro)
            n_pigment = 5 if gam_sex == "macro" else 2
            for _ in range(n_pigment):
                pt = float(self._render_rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
                gx = gam_len * 0.4 * np.sin(pt)
                gy = -gam_w * 0.15 * np.cos(pt)
                rx_p = gx * cos_a - gy * sin_a + px
                ry_p = gx * sin_a + gy * cos_a + py
                cv2.circle(img, (int(rx_p), int(ry_p)), max(1, lt),
                           self._hemozoin_color, -1)

            chr_len = max(lt, gam_len // 3)
            chr_w = max(lt, gam_w // 3)
            cv2.ellipse(img, (px, py), (chr_len, chr_w),
                        np.degrees(angle), 0, 360, chrom_color, -1)

    def _render_wbc_bf(self, img, idx):
        """Render a WBC on RGB brightfield at internal resolution."""
        wtype = self._wbc_types[idx]
        cx = self._s(self._wbc_x[idx])
        cy = self._s(self._wbc_y[idx])
        r = self._s(self._wbc_radius[idx])
        nuc_color = self._wbc_nuc

        if wtype == "neutrophil":
            cv2.circle(img, (cx, cy), r, self._wbc_cyto_neut, -1, cv2.LINE_AA)
            n_lobes = self.rng.integers(3, 6)
            for lobe in range(n_lobes):
                la = 2 * np.pi * lobe / n_lobes + self.rng.uniform(-0.3, 0.3)
                lr = r * 0.4
                lx = int(cx + lr * np.cos(la))
                ly = int(cy + lr * np.sin(la))
                lobe_r = max(1, int(r * 0.25))
                cv2.circle(img, (lx, ly), lobe_r, nuc_color, -1)
        else:  # lymphocyte
            cv2.circle(img, (cx, cy), r, self._wbc_cyto_lymph, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), max(1, int(r * 0.75)), nuc_color, -1)

    def _render_nuc(self) -> np.ndarray:
        """Render parasite chromatin channel at internal resolution."""
        s = self.internal_scale
        lt = max(1, s // 2)
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        for i in range(self.n_rbc):
            if not self._infected[i]:
                continue
            cx, cy = self._s(self._rbc_x[i]), self._s(self._rbc_y[i])
            for p in self._parasites[i]:
                px = int(round(cx + p["offset_x"] * s))
                py = int(round(cy + p["offset_y"] * s))
                stage = p["stage"]
                r = self._rbc_radius[i] * s
                size = max(2 * s, int(r * self.PARASITE_STAGES[stage]["size_ratio"]))

                if stage == "ring":
                    ring_r = max(2 * s, size)
                    for d in range(p["n_dots"]):
                        dot_a = p["angle"] + d * np.pi * 0.7
                        dx = int(ring_r * np.cos(dot_a))
                        dy = int(ring_r * np.sin(dot_a))
                        cv2.circle(img, (px + dx, py + dy),
                                   max(lt, int(size * 0.4)), 220, -1)
                elif stage == "trophozoite":
                    cv2.circle(img, (px, py), max(2 * s, int(size * 0.35)), 240, -1)
                elif stage == "schizont":
                    n_m = p["n_dots"]
                    mero_ring = max(2 * s, int(size * 0.55))
                    for m in range(n_m):
                        ma = 2 * np.pi * m / n_m
                        mx = int(px + mero_ring * np.cos(ma))
                        my = int(py + mero_ring * np.sin(ma))
                        cv2.circle(img, (mx, my), max(lt, int(size * 0.15)),
                                   200, -1)
                elif stage == "gametocyte":
                    gam_len = max(4 * s, int(r * 1.1))
                    chr_len = max(lt, gam_len // 3)
                    chr_w = max(lt, int(r * 0.45) // 3)
                    cv2.ellipse(img, (px, py), (chr_len, chr_w),
                                np.degrees(p["angle"]), 0, 360, 230, -1)

        img += 3.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_mem(self) -> np.ndarray:
        """Render infected RBC outlines at internal resolution."""
        s = self.internal_scale
        lt = max(1, s // 2)
        img = np.zeros((self._ih, self._iw), dtype=np.float32)

        for i in range(self.n_rbc):
            if not self._infected[i]:
                continue
            cx, cy = self._s(self._rbc_x[i]), self._s(self._rbc_y[i])
            r = max(2 * s, self._s(self._rbc_radius[i]))
            cv2.circle(img, (cx, cy), r, 180, lt, cv2.LINE_AA)

        img += 2.0
        return np.clip(img, 0, 255).astype(np.uint8)

    # ── Template hooks ──

    def _render_for_mode(self, mode: int) -> np.ndarray:
        """Return full-resolution image for the active channel."""
        # Re-render if dirty (infection dynamics changed)
        if self._dirty:
            self._rerender_dynamic()

        if mode == 0:
            return self._bf_full
        elif mode == 1:
            return self._nuc_full
        elif mode == 2:
            return self._mem_full
        elif mode in self._extra_channels:
            return self._extra_channels[mode].get("image", self._bf_full)
        else:
            return self._bf_full

    def _finalize_output(self, viewport: np.ndarray) -> np.ndarray:
        """Convert viewport to grayscale; handle both 2D and 3D inputs."""
        if viewport.ndim == 3:
            return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)
        return viewport

    def _age_to_stage(self, age):
        """Convert parasite age (hours) to lifecycle stage name."""
        if age < 0:
            return "gametocyte"
        for stage, (lo, hi) in self.STAGE_THRESHOLDS.items():
            if lo <= age < hi:
                return stage
        return "burst"  # age >= 48

    def _check_perfusion_drug(self):
        """Check Perfusion device state and toggle drug delivery.

        Perfusion state 4 ("Drug") → apply artemisinin (first-line antimalarial).
        Perfusion state 5 ("Drug2") → apply chloroquine.
        Other states → wash out drug if active.
        """
        if "Perfusion" not in self.state_devices:
            return
        p_state = str(self.state_devices["Perfusion"].get("label", "Off"))
        if p_state == "Drug":
            if not self._drug_active or self._drug_name != "artemisinin":
                self.apply_drug("artemisinin")
        elif p_state == "Drug2":
            if not self._drug_active or self._drug_name != "chloroquine":
                self.apply_drug("chloroquine")
        else:
            if self._drug_active:
                self.remove_drug()

    def step(self, dt: float = 1.0):
        """Advance parasite lifecycle by one time step.

        Each step = self._hours_per_step hours of simulated time.
        Parasites age, progress through stages, schizonts burst
        and release merozoites that infect new RBCs.
        """
        self._step_count += 1
        self._time += dt
        hours = self._hours_per_step * dt

        # Check perfusion device for drug delivery
        self._check_perfusion_drug()

        # Update drug effect
        self._update_drug_effect()

        burst_indices = []  # RBCs whose schizonts burst this step

        for i in range(self.n_rbc):
            if not self._infected[i] or self._destroyed[i]:
                continue

            surviving = []
            any_burst = False

            for p in self._parasites[i]:
                if p["age"] < 0:
                    # Gametocyte — stable, but can be killed by drug
                    if self._drug_active:
                        kill_rate = self._get_kill_rate("gametocyte")
                        if self.rng.random() < kill_rate:
                            continue  # killed
                    surviving.append(p)
                    continue

                # Age the parasite
                p["age"] += hours
                new_stage = self._age_to_stage(p["age"])

                if new_stage == "burst":
                    any_burst = True
                    continue  # parasite consumed in burst

                # Drug killing
                if self._drug_active:
                    kill_rate = self._get_kill_rate(new_stage)
                    if self.rng.random() < kill_rate:
                        continue  # killed by drug

                # Update stage and chromatin dots if stage changed
                if new_stage != p["stage"]:
                    params = self.PARASITE_STAGES[new_stage]
                    dot_lo, dot_hi = params["chromatin_dots"]
                    p["n_dots"] = int(self.rng.integers(dot_lo, dot_hi + 1))
                    p["stage"] = new_stage

                surviving.append(p)

            self._parasites[i] = surviving

            if any_burst:
                burst_indices.append(i)

            # If all parasites cleared, mark uninfected
            if len(surviving) == 0 and not any_burst:
                self._infected[i] = False

        # Handle bursts: destroy RBCs and create new infections
        for idx in burst_indices:
            self._handle_burst(idx)

        # Recount
        self._update_stage_counts()
        self._dirty = True

    def _handle_burst(self, rbc_idx):
        """Handle schizont burst: destroy host RBC and infect nearby cells."""
        self._destroyed[rbc_idx] = True
        self._infected[rbc_idx] = False
        self._parasites[rbc_idx] = []

        # Check parasitemia cap
        current_parasitemia = self._infected.sum() / self.n_rbc
        if current_parasitemia >= self._max_parasitemia:
            return

        # Release merozoites — infect nearby uninfected RBCs
        bx, by = self._rbc_x[rbc_idx], self._rbc_y[rbc_idx]
        n_merozoites = int(self.rng.integers(8, 24))
        infection_radius = 40.0  # world units — nearby cells

        # Find candidate RBCs
        dx = self._rbc_x - bx
        dy = self._rbc_y - by
        dist = np.sqrt(dx * dx + dy * dy)
        candidates = np.where(
            (dist < infection_radius) & ~self._infected & ~self._destroyed
        )[0]

        if len(candidates) == 0:
            return

        # Each merozoite has ~30% chance to infect
        n_new = 0
        for _ in range(n_merozoites):
            if self.rng.random() < 0.30 and n_new < len(candidates):
                target = candidates[n_new]
                self._infected[target] = True
                # New ring parasite at age 0
                r = self._rbc_radius[target]
                max_offset = r * 0.4
                is_applique = bool(self.rng.random() < self._applique_rate)
                if is_applique:
                    edge_angle = float(self.rng.uniform(0, 2 * np.pi))
                    off_x = r * 0.75 * np.cos(edge_angle)
                    off_y = r * 0.75 * np.sin(edge_angle)
                else:
                    off_x = float(self.rng.uniform(-max_offset, max_offset))
                    off_y = float(self.rng.uniform(-max_offset, max_offset))
                self._parasites[target].append({
                    "stage": "ring",
                    "offset_x": off_x,
                    "offset_y": off_y,
                    "angle": float(self.rng.uniform(0, 2 * np.pi)),
                    "n_dots": int(self.rng.integers(1, 3)),
                    "age": 0.0,
                    "gam_sex": None,
                    "applique": is_applique,
                })
                n_new += 1

    def _update_stage_counts(self):
        """Recount parasites by stage after dynamics."""
        counts = {s: 0 for s in self.PARASITE_STAGES}
        for i in range(self.n_rbc):
            for p in self._parasites[i]:
                stage = p["stage"]
                if stage in counts:
                    counts[stage] += 1
        self._stage_counts = counts
        self.n_infected = int(self._infected.sum())
        self._dominant_stage = max(counts, key=counts.get) if any(
            v > 0 for v in counts.values()) else "ring"

    def _get_kill_rate(self, stage):
        """Get drug kill probability for a given parasite stage."""
        if not self._drug_active or self._drug_name is None:
            return 0.0
        profile = self.DRUG_PROFILES.get(self._drug_name, {})
        base_rate = profile.get("kill_rates", {}).get(stage, 0.0)
        return base_rate * self._drug_effect

    def _update_drug_effect(self):
        """Update drug effect level (gradual onset/washout)."""
        if not self._drug_active and not self._drug_washing_out:
            return
        profile = self.DRUG_PROFILES.get(self._drug_name, {})
        if self._drug_washing_out:
            rate = profile.get("washout_rate", 0.05)
            self._drug_effect = max(0.0, self._drug_effect - rate)
            if self._drug_effect <= 0.01:
                self._drug_effect = 0.0
                self._drug_washing_out = False
                self._drug_name = None
        else:
            rate = profile.get("onset_rate", 0.10)
            self._drug_effect = min(1.0, self._drug_effect + rate)

    def apply_drug(self, name):
        """Apply an antimalarial drug. Available: 'artemisinin', 'chloroquine'."""
        name = name.lower()
        if name not in self.DRUG_PROFILES:
            raise ValueError(f"Unknown drug: {name}. Available: {list(self.DRUG_PROFILES)}")
        self._drug_active = True
        self._drug_name = name
        self._drug_effect = 0.0
        self._drug_washing_out = False

    def remove_drug(self):
        """Remove drug — begins washout phase."""
        if self._drug_active:
            self._drug_active = False
            self._drug_washing_out = True

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._noise_rng = np.random.default_rng(seed + 7777)
            self._render_rng = np.random.default_rng(seed + 5555)
            self._step_count = 0
            self._time = 0.0
            self._drug_active = False
            self._drug_name = None
            self._drug_effect = 0.0
            self._drug_washing_out = False
            self._generate_cells()
            self._render_full()

    def get_ground_truth(self) -> dict:
        """Return ground truth for grading."""
        # Per-RBC infection info
        infected_positions = []
        for i in range(self.n_rbc):
            if self._infected[i]:
                stages = [p["stage"] for p in self._parasites[i]]
                infected_positions.append({
                    "x": float(self._rbc_x[i]),
                    "y": float(self._rbc_y[i]),
                    "stages": stages,
                })

        n_destroyed = int(self._destroyed.sum())
        gt = {
            "n_rbc": self.n_rbc,
            "n_infected": self.n_infected,
            "parasitemia": round(self.n_infected / self.n_rbc, 4),
            "stage_counts": dict(self._stage_counts),
            "dominant_stage": self._dominant_stage,
            "n_wbc": self.n_wbc,
            "n_destroyed": n_destroyed,
            "step_count": self._step_count,
            "simulated_hours": self._time * self._hours_per_step,
            "infected_positions": infected_positions,
        }

        # Drug state
        if self._drug_active or self._drug_washing_out:
            gt["drug"] = {
                "name": self._drug_name,
                "active": self._drug_active,
                "effect": round(self._drug_effect, 3),
                "washing_out": self._drug_washing_out,
            }

        return gt

    def add_channel(self, channel_id, name, filter_label, led_label, image=None):
        self._extra_channels[channel_id] = {
            "name": name, "filter": filter_label,
            "led": led_label, "image": image,
        }
