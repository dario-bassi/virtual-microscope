"""
ZebrafishSim — 48hpf zebrafish embryo simulation backend.

Simulates a zebrafish embryo (pharyngula stage, 48 hours post-fertilization)
in lateral view:
  - Transparent body with yolk sac, pigmented eye, bright notochord
  - Beating 2-chamber heart (~150 bpm at 28°C, Q10=2.0)
  - Blood cells flowing through vasculature (DA, PCV, ISVs, DLAV)
  - Z-stack navigation through different tissue layers

Channels:
  - mode 0 (brightfield): Transparent embryo, anatomical landmarks
  - mode 1 (nucleus-channel): Tg(flk1:GFP) — all vascular endothelium
  - mode 2 (membrane-channel): Tg(myl7:mCherry) — cardiac myocytes only

World: 1024x512 pixels (wide, embryo runs left-to-right).
At 10x FOV=512, agent sees half the body; must tile to survey full embryo.

Usage:
    sim = ZebrafishSim(n_rbc=30, cardiac_freq=2.5, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class ZebrafishSim(SimBase):
    """48hpf zebrafish embryo with beating heart and blood flow."""

    continuous = True

    def __init__(self, n_rbc=30, cardiac_freq=2.5,
                 world_width=1024, world_height=512, seed=42,
                 viewport_width=512, viewport_height=512,
                 internal_scale=2):
        super().__init__(
            width=world_width, height=world_height,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=internal_scale, fixed_dt=1.0,
            auto_step=False, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
                ("mScarlet3(569/582)", "ORANGE"): 2,
            },
        )

        self._seed = seed
        # Override DOF table (zebrafish uses 3.0 for 20x instead of default 4.0)
        self._dof_table = {10: 6.0, 20: 3.0, 40: 1.5, 100: 0.6}

        # Cardiac
        self._cardiac_freq = cardiac_freq
        self._base_cardiac_freq = cardiac_freq

        # Temperature (zebrafish optimal 28°C)
        self._temperature = 28.0
        self._q10 = 2.0

        # Anesthesia (Tricaine/MS-222)
        self._anesthesia_factor = 1.0  # 1.0 = no effect, 0.0 = full arrest

        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.4, noise={"photon_scale": 8.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 700),
            1: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.0},
                vignette=0.10, rng_seed=seed + 701),
            2: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 5.0, "read_std": 2.0},
                vignette=0.10, rng_seed=seed + 702),
        }

        self._rng = self.rng

        self._generate_body()
        self._generate_vasculature()
        self._generate_heart()
        self._generate_blood_cells(n_rbc)
        self._generate_somites()

    # ── Anatomy generation ──

    def _generate_body(self):
        """Define embryo body outline, eye, yolk, notochord, brain, melanophores."""
        rng = self._rng

        # Body outline control points (lateral view: head left, tail right)
        # Dorsal edge
        self._dorsal_pts = np.array([
            [120, 248], [140, 228], [160, 218], [190, 215],
            [220, 218], [260, 220], [350, 222], [500, 226],
            [650, 234], [750, 242], [830, 250], [850, 253],
        ], dtype=np.float32)
        # Ventral edge (includes yolk bulge)
        self._ventral_pts = np.array([
            [120, 258], [140, 268], [160, 275], [190, 282],
            [220, 288], [260, 292], [350, 288], [500, 278],
            [650, 268], [750, 260], [830, 256], [850, 253],
        ], dtype=np.float32)

        # Eye (prominent dark pigmented structure)
        self._eye_x = 172.0
        self._eye_y = 228.0
        self._eye_radius = 26.0
        self._lens_radius = 10.0
        self._eye_z = 15.0  # slightly dorsal

        # Otic vesicle (small circle behind eye, visible at 20x+)
        self._otic_x = 230.0
        self._otic_y = 240.0
        self._otic_r = 12.0
        self._otic_z = 10.0

        # Brain region (head — darker translucent area)
        self._brain_x0 = 140.0
        self._brain_x1 = 235.0
        self._brain_y0 = 218.0
        self._brain_y1 = 260.0
        self._brain_z = 12.0

        # Melanophore pigment spots along body and head
        n_melano = 35
        melano_xs = np.concatenate([
            rng.uniform(140, 210, 8),     # head melanophores
            rng.uniform(250, 800, 22),     # trunk lateral line
            rng.uniform(160, 230, 5),      # dorsal head
        ])
        melano_ys = np.concatenate([
            rng.uniform(218, 250, 8),      # head (near dorsal)
            rng.uniform(224, 246, 22),      # trunk near dorsal midline
            rng.uniform(214, 222, 5),       # above dorsal edge
        ])
        melano_r = rng.uniform(1.5, 4.0, n_melano)
        self._melanophores = np.column_stack([melano_xs, melano_ys, melano_r])
        self._melanophore_z = 5.0  # superficial

        # Yolk sac (large, ventral-anterior, granular)
        self._yolk_cx = 290.0
        self._yolk_cy = 330.0
        self._yolk_rx = 95.0
        self._yolk_ry = 55.0
        self._yolk_z = -60.0

        # Yolk extension (thin tube connecting yolk to trunk)
        self._yolk_ext_x0 = 350.0
        self._yolk_ext_x1 = 480.0
        self._yolk_ext_y = 286.0
        self._yolk_ext_hy = 8.0

        # Notochord (bright rod, main landmark)
        self._noto_y = 256.0
        self._noto_x0 = 165.0
        self._noto_x1 = 835.0
        self._noto_hw = 5.0  # half-width in pixels
        self._noto_z = 0.0

        # Pre-generate yolk texture (coarse + fine granules)
        rng_y = np.random.default_rng(self._seed + 1111)
        fine = rng_y.normal(0, 1,
                            (self.height, self.width)).astype(np.float32)
        coarse = rng_y.normal(0, 1,
                              (self.height, self.width)).astype(np.float32)
        coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=3.5)
        fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=1.0)
        self._yolk_tex = coarse * 0.6 + fine * 0.4

        # Swim bladder primordia (48 hpf: just forming as a small round sac)
        # Located anterior trunk, just ventral to the notochord, between DA and gut.
        # At 48hpf: ~25 px diameter, partially inflated.  Appears as a round clear
        # region with dark refractile walls (thin epithelium) in brightfield.
        self._sb_x = 465.0   # anterior trunk (between somites 12-15)
        self._sb_y = 272.0   # ventral to DA (y=247), dorsal to gut
        self._sb_r = 13.0    # primordia radius (~25 µm at 48hpf, 1px=2µm here)
        self._sb_z = -5.0    # slightly ventral (lower Z)

        # Median fin fold (transparent membrane, dorsal + ventral from mid-trunk to tail)
        # Dorsal fin fold: extends above dorsal edge from trunk to tail tip
        self._fin_dorsal_pts = np.array([
            [350, 222], [500, 215], [650, 218], [750, 228],
            [830, 240], [855, 252],  # outer edge (above body)
            [850, 253], [830, 250], [750, 242], [650, 234],
            [500, 226], [350, 222],  # body edge (inner)
        ], dtype=np.float32)
        # Ventral fin fold: extends below ventral edge
        self._fin_ventral_pts = np.array([
            [400, 288], [500, 290], [650, 282], [750, 272],
            [830, 262], [855, 254],  # outer edge (below body)
            [850, 253], [830, 256], [750, 260], [650, 268],
            [500, 278], [400, 288],  # body edge (inner)
        ], dtype=np.float32)
        self._fin_z = 0.0

    def _generate_vasculature(self):
        """Generate vessel network: DA, PCV, ISVs, DLAV."""
        rng = self._rng

        # Dorsal aorta
        self._da_y = 247.0
        self._da_x0 = 220.0
        self._da_x1 = 790.0
        self._da_z = 8.0
        self._da_r = 3.5

        # Posterior cardinal vein
        self._pcv_y = 265.0
        self._pcv_x0 = 220.0
        self._pcv_x1 = 790.0
        self._pcv_z = -8.0
        self._pcv_r = 3.0

        # Intersegmental vessels (18 along the trunk)
        n_isv = 18
        self._isv_xs = np.linspace(265.0, 770.0, n_isv)
        self._isv_xs += rng.uniform(-1.5, 1.5, n_isv)
        self._isv_y_bot = self._da_y
        self._isv_y_top = 200.0
        self._isv_z_bot = self._da_z
        self._isv_z_top = 30.0
        self._isv_r = 1.5
        self._n_isv = n_isv
        # Per-ISV growth fraction: 1.0 = normal, 0.0 = absent
        self._isv_growth = np.ones(n_isv, dtype=np.float64)

        # DLAV (dorsal longitudinal anastomotic vessel)
        self._dlav_y = self._isv_y_top
        self._dlav_x0 = self._isv_xs[0] - 5
        self._dlav_x1 = self._isv_xs[-1] + 5
        self._dlav_z = 30.0
        self._dlav_r = 1.2

        # GFP intensities
        self._da_gfp = 175.0
        self._pcv_gfp = 155.0
        self._isv_gfp = 135.0 + rng.uniform(-8, 8, n_isv)
        self._dlav_gfp = 125.0

    def _generate_heart(self):
        """Two-chamber heart with oscillation parameters."""
        self._heart_x = 198.0
        self._heart_y = 290.0
        self._heart_z = -5.0

        self._vent_x = 206.0
        self._vent_y = 284.0
        self._vent_r0 = 13.0

        self._atri_x = 188.0
        self._atri_y = 296.0
        self._atri_r0 = 15.0

        self._atri_phase_lead = np.pi / 4

        # Pericardial edema (normal=0, edema inflates pericardial sac)
        self._pericardial_edema = 0.0  # 0=normal, 1=mild, 2=moderate, 3=severe
        self._cardiac_gfp = 195.0

    def _generate_blood_cells(self, n_rbc):
        """RBCs on parametric vessel paths."""
        rng = self._rng
        self._n_rbc = n_rbc

        n_da = int(n_rbc * 0.4)
        n_pcv = int(n_rbc * 0.3)
        n_isv = n_rbc - n_da - n_pcv

        path_type = []
        path_param = []
        isv_idx = []
        isv_dir = []

        for _ in range(n_da):
            path_type.append(0)  # DA
            path_param.append(rng.uniform(0, 1))
            isv_idx.append(-1)
            isv_dir.append(0)

        for _ in range(n_pcv):
            path_type.append(1)  # PCV
            path_param.append(rng.uniform(0, 1))
            isv_idx.append(-1)
            isv_dir.append(0)

        for _ in range(n_isv):
            path_type.append(2)  # ISV
            path_param.append(rng.uniform(0, 1))
            isv_idx.append(int(rng.integers(0, self._n_isv)))
            isv_dir.append(int(rng.choice([-1, 1])))

        self._rbc_ptype = np.array(path_type, dtype=np.int32)
        self._rbc_param = np.array(path_param, dtype=np.float64)
        self._rbc_isv_idx = np.array(isv_idx, dtype=np.int32)
        self._rbc_isv_dir = np.array(isv_dir, dtype=np.int32)
        self._rbc_size = rng.uniform(1.5, 2.5, n_rbc).astype(np.float32)
        # Small fixed lateral offsets within vessel lumen
        self._rbc_y_off = rng.uniform(-1.0, 1.0, n_rbc).astype(np.float32)

        # Base speeds (world px/s). 5 µm/px scale.
        self._da_speed = 40.0   # ~200 µm/s
        self._pcv_speed = 30.0  # ~150 µm/s
        self._isv_speed = 10.0  # ~50 µm/s

    def _generate_somites(self):
        """Somite boundary positions (chevron pattern)."""
        rng = self._rng
        n = 28
        self._somite_xs = np.linspace(255.0, 785.0, n)
        self._somite_xs += rng.uniform(-1.2, 1.2, n)
        self._somite_z = 0.0
        self._somite_chevron_angle = 12.0

    # ── Helpers ──

    def _cardiac_phase(self):
        return 2 * np.pi * self._cardiac_freq * self._time

    def _ventricle_state(self):
        phi = self._cardiac_phase()
        contraction = 0.20 * np.sin(phi)
        r = self._vent_r0 * (1.0 - contraction)
        wall_factor = 1.0 + 0.4 * np.sin(phi)
        return r, wall_factor

    def _atrium_state(self):
        phi = self._cardiac_phase() - self._atri_phase_lead
        contraction = 0.18 * np.sin(phi)
        r = self._atri_r0 * (1.0 - contraction)
        wall_factor = 1.0 + 0.35 * np.sin(phi)
        return r, wall_factor

    def _defocus_weight(self, struct_z, extent=5.0):
        """Visibility weight for a structure at given Z depth."""
        focal_z = self.focal_plane - self.tissue_z
        dz = abs(focal_z - struct_z)
        half_dof = self._dof / 2.0
        threshold = half_dof + extent / 2.0

        if dz <= threshold:
            return 1.0, 0.0
        elif dz < half_dof * 6:
            frac = (dz - threshold) / (half_dof * 5)
            opacity = max(0.05, 1.0 - frac)
            blur = max(0.0, (dz - threshold) * 0.5)
            return opacity, blur
        return 0.0, 0.0

    def _rbc_positions(self):
        """Current (x, y, z) for all RBCs."""
        xs = np.empty(self._n_rbc)
        ys = np.empty(self._n_rbc)
        zs = np.empty(self._n_rbc)

        for i in range(self._n_rbc):
            t = self._rbc_param[i] % 1.0
            pt = self._rbc_ptype[i]
            off = self._rbc_y_off[i]

            if pt == 0:  # DA
                xs[i] = self._da_x0 + t * (self._da_x1 - self._da_x0)
                ys[i] = self._da_y + off
                zs[i] = self._da_z
            elif pt == 1:  # PCV (flows left)
                xs[i] = self._pcv_x1 - t * (self._pcv_x1 - self._pcv_x0)
                ys[i] = self._pcv_y + off
                zs[i] = self._pcv_z
            else:  # ISV
                idx = self._rbc_isv_idx[i]
                ix = self._isv_xs[idx]
                xs[i] = ix + off * 0.5
                if self._rbc_isv_dir[i] > 0:  # up
                    ys[i] = self._isv_y_bot - t * (self._isv_y_bot - self._isv_y_top)
                    zs[i] = self._isv_z_bot + t * (self._isv_z_top - self._isv_z_bot)
                else:  # down
                    ys[i] = self._isv_y_top + t * (self._isv_y_bot - self._isv_y_top)
                    zs[i] = self._isv_z_top - t * (self._isv_z_top - self._isv_z_bot)

        return xs, ys, zs

    # ── Rendering ──

    def _render_for_mode(self, mode):
        if mode == 0:
            return self._render_bf()
        elif mode == 1:
            return self._render_vascular_gfp()
        elif mode == 2:
            return self._render_cardiac_gfp()
        return self._render_bf()

    def _auto_step_tick(self):
        """Read environmental controls, then auto-step."""
        self._read_temperature()
        self._read_anesthesia()
        super()._auto_step_tick()

    def _render_bf(self):
        """Brightfield: transparent embryo, high-contrast landmarks."""
        s = self.internal_scale
        ih, iw = self._ih, self._iw
        img = np.full((ih, iw), 195.0, dtype=np.float32)  # bright BF background

        # ── Body outline (clearly visible boundary) ──
        body_op, _ = self._defocus_weight(0.0, extent=240.0)
        if body_op > 0.1:
            d_pts = np.array([[self._s(p[0]), self._s(p[1])]
                              for p in self._dorsal_pts], dtype=np.int32)
            v_pts = np.array([[self._s(p[0]), self._s(p[1])]
                              for p in self._ventral_pts], dtype=np.int32)
            body_poly = np.concatenate([d_pts, v_pts[::-1]])

            # Body fill: semi-transparent tissue (clearly distinct from bg)
            overlay = img.copy()
            cv2.fillPoly(overlay, [body_poly], 172.0)
            alpha = 0.45 * body_op
            img = img * (1 - alpha) + overlay * alpha

            # Body edge lines (distinct, dark outlines)
            thick = max(1, self._s(0.8))
            edge_val = float(130 * body_op)
            cv2.polylines(img, [d_pts], False, edge_val,
                          thick, lineType=cv2.LINE_AA)
            cv2.polylines(img, [v_pts], False, edge_val,
                          thick, lineType=cv2.LINE_AA)

        # ── Median fin fold (very faint transparent membrane) ──
        fin_op, _ = self._defocus_weight(self._fin_z, extent=15.0)
        if fin_op > 0.1:
            for fin_pts in [self._fin_dorsal_pts, self._fin_ventral_pts]:
                fp = np.array([[self._s(p[0]), self._s(p[1])]
                               for p in fin_pts], dtype=np.int32)
                fin_overlay = img.copy()
                cv2.fillPoly(fin_overlay, [fp], 190.0)
                al = 0.12 * fin_op  # very faint
                img = img * (1 - al) + fin_overlay * al
                # Thin edge line
                cv2.polylines(img, [fp[:len(fp)//2]], False,
                              float(175 * fin_op),
                              max(1, self._s(0.3)), lineType=cv2.LINE_AA)

        # ── Brain region (darker head interior) ──
        brain_op, _ = self._defocus_weight(self._brain_z, extent=35.0)
        if brain_op > 0.1:
            bx0 = self._s(self._brain_x0)
            bx1 = self._s(self._brain_x1)
            by0 = self._s(self._brain_y0)
            by1 = self._s(self._brain_y1)
            brain_mask = np.zeros((ih, iw), dtype=np.float32)
            cv2.ellipse(brain_mask,
                        ((bx0 + bx1) // 2, (by0 + by1) // 2),
                        ((bx1 - bx0) // 2, (by1 - by0) // 2),
                        0, 0, 360, 1.0, -1)
            brain_mask = cv2.GaussianBlur(brain_mask, (0, 0),
                                          sigmaX=self._sf(5.0))
            al = brain_mask * brain_op * 0.25
            img = img * (1 - al) + 155.0 * al

        # ── Yolk sac ──
        yolk_op, yolk_bl = self._defocus_weight(self._yolk_z, extent=90.0)
        if yolk_op > 0.08:
            yx = self._s(self._yolk_cx)
            yy = self._s(self._yolk_cy)
            yrx = self._s(self._yolk_rx)
            yry = self._s(self._yolk_ry)

            yolk_mask = np.zeros((ih, iw), dtype=np.float32)
            cv2.ellipse(yolk_mask, (yx, yy), (yrx, yry), 0, 0, 360, 1.0, -1)

            # Granular lipid droplet texture
            if s > 1:
                tex = cv2.resize(self._yolk_tex, (iw, ih),
                                 interpolation=cv2.INTER_LINEAR)
            else:
                tex = self._yolk_tex
            yolk_val = 148.0 + tex * 8.0  # more granular contrast

            al = yolk_mask * yolk_op * 0.65
            if yolk_bl > 0.5:
                al = cv2.GaussianBlur(al, (0, 0),
                                       sigmaX=max(1, self._sf(yolk_bl)))
            img = img * (1 - al) + yolk_val * al

            # Yolk edge (prominent)
            cv2.ellipse(img, (yx, yy), (yrx, yry), 0, 0, 360,
                        float(125 * yolk_op),
                        max(1, self._s(1.2)), lineType=cv2.LINE_AA)

        # ── Yolk extension (thin tube connecting yolk to trunk) ──
        if yolk_op > 0.08:
            ye_x0 = self._s(self._yolk_ext_x0)
            ye_x1 = self._s(self._yolk_ext_x1)
            ye_y = self._s(self._yolk_ext_y)
            ye_hy = self._s(self._yolk_ext_hy)
            ye_val = float(158 * yolk_op)
            cv2.rectangle(img, (ye_x0, ye_y - ye_hy),
                          (ye_x1, ye_y + ye_hy), ye_val, -1)
            cv2.rectangle(img, (ye_x0, ye_y - ye_hy),
                          (ye_x1, ye_y + ye_hy),
                          float(138 * yolk_op), max(1, self._s(0.5)))

        # ── Eye (dark pigmented — most visible landmark) ──
        eye_op, _ = self._defocus_weight(self._eye_z, extent=45.0)
        if eye_op > 0.08:
            ex = self._s(self._eye_x)
            ey = self._s(self._eye_y)
            er = self._s(self._eye_radius)
            lr = self._s(self._lens_radius)

            # Retinal pigment epithelium (very dark ring)
            cv2.circle(img, (ex, ey), er,
                       float(65 - 20 * eye_op), -1, lineType=cv2.LINE_AA)
            # Choroid fissure (thin dark line across retina)
            cv2.line(img, (ex - er + self._s(2), ey),
                     (ex + er - self._s(2), ey),
                     float(50 * eye_op), max(1, self._s(0.5)),
                     lineType=cv2.LINE_AA)
            # Lens (brighter center disc — refractive)
            cv2.circle(img, (ex, ey), lr,
                       float(95 - 15 * eye_op), -1, lineType=cv2.LINE_AA)
            # Lens edge ring
            cv2.circle(img, (ex, ey), lr,
                       float(45 * eye_op), max(1, self._s(0.6)),
                       lineType=cv2.LINE_AA)

        # ── Otic vesicle (small circle behind eye) ──
        otic_op, _ = self._defocus_weight(self._otic_z, extent=20.0)
        if otic_op > 0.1 and self.current_objectiv >= 20:
            ox = self._s(self._otic_x)
            oy = self._s(self._otic_y)
            ore = self._s(self._otic_r)
            # Hollow circle with faint interior
            cv2.circle(img, (ox, oy), ore, float(160 * otic_op),
                       -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (ox, oy), ore, float(135 * otic_op),
                       max(1, self._s(0.8)), lineType=cv2.LINE_AA)
            # Otolith (tiny bright spot inside)
            cv2.circle(img, (ox + self._s(2), oy - self._s(1)),
                       max(1, self._s(2.0)),
                       float(200 * otic_op), -1)

        # ── Melanophore pigment spots ──
        mel_op, _ = self._defocus_weight(self._melanophore_z, extent=8.0)
        if mel_op > 0.1:
            for mx, my, mr in self._melanophores:
                imx = self._s(mx)
                imy = self._s(my)
                imr = max(1, self._s(mr))
                # Dendritic shape approximation: dark blob + extensions
                cv2.circle(img, (imx, imy), imr,
                           float(75 - 30 * mel_op), -1, lineType=cv2.LINE_AA)
                if mr > 2.5 and self.current_objectiv >= 20:
                    # Dendritic arms at high magnification
                    arm_len = self._s(mr * 1.5)
                    for ang in [0, 60, 120, 200, 280]:
                        dx = int(arm_len * np.cos(np.radians(ang)))
                        dy = int(arm_len * np.sin(np.radians(ang)))
                        cv2.line(img, (imx, imy), (imx + dx, imy + dy),
                                 float(100 - 25 * mel_op), 1,
                                 lineType=cv2.LINE_AA)

        # ── Notochord (bright rod) ──
        noto_op, _ = self._defocus_weight(self._noto_z, extent=10.0)
        if noto_op > 0.1:
            nx0 = self._s(self._noto_x0)
            nx1 = self._s(self._noto_x1)
            ny = self._s(self._noto_y)
            nw = max(1, self._s(self._noto_hw))

            nval = 208.0 * noto_op
            cv2.rectangle(img, (nx0, ny - nw), (nx1, ny + nw),
                          float(nval), -1)
            # Notochord edges (visible boundary lines)
            cv2.line(img, (nx0, ny - nw), (nx1, ny - nw),
                     float(175 * noto_op), 1, lineType=cv2.LINE_AA)
            cv2.line(img, (nx0, ny + nw), (nx1, ny + nw),
                     float(175 * noto_op), 1, lineType=cv2.LINE_AA)

            # Vacuolated cell boundaries (visible at high mag)
            if self.current_objectiv >= 20:
                n_cells = 55
                cell_xs = np.linspace(nx0 + self._s(3), nx1 - self._s(3),
                                      n_cells)
                for cx in cell_xs:
                    cv2.line(img, (int(cx), ny - nw + 1),
                             (int(cx), ny + nw - 1),
                             float(190.0 * noto_op), 1)

        # ── Somites (chevron lines — clearly segmented) ──
        som_op, _ = self._defocus_weight(self._somite_z, extent=20.0)
        if som_op > 0.03:
            ny_int = self._s(self._noto_y)
            half_h = self._s(28)
            ang = np.radians(self._somite_chevron_angle)
            dx = int(half_h * np.tan(ang))
            thick = max(1, self._s(0.5))
            val = float(165 * som_op)

            for sx in self._somite_xs:
                sxi = self._s(sx)
                cv2.line(img, (sxi, ny_int), (sxi + dx, ny_int - half_h),
                         val, thick, lineType=cv2.LINE_AA)
                cv2.line(img, (sxi, ny_int), (sxi + dx, ny_int + half_h),
                         val, thick, lineType=cv2.LINE_AA)

        # ── Swim bladder primordia (48 hpf — just inflating) ──
        # Clear round structure visible ventral to notochord, between DA and gut.
        # Appears as a bright lumen (air-filled = optically clear) with thin dark wall.
        sb_op, _ = self._defocus_weight(self._sb_z, extent=14.0)
        if sb_op > 0.05:
            sbx = self._s(self._sb_x)
            sby = self._s(self._sb_y)
            sbr = max(2, self._s(self._sb_r))
            # Bright lumen (partially inflated — bright because optically clear)
            cv2.circle(img, (sbx, sby), sbr,
                       float(215.0 * sb_op), -1, lineType=cv2.LINE_AA)
            # Thin dark epithelial wall (single-cell layer)
            wall_thick = max(1, self._s(0.8))
            cv2.circle(img, (sbx, sby), sbr,
                       float(155.0 * sb_op), wall_thick, lineType=cv2.LINE_AA)
            # Slight blur for glass-like transparency
            if sb_op > 0.5 and self.current_objectiv <= 20:
                # At low mag, just a bright oval
                pass
            # Inner highlight (reflection/refraction at top of sphere)
            highlight_x = sbx - self._s(self._sb_r * 0.3)
            highlight_y = sby - self._s(self._sb_r * 0.25)
            highlight_r = max(1, self._s(self._sb_r * 0.25))
            cv2.circle(img, (highlight_x, highlight_y), highlight_r,
                       float(220.0 * sb_op), -1, lineType=cv2.LINE_AA)

        # ── Pericardial cavity (scales with edema) ──
        h_op, _ = self._defocus_weight(self._heart_z, extent=28.0)
        if h_op > 0.1:
            # Pericardial space — inflated by edema (fluid accumulation)
            edema = self._pericardial_edema
            pcx = self._s((self._vent_x + self._atri_x) / 2)
            pcy = self._s((self._vent_y + self._atri_y) / 2)
            # Normal: 24px radius; edema adds 8px per severity level
            pcr = self._s(24.0 + edema * 8.0)
            peri_mask = np.zeros((ih, iw), dtype=np.float32)
            cv2.circle(peri_mask, (pcx, pcy), pcr, 1.0, -1)
            peri_mask = cv2.GaussianBlur(peri_mask, (0, 0),
                                          sigmaX=self._sf(4.0 + edema))
            # More transparent with edema (fluid is watery)
            al = peri_mask * h_op * (0.15 + edema * 0.12)
            img = img * (1 - al) + (195.0 + edema * 2) * al

            # Heart chambers (dark beating)
            v_r, _ = self._ventricle_state()
            a_r, _ = self._atrium_state()

            vx, vy = self._s(self._vent_x), self._s(self._vent_y)
            vr = max(2, self._s(v_r))
            ax, ay = self._s(self._atri_x), self._s(self._atri_y)
            ar = max(2, self._s(a_r))

            cv2.circle(img, (vx, vy), vr, float(128 - 35 * h_op),
                       -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (ax, ay), ar, float(132 - 28 * h_op),
                       -1, lineType=cv2.LINE_AA)
            # Myocardial wall outlines
            cv2.circle(img, (vx, vy), vr, float(160 * h_op),
                       max(1, self._s(1.2)), lineType=cv2.LINE_AA)
            cv2.circle(img, (ax, ay), ar, float(160 * h_op),
                       max(1, self._s(1.2)), lineType=cv2.LINE_AA)

        # ── Blood cells (faint at 20x, clear at 40x+) ──
        if self.current_objectiv >= 20:
            rbc_xs, rbc_ys, rbc_zs = self._rbc_positions()
            # Contrast scales with magnification
            contrast = 0.3 if self.current_objectiv == 20 else 1.0
            for i in range(self._n_rbc):
                rop, _ = self._defocus_weight(rbc_zs[i], extent=2.0)
                if rop < 0.15:
                    continue
                rx = self._s(rbc_xs[i])
                ry = self._s(rbc_ys[i])
                rr = max(1, self._s(self._rbc_size[i]))
                val = float(img.mean() - (img.mean() - 100 + 50 * rop) * contrast)
                cv2.circle(img, (rx, ry), rr, val, -1)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_vascular_gfp(self):
        """Tg(flk1:GFP) — all vascular endothelium glows."""
        s = self.internal_scale
        ih, iw = self._ih, self._iw
        img = np.zeros((ih, iw), dtype=np.float32)

        # ── DA ──
        da_op, _ = self._defocus_weight(self._da_z, extent=7.0)
        if da_op > 0.05:
            x0 = self._s(self._da_x0)
            x1 = self._s(self._da_x1)
            dy = self._s(self._da_y)
            dr = max(1, self._s(self._da_r))
            cv2.line(img, (x0, dy), (x1, dy),
                     float(self._da_gfp * da_op), dr * 2, lineType=cv2.LINE_AA)

        # ── PCV ──
        pcv_op, _ = self._defocus_weight(self._pcv_z, extent=6.0)
        if pcv_op > 0.05:
            x0 = self._s(self._pcv_x0)
            x1 = self._s(self._pcv_x1)
            py = self._s(self._pcv_y)
            pr = max(1, self._s(self._pcv_r))
            cv2.line(img, (x0, py), (x1, py),
                     float(self._pcv_gfp * pcv_op), pr * 2,
                     lineType=cv2.LINE_AA)

        # ── ISVs (Z varies along length, respects growth fraction) ──
        n_segs = 6
        for j, ix_w in enumerate(self._isv_xs):
            growth = self._isv_growth[j]
            if growth < 0.01:
                continue  # absent ISV
            for seg in range(n_segs):
                f0 = seg / n_segs
                f1 = (seg + 1) / n_segs
                # Skip segments beyond the growth fraction
                if f0 >= growth:
                    break
                f1 = min(f1, growth)
                fm = (f0 + f1) / 2.0

                y0 = self._isv_y_bot - f0 * (self._isv_y_bot - self._isv_y_top)
                y1 = self._isv_y_bot - f1 * (self._isv_y_bot - self._isv_y_top)
                z_seg = (self._isv_z_bot
                         + fm * (self._isv_z_top - self._isv_z_bot))

                iop, _ = self._defocus_weight(z_seg, extent=3.0)
                if iop < 0.05:
                    continue

                ix = self._s(ix_w)
                iy0 = self._s(y0)
                iy1 = self._s(y1)
                ir = max(1, self._s(self._isv_r))
                cv2.line(img, (ix, iy0), (ix, iy1),
                         float(self._isv_gfp[j] * iop),
                         ir * 2, lineType=cv2.LINE_AA)

        # ── DLAV (gaps where ISVs are absent/truncated) ──
        dlav_op, _ = self._defocus_weight(self._dlav_z, extent=3.0)
        if dlav_op > 0.05:
            dly = self._s(self._dlav_y)
            dlr = max(1, self._s(self._dlav_r))
            dlav_val = float(self._dlav_gfp * dlav_op)
            # Draw DLAV segments between ISVs that reach the top
            for j in range(self._n_isv - 1):
                # Only draw segment if both flanking ISVs reach full growth
                if (self._isv_growth[j] >= 0.95
                        and self._isv_growth[j + 1] >= 0.95):
                    sx0 = self._s(self._isv_xs[j])
                    sx1 = self._s(self._isv_xs[j + 1])
                    cv2.line(img, (sx0, dly), (sx1, dly),
                             dlav_val, dlr * 2, lineType=cv2.LINE_AA)

        # ── Heart endocardium ──
        h_op, _ = self._defocus_weight(self._heart_z, extent=28.0)
        if h_op > 0.08:
            v_r, _ = self._ventricle_state()
            a_r, _ = self._atrium_state()
            vx, vy = self._s(self._vent_x), self._s(self._vent_y)
            ax, ay = self._s(self._atri_x), self._s(self._atri_y)
            vr = max(2, self._s(v_r))
            ar = max(2, self._s(a_r))
            endo_val = 115.0 * h_op
            cv2.circle(img, (vx, vy), vr, float(endo_val),
                       max(1, self._s(1.3)), lineType=cv2.LINE_AA)
            cv2.circle(img, (ax, ay), ar, float(endo_val),
                       max(1, self._s(1.3)), lineType=cv2.LINE_AA)

        # ── RBC dark shadows in GFP lumen (≥20x) ──
        # In Tg(flk1:GFP) images, RBCs displace GFP+ endothelium and appear
        # as dark moving voids inside the bright vessel lumen.  This is one of
        # the most characteristic features of in vivo zebrafish vascular imaging.
        if self.current_objectiv >= 20:
            rbc_xs, rbc_ys, rbc_zs = self._rbc_positions()
            for i in range(self._n_rbc):
                rop, _ = self._defocus_weight(rbc_zs[i], extent=2.0)
                if rop < 0.15:
                    continue
                rx = self._s(rbc_xs[i])
                ry = self._s(rbc_ys[i])
                # Slightly smaller radius than BF (only displaces inner lumen GFP)
                rr = max(1, self._s(self._rbc_size[i] * 0.85))
                # Dark shadow: near-zero GFP where RBC occludes the lumen
                dark_val = float(1.5 * max(0.0, 1.0 - rop))
                cv2.circle(img, (rx, ry), rr, dark_val, -1, lineType=cv2.LINE_AA)

        img += 1.5  # autofluorescence
        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_cardiac_gfp(self):
        """Tg(myl7:mCherry) — cardiac myocytes, oscillating brightness."""
        ih, iw = self._ih, self._iw
        img = np.zeros((ih, iw), dtype=np.float32)

        h_op, h_blur = self._defocus_weight(self._heart_z, extent=28.0)
        if h_op > 0.05:
            v_r, v_wf = self._ventricle_state()
            a_r, a_wf = self._atrium_state()

            vx, vy = self._s(self._vent_x), self._s(self._vent_y)
            vr = max(2, self._s(v_r))
            ax, ay = self._s(self._atri_x), self._s(self._atri_y)
            ar = max(2, self._s(a_r))

            # Ventricle myocardium (filled ring)
            v_inner = max(1, int(vr * 0.65))
            v_val = self._cardiac_gfp * h_op * v_wf
            cv2.circle(img, (vx, vy), vr, float(v_val),
                       -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (vx, vy), v_inner, 0.0,
                       -1, lineType=cv2.LINE_AA)

            # Atrium (slightly dimmer)
            a_inner = max(1, int(ar * 0.68))
            a_val = self._cardiac_gfp * h_op * a_wf * 0.85
            cv2.circle(img, (ax, ay), ar, float(a_val),
                       -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (ax, ay), a_inner, 0.0,
                       -1, lineType=cv2.LINE_AA)

            # AV canal
            avc_x = self._s((self._vent_x + self._atri_x) / 2)
            avc_y = self._s((self._vent_y + self._atri_y) / 2)
            avc_r = self._s(3.5)
            cv2.circle(img, (avc_x, avc_y), avc_r,
                       float((v_val + a_val) * 0.35), -1)

            if h_blur > 0.5:
                sigma = max(1, self._sf(h_blur))
                img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)

        img += 1.0
        return np.clip(img, 0, 255).astype(np.uint8)

    def _crop_fov(self, full_img):
        s = self.internal_scale
        fov_map = {10: 512, 20: 256, 40: 128, 100: 64}
        fov_world = fov_map.get(self.current_objectiv, 512)
        fov_int = fov_world * s

        ih, iw = full_img.shape[:2]

        # Center FOV on stage position (camera_offset + viewport//2 = stage_pos)
        cx = (int(self.camera_offset[0]) + self.viewport_width // 2) * s
        cy = (int(self.camera_offset[1]) + self.viewport_height // 2) * s
        half = fov_int // 2
        ox = max(0, min(cx - half, iw - fov_int))
        oy = max(0, min(cy - half, ih - fov_int))

        crop = full_img[oy:oy + fov_int, ox:ox + fov_int]

        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            bg = 192 if self.mode == 0 else 0
            padded = np.full((fov_int, fov_int), bg, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        out_w, out_h = self.viewport_width, self.viewport_height
        interp = cv2.INTER_AREA if fov_int > out_w else cv2.INTER_LINEAR
        return cv2.resize(crop, (out_w, out_h), interpolation=interp)

    # ── Device state ──

    def _read_temperature(self):
        if "Temperature" not in self.state_devices:
            return
        ts = self.state_devices["Temperature"]
        label = ts.get("label", "28")
        try:
            temp = float(label)
        except (ValueError, TypeError):
            temp = 28.0
        if temp != self._temperature:
            self._temperature = temp
        self._update_cardiac_freq()

    def _read_anesthesia(self):
        """Read Tricaine anesthesia state → cardiac suppression factor."""
        if "Anesthesia" not in self.state_devices:
            return
        ts = self.state_devices["Anesthesia"]
        state = ts.get("state", 0)
        if isinstance(state, str):
            try:
                state = int(state)
            except (ValueError, TypeError):
                state = 0
        # State 0=None(1.0), 1=0.01%(0.5), 2=0.02%(0.2), 3=0.04%(0.05)
        factors = [1.0, 0.50, 0.20, 0.05]
        new_factor = factors[min(state, len(factors) - 1)]
        if new_factor != self._anesthesia_factor:
            self._anesthesia_factor = new_factor
            self._update_cardiac_freq()

    def _update_cardiac_freq(self):
        """Recompute cardiac frequency from temperature + anesthesia."""
        q10 = self._q10 ** ((self._temperature - 28.0) / 10.0)
        self._cardiac_freq = (self._base_cardiac_freq * q10
                              * self._anesthesia_factor)

    # ── Dynamics ──

    def step(self, dt: float = 1.0):
        """Advance: cardiac phase, blood cell positions, Z-drift."""
        q10 = self._q10 ** ((self._temperature - 28.0) / 10.0)
        af = self._anesthesia_factor  # cardiac & flow suppression
        self._time += dt

        # Move blood cells (pulsatile flow, suppressed by anesthesia)
        phi = self._cardiac_phase()
        da_spd = self._da_speed * (1.0 + 0.4 * np.sin(phi)) * q10 * af
        pcv_spd = self._pcv_speed * (1.0 + 0.15 * np.sin(phi)) * q10 * af
        isv_spd = self._isv_speed * q10 * af

        da_len = self._da_x1 - self._da_x0
        pcv_len = self._pcv_x1 - self._pcv_x0
        isv_len = self._isv_y_bot - self._isv_y_top

        da_mask = self._rbc_ptype == 0
        pcv_mask = self._rbc_ptype == 1
        isv_mask = self._rbc_ptype == 2

        self._rbc_param[da_mask] += da_spd * dt / da_len
        self._rbc_param[pcv_mask] += pcv_spd * dt / pcv_len
        self._rbc_param[isv_mask] += isv_spd * dt / isv_len
        self._rbc_param %= 1.0

        # Z-drift
        if self.z_drift_rate != 0 or self.z_drift_noise > 0:
            dz = self.z_drift_rate * dt
            if self.z_drift_noise > 0:
                dz += self._rng.normal(0, self.z_drift_noise * np.sqrt(dt))
            self.tissue_z += dz

    def step_autonomous(self, dt: float = 1.0):
        self.step(dt)

    # ── ISV defects ──

    def set_isv_defects(self, n_absent=0, n_truncated=0,
                        truncation_range=(0.2, 0.6), seed=None):
        """Introduce ISV growth defects (angiogenesis inhibitor phenotype).

        Args:
            n_absent: Number of ISVs to remove completely (growth=0)
            n_truncated: Number of ISVs with partial growth
            truncation_range: (min, max) fraction for truncated ISVs
            seed: Optional RNG seed for reproducibility
        """
        rng = np.random.default_rng(seed or self._seed + 9999)
        self._isv_growth[:] = 1.0

        indices = rng.permutation(self._n_isv)
        n_a = min(n_absent, self._n_isv)
        n_t = min(n_truncated, self._n_isv - n_a)

        for i in indices[:n_a]:
            self._isv_growth[i] = 0.0
        for i in indices[n_a:n_a + n_t]:
            self._isv_growth[i] = rng.uniform(*truncation_range)

    def get_isv_state(self):
        """Return per-ISV growth fractions and defect counts."""
        n_normal = int(np.sum(self._isv_growth >= 0.99))
        n_absent = int(np.sum(self._isv_growth < 0.01))
        n_truncated = self._n_isv - n_normal - n_absent
        return {
            "n_isv": self._n_isv,
            "n_normal": n_normal,
            "n_absent": n_absent,
            "n_truncated": n_truncated,
            "growth_fractions": self._isv_growth.tolist(),
        }

    # ── Pericardial edema ──

    def set_pericardial_edema(self, severity=1.0):
        """Set pericardial edema severity (toxicity phenotype).

        Args:
            severity: 0=normal, 1=mild, 2=moderate, 3=severe
                Edema inflates the pericardial sac, creating a visible
                fluid-filled space around the heart in BF.
        """
        self._pericardial_edema = float(severity)

    def get_pericardial_edema(self):
        return self._pericardial_edema

    # ── Ground truth ──

    def get_ground_truth(self):
        focal_z = self.focal_plane - self.tissue_z
        phi = self._cardiac_phase()
        v_r, _ = self._ventricle_state()
        a_r, _ = self._atrium_state()

        visible = []
        for name, z, ext in [
            ("yolk", self._yolk_z, 90.0),
            ("notochord", self._noto_z, 10.0),
            ("dorsal_aorta", self._da_z, 7.0),
            ("pcv", self._pcv_z, 6.0),
            ("heart", self._heart_z, 28.0),
            ("isv_base", self._isv_z_bot, 3.0),
            ("dlav", self._dlav_z, 3.0),
            ("eye", self._eye_z, 45.0),
            ("swim_bladder", self._sb_z, 14.0),
        ]:
            op, _ = self._defocus_weight(z, ext)
            if op > 0.3:
                visible.append(name)

        rbc_xs, rbc_ys, rbc_zs = self._rbc_positions()
        n_vis_rbc = sum(1 for i in range(self._n_rbc)
                        if self._defocus_weight(rbc_zs[i], 2.0)[0] > 0.3)

        q10 = self._q10 ** ((self._temperature - 28.0) / 10.0)
        af = self._anesthesia_factor
        da_spd_inst = self._da_speed * (1 + 0.4 * np.sin(phi)) * q10 * af

        return {
            "focal_z_um": round(float(focal_z), 1),
            "cardiac_freq_hz": round(float(self._cardiac_freq), 2),
            "cardiac_bpm": round(float(self._cardiac_freq * 60), 0),
            "cardiac_phase_rad": round(float(phi % (2 * np.pi)), 2),
            "ventricle_radius_px": round(float(v_r), 1),
            "atrium_radius_px": round(float(a_r), 1),
            "temperature_c": round(float(self._temperature), 1),
            "anesthesia_factor": round(float(af), 2),
            "pericardial_edema": round(float(self._pericardial_edema), 1),
            "n_rbc": self._n_rbc,
            "n_visible_rbc": n_vis_rbc,
            "da_speed_um_s": round(float(da_spd_inst * 5.0), 1),
            "visible_structures": visible,
            "n_isv": self._n_isv,
            "n_isv_normal": int(np.sum(self._isv_growth >= 0.99)),
            "n_isv_absent": int(np.sum(self._isv_growth < 0.01)),
            "time_s": round(float(self._time), 2),
        }

    def get_heart_rate(self):
        return round(float(self._cardiac_freq * 60), 1)

    def get_z_anatomy(self):
        return {
            "yolk_z": self._yolk_z,
            "pcv_z": self._pcv_z,
            "notochord_z": self._noto_z,
            "da_z": self._da_z,
            "isv_z_range": (self._isv_z_bot, self._isv_z_top),
            "dlav_z": self._dlav_z,
            "heart_z": self._heart_z,
            "eye_z": self._eye_z,
            "swim_bladder_z": self._sb_z,
            "swim_bladder_x": self._sb_x,
            "swim_bladder_y": self._sb_y,
            "swim_bladder_r": self._sb_r,
        }
