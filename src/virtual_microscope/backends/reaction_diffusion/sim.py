"""
ReactionDiffusionSim — Gray-Scott reaction-diffusion pattern backend.

Renders traveling waves, spirals, and self-replicating spots from a
Gray-Scott activator-inhibitor system. Can be perturbed with optogenetics
(SLM mask locally changes feed/kill rates).

Usage via SimulationBridge:
    sim = ReactionDiffusionSim(grid_size=512, seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2
from virtual_microscope.base import SimBase
from virtual_microscope.pipeline.optical_pipeline import OpticalPipeline


class ReactionDiffusionSim(SimBase):
    """Gray-Scott reaction-diffusion simulation compatible with SimulationBridge.

    The system evolves two chemicals (U and V) on a 2D grid:
      dU/dt = Du * laplacian(U) - U*V^2 + F*(1-U)
      dV/dt = Dv * laplacian(V) + U*V^2 - (F+K)*V

    Different (F, K) parameters produce different patterns:
      - Spirals:  F=0.014, K=0.045
      - Spots:    F=0.035, K=0.065
      - Stripes:  F=0.025, K=0.060
      - Waves:    F=0.018, K=0.050

    Channels:
      - mode 0: Brightfield — phase-contrast-like view of both chemicals
      - mode 1: GFP-U — activator concentration (nucleus channel)
      - mode 2: mCherry-V — inhibitor concentration (membrane channel)

    Optogenetics: SLM mask in snap_frame locally increases feed rate F,
    nucleating new wave fronts in stimulated regions.
    """

    continuous = True
    _default_temperature = 25.0

    # Preset parameter sets (standard Du=0.21, Dv=0.105)
    PRESETS = {
        "spirals":  {"F": 0.026, "K": 0.051, "Du": 0.21, "Dv": 0.105},
        "spots":    {"F": 0.035, "K": 0.060, "Du": 0.21, "Dv": 0.105},
        "stripes":  {"F": 0.040, "K": 0.060, "Du": 0.21, "Dv": 0.105},
        "waves":    {"F": 0.018, "K": 0.050, "Du": 0.21, "Dv": 0.105},
        "mitosis":  {"F": 0.034, "K": 0.059, "Du": 0.21, "Dv": 0.105},
        "coral":    {"F": 0.058, "K": 0.063, "Du": 0.21, "Dv": 0.105},
    }

    def __init__(
        self,
        grid_size: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        preset: str = "waves",
        F: float = None,
        K: float = None,
        Du: float = None,
        Dv: float = None,
        dt: float = 1.0,
        steps_per_snap: int = 200,
        seed: int = 42,
        fixed_dt: float = 0.0,
        noise_amplitude: float = 0.001,
    ):
        super().__init__(
            width=grid_size, height=grid_size,
            viewport_width=viewport_width, viewport_height=viewport_height,
            seed=seed, internal_scale=1, fixed_dt=fixed_dt,
            auto_step=True, snaps_per_step=1,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
                ("mScarlet3(569/582)", "ORANGE"): 2,
            },
        )

        # Override: no 100x objective for this sim
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40}
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5}
        self._blur_scale_table = {10: 0.3, 20: 0.5, 40: 1.0}

        # Load preset parameters (can be overridden)
        params = self.PRESETS.get(preset, self.PRESETS["waves"]).copy()
        self.F = F if F is not None else params["F"]
        self.K = K if K is not None else params["K"]
        self.Du = Du if Du is not None else params["Du"]
        self.Dv = Dv if Dv is not None else params["Dv"]
        self.dt = dt
        self.steps_per_snap = steps_per_snap
        self.noise_amplitude = noise_amplitude

        self._noise_rng = np.random.default_rng(seed + 7777)

        # Optical pipelines per channel
        self._pipeline = {
            0: OpticalPipeline(
                psf_sigma=0.6, noise={"photon_scale": 6.0, "read_std": 2.0},
                vignette=0.08, rng_seed=seed + 200),
            1: OpticalPipeline(
                psf_sigma=1.0, noise={"photon_scale": 3.0, "read_std": 3.0},
                vignette=0.12, rng_seed=seed + 201),
            2: OpticalPipeline(
                psf_sigma=1.0, noise={"photon_scale": 3.0, "read_std": 3.0},
                vignette=0.12, rng_seed=seed + 202),
        }

        # Initialize grid
        self.U = np.ones((grid_size, grid_size), dtype=np.float64)
        self.V = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Seed initial perturbation
        self._seed_perturbation(seed)

        # Pre-evolve to get interesting patterns
        # Slower presets need more evolution time
        pre_steps = {"spots": 4000, "stripes": 4000, "coral": 3000,
                     "mitosis": 3000}.get(preset, 2000)
        self._evolve(pre_steps)

        # Optogenetic stimulation state
        self._stim_mask = None  # set by SLM
        self._stim_strength = 0.05  # local F increase when stimulated (excite)
        self._inhibit_strength = 0.02  # local K increase when inhibited
        self._slm_mode = 0  # 0 = excite (increase F), 1 = inhibit (increase K)

        # Stage drift attributes inherited from SimBase

    def _seed_perturbation(self, seed: int):
        """Add initial seed regions to break symmetry."""
        rng = np.random.default_rng(seed)
        n_seeds = rng.integers(4, 10)
        for _ in range(n_seeds):
            cx = rng.integers(60, self.width - 60)
            cy = rng.integers(60, self.height - 60)
            r = rng.integers(12, 30)
            Y, X = np.ogrid[:self.height, :self.width]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
            self.U[mask] = 0.50 + rng.uniform(-0.02, 0.02, mask.sum())
            self.V[mask] = 0.25 + rng.uniform(-0.02, 0.02, mask.sum())

    def _laplacian(self, Z: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian with periodic boundary conditions."""
        return (
            np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
            4 * Z
        )

    def _evolve(self, n_steps: int):
        """Evolve the Gray-Scott system for n_steps with stochastic noise."""
        U, V = self.U, self.V
        F, K = self.F, self.K
        Du, Dv = self.Du, self.Dv
        dt = self.dt
        noise_amp = self.noise_amplitude

        for _ in range(n_steps):
            uvv = U * V * V
            lu = self._laplacian(U)
            lv = self._laplacian(V)

            U += dt * (Du * lu - uvv + F * (1.0 - U))
            V += dt * (Dv * lv + uvv - (F + K) * V)

            # Stochastic noise (thermal fluctuations)
            if noise_amp > 0:
                V += noise_amp * self._noise_rng.normal(0, 1, V.shape) * dt

            # Clamp to valid range
            np.clip(U, 0, 1, out=U)
            np.clip(V, 0, 1, out=V)

        self.U = U
        self.V = V

    def _temp_rate_factor(self) -> float:
        """Temperature-dependent rate factor for Gray-Scott reaction-diffusion.

        Arrhenius-like scaling. Reaction and diffusion rates both increase
        with temperature. Reference at 25°C. Q10 ~ 2.0.
        """
        temp = self._get_temperature()
        if temp < 5:
            return 0.1
        return 2.0 ** ((temp - 25) / 10.0)

    def step(self, dt: float = 1.0):
        """Advance simulation, scaling PDE steps proportional to dt."""
        temp_factor = self._temp_rate_factor()
        n_steps = max(1, round(dt * self.steps_per_snap * temp_factor))

        if self._stim_mask is not None:
            stim = self._stim_mask.astype(np.float64)
            if self._slm_mode == 0:
                # EXCITE: increase F locally → nucleate new patterns
                F_map = self.F + stim * self._stim_strength
                K_val = self.K
            else:
                # INHIBIT: increase K locally → suppress patterns
                F_map = self.F
                K_val = self.K + stim * self._inhibit_strength
            self._evolve_with_stim(n_steps, F_map, K_val)
        else:
            self._evolve(n_steps)

        self._time += n_steps * self.dt

    def step_autonomous(self, dt: float = 1.0):
        """Advance simulation WITH SLM optogenetic effects.

        For reaction-diffusion, optogenetic illumination is continuous
        (independent of imaging), so the background thread must apply
        the SLM mask at every step — unlike photobleaching which is
        observation-coupled.
        """
        # Read SLM-Mode from state_devices (updated by core.setState)
        if "SLM-Mode" in self.state_devices:
            mode_dev = self.state_devices["SLM-Mode"]
            label = mode_dev.get("label", mode_dev.get("Label", "excite"))
            self._slm_mode = 1 if label == "inhibit" else 0

        temp_factor = self._temp_rate_factor()
        n_steps = max(1, round(dt * self.steps_per_snap * temp_factor))

        if self._stim_mask is not None:
            stim = self._stim_mask.astype(np.float64)
            if self._slm_mode == 0:
                F_map = self.F + stim * self._stim_strength
                K_val = self.K
            else:
                F_map = self.F
                K_val = self.K + stim * self._inhibit_strength
            self._evolve_with_stim(n_steps, F_map, K_val)
        else:
            self._evolve(n_steps)

        self._time += n_steps * self.dt

    def _evolve_with_stim(self, n_steps: int, F_map, K_map):
        """Evolve with spatially varying F and/or K (optogenetic stimulation).

        F_map and K_map can be scalar or 2D array — numpy broadcasts correctly.
        Noise is still active during stimulated evolution.
        """
        U, V = self.U, self.V
        Du, Dv = self.Du, self.Dv
        dt = self.dt
        noise_amp = self.noise_amplitude

        for _ in range(n_steps):
            uvv = U * V * V
            lu = self._laplacian(U)
            lv = self._laplacian(V)

            U += dt * (Du * lu - uvv + F_map * (1.0 - U))
            V += dt * (Dv * lv + uvv - (F_map + K_map) * V)

            if noise_amp > 0:
                V += noise_amp * self._noise_rng.normal(0, 1, V.shape) * dt

            np.clip(U, 0, 1, out=U)
            np.clip(V, 0, 1, out=V)

        self.U = U
        self.V = V

    # ── Rendering ──

    def _render_bf_full(self) -> np.ndarray:
        """Brightfield: transmitted-light view of thin-film chemical reaction.

        Real BZ/CIMA reactions viewed under transmitted light show:
        - Moderate contrast (~30-60% modulation, not full 0-255 range)
        - Smooth, diffuse pattern boundaries from chemical diffusion
        - Asymmetric DIC-like shading relief at concentration gradients
        - Low-frequency background variation from film thickness unevenness
        """
        # Smooth the concentration fields slightly (chemical diffusion blur)
        # to avoid overly sharp binary boundaries
        v_smooth = cv2.GaussianBlur(self.V.astype(np.float32), (0, 0), 1.2)
        u_smooth = cv2.GaussianBlur(self.U.astype(np.float32), (0, 0), 1.2)

        # Sigmoid transfer function: softens transitions, more realistic
        # than linear mapping. Centered at V=0.15 (typical wavefront threshold)
        v_sig = 1.0 / (1.0 + np.exp(-12 * (v_smooth - 0.15)))

        # Transmitted light: V-rich regions absorb more (darker).
        # Compressed range: background ~170, pattern regions ~95
        # (~44% modulation, realistic for thin-film chemistry)
        base = 170.0 - v_sig * 75.0

        # Low-frequency background variation (film thickness, illumination)
        bg_rng = np.random.default_rng(self.rng.integers(0, 2**31) + 8888)
        bg_noise = bg_rng.normal(0, 1, (self.height // 8, self.width // 8)).astype(np.float32)
        bg_noise = cv2.resize(bg_noise, (self.width, self.height),
                              interpolation=cv2.INTER_LINEAR)
        bg_noise = cv2.GaussianBlur(bg_noise, (0, 0), 8.0)
        base += bg_noise * 4.0  # ~2-3% amplitude variation

        # Asymmetric DIC-like shading: directional gradient creates
        # bright-on-one-side, dark-on-other relief appearance
        grad_x = cv2.Sobel(v_smooth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(v_smooth, cv2.CV_32F, 0, 1, ksize=3)
        # 45-degree illumination direction
        dic_relief = (grad_x * 0.7 + grad_y * 0.7) * 80
        base += np.clip(dic_relief, -15, 15)

        # Faint refractive halo at sharp wavefronts (phase-like edge enhancement)
        laplacian_v = cv2.Laplacian(v_smooth, cv2.CV_32F, ksize=3)
        base += np.clip(laplacian_v * 25, -8, 8)

        combined = np.clip(base, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        return img

    def _render_nuc_full(self) -> np.ndarray:
        """GFP-V channel: inhibitor concentration (fluorescent reporter).

        Fluorescence imaging of V (inhibitor) pattern — bright where
        concentration is high, dark background with realistic falloff.
        Note: snap_frame exposure scaling (0.5x at default 50ms) means
        raw values need to be high enough to remain visible after scaling.
        """
        v_smooth = cv2.GaussianBlur(self.V.astype(np.float32), (0, 0), 0.8)

        # Nonlinear mapping: emphasize high concentrations, suppress baseline
        # Sigmoid centered at V=0.10 with moderate steepness
        v_norm = 1.0 / (1.0 + np.exp(-10 * (v_smooth - 0.10)))

        # Scale to fluorescence range: peak ~240, background ~3
        # (high values compensate for 0.5x exposure scaling)
        v_img = (v_norm * 237 + 3).clip(0, 255)

        # OOF haze: faint broad glow around bright regions (defocused planes)
        haze = cv2.GaussianBlur(v_img, (0, 0), 12.0)
        v_img = v_img * 0.85 + haze * 0.15

        v_img = v_img.clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(v_img, cv2.COLOR_GRAY2BGR)
        return img

    def _render_mem_full(self) -> np.ndarray:
        """mCherry-U channel: activator depletion zones (1-U fluorescence).

        Where U is consumed (pattern regions), (1-U) is high — represents
        a reporter inversely coupled to the activator.
        """
        u_smooth = cv2.GaussianBlur(self.U.astype(np.float32), (0, 0), 0.8)

        # U ranges from ~0.4 (in pattern) to ~1.0 (background).
        # (1-U) is ~0.0 at background, ~0.6 in pattern.
        u_dep = 1.0 - u_smooth
        # Sigmoid: sharpen the transition, suppress noise at baseline
        u_norm = 1.0 / (1.0 + np.exp(-8 * (u_dep - 0.15)))

        # Scale to fluorescence: peak ~230, background ~3
        u_img = (u_norm * 227 + 3).clip(0, 255)

        # OOF haze
        haze = cv2.GaussianBlur(u_img, (0, 0), 10.0)
        u_img = u_img * 0.88 + haze * 0.12

        u_img = u_img.clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(u_img, cv2.COLOR_GRAY2BGR)
        return img

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply Poisson + Gaussian noise."""
        f = img.astype(np.float32)
        if f.max() > 0:
            photon_scale = 80.0
            photons = f / 255.0 * photon_scale
            noisy = self._noise_rng.poisson(np.clip(photons, 0, None))
            f = noisy.astype(np.float32) / photon_scale * 255.0
        f += self._noise_rng.normal(0, 2, f.shape).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    # ── Template-method hooks ──

    def _render_for_mode(self, mode):
        if mode == 0:
            return self._render_bf_full()
        elif mode == 1:
            return self._render_nuc_full()
        elif mode == 2:
            return self._render_mem_full()
        elif mode in self._extra_channels:
            return self._extra_channels[mode]["image"]
        return self._render_bf_full()

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Clear SLM stimulation each frame, then delegate to SimBase pipeline.

        The stim mask is ephemeral: active only for the snap in which it is
        supplied. Without this reset, a previously sent mask would persist
        across subsequent snaps that do not provide one.
        """
        if mask is None:
            self._stim_mask = None
        return super().snap_frame(mask=mask, exposure=exposure,
                                  intensity=intensity, **kwargs)

    def _handle_mask(self, mask):
        """Apply SLM optogenetic mask: downsample to grid and store for step().

        Reads SLM-Mode device to determine excite vs inhibit, then resizes the
        viewport-space mask to grid resolution for use in the PDE solver.
        """
        # Read SLM-Mode device if present (0=excite, 1=inhibit)
        if "SLM-Mode" in self.state_devices:
            mode_dev = self.state_devices["SLM-Mode"]
            label = mode_dev.get("label", mode_dev.get("Label", "excite"))
            self._slm_mode = 1 if label == "inhibit" else 0

        # Resize mask to grid size if needed
        if mask.shape != (self.height, self.width):
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (self.width, self.height),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_resized = mask.astype(bool)
        self._stim_mask = mask_resized

    def _get_pad_bg(self) -> int:
        """Background for out-of-bounds padding: 120 for BF, 0 for fluoro."""
        return 120 if self.mode == 0 else 0

    def reset(self, seed: int = None):
        """Reset to initial conditions."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.U = np.ones((self.height, self.width), dtype=np.float64)
        self.V = np.zeros((self.height, self.width), dtype=np.float64)
        self._seed_perturbation(seed or 42)
        self._evolve(2000)
        self._snap_count = 0
        self._time = 0.0

    # ── Ground truth for grading ──

    def get_wave_speed(self, n_steps: int = 500) -> float:
        """Estimate wave propagation speed by tracking front movement."""
        # Save state
        U_save = self.U.copy()
        V_save = self.V.copy()

        # Find current wave fronts (V > threshold)
        thresh = 0.15
        front_before = (self.V > thresh).astype(np.float64)
        centroid_before = np.array(np.where(front_before > 0)).mean(axis=1)

        # Evolve
        self._evolve(n_steps)
        front_after = (self.V > thresh).astype(np.float64)
        centroid_after = np.array(np.where(front_after > 0)).mean(axis=1)

        # Restore
        self.U = U_save
        self.V = V_save

        dist = np.linalg.norm(centroid_after - centroid_before)
        return dist / n_steps

    def get_pattern_stats(self) -> dict:
        """Return statistics about the current pattern."""
        v = self.V
        thresh = 0.10
        active = v > thresh
        n_active = int(active.sum())
        total = self.width * self.height
        coverage = round(n_active / total, 3)

        # Mean and max concentrations
        mean_v = round(float(v.mean()), 4)
        max_v = round(float(v.max()), 4)
        mean_u = round(float(self.U.mean()), 4)

        return {
            "coverage_v": coverage,
            "mean_v": mean_v,
            "max_v": max_v,
            "mean_u": mean_u,
            "n_active_pixels": n_active,
            "grid_size": self.width,
            "F": self.F,
            "K": self.K,
            "time": self._time,
        }

    def get_ground_truth(self) -> dict:
        """Return full ground truth for grading."""
        stats = self.get_pattern_stats()
        stats["slm_mode"] = "inhibit" if self._slm_mode == 1 else "excite"
        return stats
