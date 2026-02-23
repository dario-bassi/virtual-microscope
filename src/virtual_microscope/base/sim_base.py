"""SimBase — Abstract base class for microscope simulation backends.

Extracts ~1800 lines of duplicated boilerplate shared across 20+ standalone
sims into a single ABC.  Every microscope-style backend (i.e. one that has
objectives, FOV cropping, defocus, etc.) should inherit from SimBase and
implement :meth:`_render_for_mode`.

Non-microscope sims (PlateReaderSim, FlowCytometrySim, GelDocSim,
HemocytometerSim, ColonySim) stay duck-typed and do NOT need this base.
"""

from abc import ABC, abstractmethod

import cv2
import numpy as np


class SimBase(ABC):
    """Abstract base class providing the SimulationBridge-compatible interface.

    Subclasses must implement :meth:`_render_for_mode`.  The canonical
    ``snap_frame()`` pipeline is provided as a concrete template method.
    Override ``_handle_mask()`` for SLM / optogenetic / drug stimulation,
    and ``_finalize_output()`` to change the output format (e.g. RGB).

    Class attributes:
        continuous: Set to ``True`` on dynamic backends whose simulation
            should advance in real-time between snaps.  ``load_cfg()`` uses
            this to auto-start a :class:`RealtimeEngine`.
    """

    continuous: bool = False

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        viewport_width: int = 512,
        viewport_height: int = 512,
        seed: int = 42,
        internal_scale: int = 4,
        mode_map: dict | None = None,
        fixed_dt: float = 0.0,
        auto_step: bool = False,
        snaps_per_step: int = 1,
    ):
        # ── World dimensions ──
        self.width = width
        self.height = height
        self.internal_scale = internal_scale
        self._iw = width * internal_scale
        self._ih = height * internal_scale
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # ── Camera / state (SimulationBridge interface) ──
        self.camera_offset = np.array([
            (width - viewport_width) / 2.0,
            (height - viewport_height) / 2.0,
        ])
        self.focal_plane = 0.0
        self.tissue_z = 0.0
        self.state_devices: dict = {}
        self.mode = 0
        self.current_objectiv = 10

        # ── Objective / DOF tables ──
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40, "100x": 100}
        self._dof_table = {10: 6.0, 20: 4.0, 40: 1.5, 100: 0.6}
        self._dof = 6.0
        self._blur_scale_table = {10: 0.3, 20: 0.5, 40: 1.0, 100: 2.0}

        # ── Channel routing ──
        self._extra_channels: dict = {}
        self._mode_map: dict = mode_map if mode_map is not None else {}

        # ── Auto-step control ──
        self.auto_step = auto_step
        self.snaps_per_step = snaps_per_step
        self.fixed_dt = fixed_dt

        # ── Timing / snap counter ──
        self._snap_count = 0
        self._time = 0.0

        # ── Z-drift ──
        self.z_drift_rate = 0.0
        self.z_drift_noise = 0.0

        # ── Stage drift ──
        self.stage_drift_rate = 0.0
        self.stage_drift_noise = 0.0
        self._drift_accumulator = np.array([0.0, 0.0])

        # ── Optical pipelines (subclasses populate) ──
        self._pipeline: dict = {}

        # ── RNG ──
        self.rng = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────
    # Coordinate scaling
    # ──────────────────────────────────────────────────────────

    def _s(self, v):
        """Scale world coordinate to internal resolution (int)."""
        return int(round(v * self.internal_scale))

    def _sf(self, v):
        """Scale world coordinate to internal resolution (float)."""
        return v * self.internal_scale

    # ──────────────────────────────────────────────────────────
    # Device-state helpers
    # ──────────────────────────────────────────────────────────

    def _update_mode(self):
        """Update rendering mode via ``_mode_map`` lookup."""
        if "Filter Wheel" not in self.state_devices or "LED" not in self.state_devices:
            return  # keep current mode when no devices are registered
        filt = self.state_devices["Filter Wheel"]
        led = self.state_devices["LED"]
        filter_label = filt.get("label", filt.get("Label", ""))
        led_label = led.get("label", led.get("Label", ""))

        key = (filter_label, led_label)
        if key in self._mode_map:
            self.mode = self._mode_map[key]
            return
        for mode_id, ch_info in self._extra_channels.items():
            if filter_label == ch_info["filter"] and led_label == ch_info["led"]:
                self.mode = mode_id
                return
        self.mode = 0

    def _update_objectif(self):
        """Update objective magnification from device state."""
        if "Objective" not in self.state_devices:
            return
        obj = self.state_devices["Objective"]
        lbl = obj.get("Label", obj.get("label", ""))
        if lbl in self._objectif_dict:
            mag = self._objectif_dict[lbl]
            dof = self._dof_table.get(mag, 6.0)
            self.current_objectiv = mag
            self._dof = dof
            self._on_objective_changed(mag, dof)

    def _on_objective_changed(self, mag: int, dof: float):
        """Hook called when the objective changes. Override in subclasses."""

    def set_focal_plane(self, z: float):
        """Set focal plane position (µm)."""
        self.focal_plane = z

    def update_state(self, dict_state: dict):
        """Replace device-state dict (called by SimulationBridge)."""
        self.state_devices = dict_state

    # ──────────────────────────────────────────────────────────
    # FOV cropping
    # ──────────────────────────────────────────────────────────

    def _crop_fov(self, full):
        """Crop FOV from internal-resolution buffer, resize to viewport."""
        s = self.internal_scale
        ih, iw = full.shape[:2]
        out_w, out_h = self.viewport_width, self.viewport_height
        obj = self.current_objectiv

        fov_map = {100: 64, 40: 128, 20: 256}
        fov_world = fov_map.get(obj, min(512, self.width))
        fov_int = fov_world * s

        # Stage center in world coords → internal coords
        cx_world = int(self.camera_offset[0] + self._drift_accumulator[0]) + out_w // 2
        cy_world = int(self.camera_offset[1] + self._drift_accumulator[1]) + out_h // 2
        cx_int = int(cx_world * s)
        cy_int = int(cy_world * s)

        half = fov_int // 2
        x0 = max(0, min(cx_int - half, iw - fov_int))
        y0 = max(0, min(cy_int - half, ih - fov_int))

        crop = full[y0:y0 + fov_int, x0:x0 + fov_int].copy()

        # Pad if crop extends beyond full image
        ch = crop.shape[2] if crop.ndim == 3 else 0
        if crop.shape[0] < fov_int or crop.shape[1] < fov_int:
            bg = self._get_pad_bg()
            if ch > 0:
                padded = np.full((fov_int, fov_int, ch), bg, dtype=crop.dtype)
            else:
                padded = np.full((fov_int, fov_int), bg, dtype=crop.dtype)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        if crop.shape[0] > out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        elif crop.shape[0] < out_h:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return crop

    def _get_pad_bg(self) -> int:
        """Background value for out-of-bounds padding in ``_crop_fov``."""
        return 140 if self.mode == 0 else 0

    # ──────────────────────────────────────────────────────────
    # Defocus
    # ──────────────────────────────────────────────────────────

    def _apply_defocus(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian defocus blur based on distance from focal plane."""
        dz = abs(self.focal_plane - self.tissue_z)
        half_dof = self._dof / 2.0
        if dz <= half_dof:
            return img
        sigma = min((dz - half_dof) * self._blur_scale_table.get(
            self.current_objectiv, 0.5), 30.0)
        if sigma < 0.3:
            return img
        return cv2.GaussianBlur(img, (0, 0), sigma)

    # ──────────────────────────────────────────────────────────
    # Exposure & pipeline helpers
    # ──────────────────────────────────────────────────────────

    def _apply_exposure(self, viewport: np.ndarray, exposure: float,
                        intensity: float) -> np.ndarray:
        """Apply exposure and intensity scaling.

        BF (mode 0) uses 2x base so default exposure=50 gives full contrast.
        Fluorescence uses standard 1x photon-collection model.
        """
        if self.mode == 0:
            scale = min(intensity * 0.02 * exposure, 2.0)
        else:
            scale = intensity * 0.01 * exposure
        return (viewport.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)

    def _apply_pipeline(self, viewport: np.ndarray,
                        exposure: float) -> np.ndarray:
        """Apply optical pipeline (noise, PSF, vignette, bleaching)."""
        if self.mode in self._pipeline:
            pipe = self._pipeline[self.mode]
            if pipe.photobleach_rate > 0 and self.mode > 0:
                viewport = pipe.apply_with_bleach(viewport, exposure_ms=exposure)
            else:
                viewport = pipe.apply(viewport, exposure_ms=exposure)
        return viewport

    # ──────────────────────────────────────────────────────────
    # Auto-step
    # ──────────────────────────────────────────────────────────

    def _auto_step_tick(self):
        """Call ``step()`` if auto-step is enabled and snap count matches."""
        if (self.auto_step and self._snap_count > 0
                and self._snap_count % self.snaps_per_step == 0):
            self.step()

    # ──────────────────────────────────────────────────────────
    # Photobleaching
    # ──────────────────────────────────────────────────────────

    def enable_photobleaching(self, rate: float = 0.001):
        """Enable photobleaching on fluorescence channels."""
        for ch in [1, 2]:
            if ch in self._pipeline:
                self._pipeline[ch].photobleach_rate = rate

    def reset_photobleaching(self):
        """Reset accumulated photobleaching on all channels."""
        for pipe in self._pipeline.values():
            pipe.reset_bleach()

    # ──────────────────────────────────────────────────────────
    # Z-drift
    # ──────────────────────────────────────────────────────────

    def get_z_drift(self) -> float:
        """Return cumulative Z-drift (µm)."""
        return self.tissue_z

    def reset_z_drift(self):
        """Reset Z-drift to zero."""
        self.tissue_z = 0.0

    # ──────────────────────────────────────────────────────────
    # Template method — snap_frame pipeline
    # ──────────────────────────────────────────────────────────

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0,
                   **kwargs) -> np.ndarray:
        """Capture a rendered frame (template method).

        Orchestrates the canonical 9-step pipeline.  Subclasses should
        override ``_render_for_mode`` (required), ``_handle_mask``, or
        ``_finalize_output`` rather than replacing this method.
        """
        self._update_mode()
        self._update_objectif()
        if mask is not None:
            self._handle_mask(mask)
        self._auto_step_tick()
        self._snap_count += 1

        full = self._render_for_mode(self.mode)

        viewport = self._crop_fov(full)
        viewport = self._apply_defocus(viewport)
        viewport = self._apply_pipeline(viewport, exposure)
        viewport = self._apply_exposure(viewport, exposure, intensity)
        return self._finalize_output(viewport)

    # ──────────────────────────────────────────────────────────
    # Abstract / stubs
    # ──────────────────────────────────────────────────────────

    @abstractmethod
    def _render_for_mode(self, mode: int) -> np.ndarray:
        """Render full-resolution image for the given channel *mode*.

        Must return a BGR ``uint8`` image at internal resolution
        (``self._iw × self._ih``).  The base-class pipeline handles
        FOV cropping, defocus, noise, exposure, and grayscale conversion.
        """
        ...

    def _handle_mask(self, mask: np.ndarray) -> None:
        """Process an SLM / stimulation mask.  Override in subclasses."""

    def _finalize_output(self, viewport: np.ndarray) -> np.ndarray:
        """Convert viewport to final output format (BGR → grayscale)."""
        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def step(self, dt: float = 1.0):
        """Advance simulation by *dt*.  Override in dynamic sims."""

    def step_autonomous(self, dt: float = 1.0):
        """Background dynamics step (no imaging side effects)."""
        self.step(dt)

    def reset(self, seed: int | None = None):
        """Reset simulation state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def get_ground_truth(self) -> dict:
        """Return ground truth data.  Override in subclasses."""
        return {}
