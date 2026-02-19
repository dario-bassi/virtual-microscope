"""SimulationBridge — connects pymmcore device adapters to simulation backends.

The bridge is a singleton (GLOBAL_BRIDGE) that all device adapters reference.
Swap it to swap the entire simulated sample without reloading devices.

All simulation backends must implement:
  - snap_frame(mask=None, exposure=1.0, intensity=1.0) -> np.ndarray
  - camera_offset: np.ndarray  (x, y) rendering origin
  - state_devices: dict        {name: {state, label}}
  - set_focal_plane(z: float)
  - viewport_width, viewport_height: int
"""

import threading

import numpy as np

# Singleton — all device adapters import this module and read GLOBAL_BRIDGE
GLOBAL_BRIDGE = None

# Event set by SimServer.initialize() when GLOBAL_BRIDGE is ready.
# State devices wait on this during their own initialize() so they can push
# initial state to the bridge even when initializeAllDevices() runs in parallel.
bridge_ready = threading.Event()


class SimulationBridge:
    """Bridge between pymmcore device adapters and simulation backends.

    Acts as a duck-typed façade: any sim with snap_frame(), set_focal_plane(),
    camera_offset, state_devices, viewport_width/height is compatible.
    """

    # Half-width of the base 10x render in µm (512 px at 1 µm/px).
    _BASE_HALF = 256

    def __init__(self, microscope_sim) -> None:
        if microscope_sim is None:
            raise ValueError("The microscope simulation must be initialized.")
        self._sim = microscope_sim
        self._current_slm_mask = None
        self._stage_position = (0.0, 0.0)

    @property
    def sim(self):
        """Direct access to the underlying simulation object."""
        return self._sim

    def snap(self, exposure: float, brightness: float, gain: float = 1.0,
             **kwargs) -> np.ndarray:
        mask = self.get_slm_mask()
        img = self._sim.snap_frame(mask=mask, exposure=exposure,
                                   intensity=brightness, **kwargs)
        # Analog gain: amplifies signal AND noise (electronic amplification).
        if gain != 1.0:
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return img

    def set_stage(self, x: float, y: float) -> None:
        """Stage position is FOV center; convert to rendering origin (top-left of 10x image)."""
        self._stage_position = (x, y)
        self._sim.camera_offset = np.array([x - self._BASE_HALF,
                                             y - self._BASE_HALF])

    def set_focus(self, z: float) -> None:
        self._sim.set_focal_plane(z)

    def update_state(self, dict_state: dict) -> None:
        self._sim.state_devices.update(dict_state)

    def set_slm_mask(self, mask: np.ndarray) -> None:
        """Called by SLM device when pattern changes."""
        self._current_slm_mask = mask

    def get_slm_mask(self) -> np.ndarray:
        """Called by camera device when capturing."""
        if self._current_slm_mask is not None:
            return self._current_slm_mask
        # Default: no stimulation
        return np.zeros(
            (self._sim.viewport_height, self._sim.viewport_width), dtype=bool
        )
