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

    def __init__(self, microscope_sim, factory=None) -> None:
        if microscope_sim is None:
            raise ValueError("The microscope simulation must be initialized.")
        self._sim = microscope_sim
        self._current_slm_mask = None
        self._engine = None  # RealtimeEngine, set by load_cfg / SimServer
        self._slm_processor = None  # lazy-init SLMProcessor
        self._factory = factory  # callable(**kw) -> new sim instance
        self._factory_kwargs: dict = {}  # params used to create current sim

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
        """Stage position relative to world center; convert to rendering origin.

        (0, 0) centres the viewport on the world.  The stage coordinate
        is added to the world centre so positive values pan right/down.
        """
        sim = self._sim
        world_cx = sim.width / 2.0
        world_cy = sim.height / 2.0
        self._sim.camera_offset = np.array([
            world_cx + x - self._BASE_HALF,
            world_cy + y - self._BASE_HALF,
        ])

    def set_focus(self, z: float) -> None:
        self._sim.set_focal_plane(z)

    def update_state(self, dict_state: dict) -> None:
        self._sim.state_devices.update(dict_state)

    @property
    def slm_processor(self):
        """Lazily-initialized SLMProcessor for centralized mask handling."""
        if self._slm_processor is None:
            from virtual_microscope.engine.slm_processor import SLMProcessor
            sim = self._sim
            w = getattr(sim, 'width', getattr(sim, 'world_size', 512))
            h = getattr(sim, 'height', getattr(sim, 'world_size', 512))
            self._slm_processor = SLMProcessor(w, h)
        return self._slm_processor

    def set_slm_mask(self, mask: np.ndarray) -> None:
        """Called by SLM device when pattern changes.

        Updates the SLM processor's stimulation field and also propagates
        to the sim's ``_stim_mask`` so that background-thread step() calls
        (RealtimeEngine) pick up the mask immediately.
        """
        self._current_slm_mask = mask

        # Update SLM processor
        if mask is not None:
            sim = self._sim
            self.slm_processor.update_mask(
                mask,
                camera_offset=sim.camera_offset,
                objective_mag=getattr(sim, 'current_objectiv', 10),
                viewport_width=sim.viewport_width,
                viewport_height=sim.viewport_height,
            )

        # Backward compat: propagate to sim._stim_mask
        if hasattr(self._sim, '_stim_mask'):
            proc = self.slm_processor
            self._sim._stim_mask = proc.raw_mask if mask is not None else None

    def get_slm_mask(self) -> np.ndarray:
        """Called by camera device when capturing."""
        if self._current_slm_mask is not None:
            return self._current_slm_mask
        # Default: no stimulation
        return np.zeros(
            (self._sim.viewport_height, self._sim.viewport_width), dtype=bool
        )

    # ── Experiment lifecycle ──

    def reset_simulation(self, seed: int | None = None) -> None:
        """Soft reset: call sim.reset(seed) and clear SLM state.

        Keeps the same sim instance — just resets internal state.
        """
        if hasattr(self._sim, 'reset'):
            self._sim.reset(seed)
        if self._slm_processor is not None:
            self._slm_processor.reset()
        self._current_slm_mask = None

    def recreate_simulation(self, seed: int | None = None) -> None:
        """Hard reset: create a new sim via factory, transfer device state.

        Requires ``factory`` to have been passed at construction.
        """
        if self._factory is None:
            # Fallback to soft reset if no factory
            self.reset_simulation(seed)
            return

        # Build new sim with same params + updated seed
        kwargs = dict(self._factory_kwargs)
        if seed is not None:
            kwargs['seed'] = seed
        new_sim = self._factory(**kwargs)

        # Transfer device state and camera
        new_sim.state_devices = dict(self._sim.state_devices)
        new_sim.camera_offset = self._sim.camera_offset.copy()
        new_sim.focal_plane = self._sim.focal_plane

        # Swap sim reference
        old_sim = self._sim
        self._sim = new_sim

        # Update engine's sim reference if running
        if self._engine is not None:
            self._engine._sim = new_sim
            # Re-patch snap_frame
            self._engine.patch_snap_frame()

        # Reset SLM state
        if self._slm_processor is not None:
            self._slm_processor.reset()
        self._current_slm_mask = None


def set_global_bridge(bridge: SimulationBridge) -> None:
    """Replace the global bridge, stopping any running engine on the old one.

    Use this instead of assigning ``GLOBAL_BRIDGE`` directly so that
    a running RealtimeEngine is properly shut down before the sim reference
    becomes stale.
    """
    global GLOBAL_BRIDGE
    old = GLOBAL_BRIDGE
    if old is not None and hasattr(old, '_engine') and old._engine is not None:
        old._engine.stop()
        old._engine = None
    GLOBAL_BRIDGE = bridge
    bridge_ready.set()
