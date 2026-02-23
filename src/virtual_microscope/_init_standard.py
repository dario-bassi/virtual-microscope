"""Standard device stack initializer — shared by all programmatic setup_* functions.

The primary entry point is ``load_cfg(sim, cfg_path)`` which:
  1. Wires the simulation into the global bridge
  2. Loads the backend .cfg (devices, channels, everything)
  3. Installs pixel-size helper
  4. Optionally auto-starts the RealtimeEngine for continuous sims

For standalone (non-.cfg) backends, use ``load_standalone(sim, channels, ...)``.

Usage:
    from virtual_microscope._init_standard import load_cfg
    core = load_cfg(sim, Path(__file__).parent / "bacteria.cfg")

    from virtual_microscope._init_standard import load_standalone
    core = load_standalone(sim, channels={"brightfield": ("Channel", "Label", "bf")})
"""

import logging
from pathlib import Path

from pymmcore_plus.experimental.unicore import UniMMCore

import virtual_microscope.engine.simulation_bridge as bridge_module
from virtual_microscope.engine.simulation_bridge import SimulationBridge, set_global_bridge

logger = logging.getLogger(__name__)


def _install_pixel_size(core: UniMMCore, sim) -> None:
    """Install getPixelSizeUm() returning correct um/camera-px for current objective.

    Camera always outputs 512x512 px regardless of objective.
    FOV in world px: 10x=512, 20x=256, 40x=128, 100x=64.
    """
    world_px_um = getattr(sim, 'world_pixel_size_um', 1.0)
    _obj_factors = {0: 1, 1: 2, 2: 4, 3: 8}

    def _get_pixel_size_um(cached=False):
        obj_state = core.getState("Objective")
        factor = _obj_factors.get(obj_state, 1)
        return world_px_um / factor

    core.getPixelSizeUm = _get_pixel_size_um


def load_cfg(sim, cfg_path: Path, *,
             factory=None,
             factory_kwargs: dict | None = None,
             realtime: bool | None = None,
             time_scale: float = 1.0,
             tick_hz: int = 10,
             idle_timeout: float = 30.0) -> UniMMCore:
    """Load a backend .cfg after pre-creating the simulation with custom params.

    This is the single entry point for programmatic setup_*_microscope() calls.
    The .cfg is the sole source of truth for devices and channel definitions.

    Args:
        sim: Simulation backend instance (already created with custom params).
        cfg_path: Path to the backend's .cfg file.
        factory: Optional callable(**kwargs) that creates a new sim instance.
            Used by ``NewExperimentDevice`` for hard-reset (recreate).
        factory_kwargs: Dict of keyword arguments used to create the current
            sim.  Stored so ``recreate_simulation()`` can reproduce it.
        realtime: Whether to auto-start the RealtimeEngine.  ``None`` (default)
            auto-detects from ``sim.continuous``.  ``True`` forces it on,
            ``False`` forces it off.
        time_scale: Simulation time multiplier (1.0 = real-time).
        tick_hz: Background tick rate (steps/second).
        idle_timeout: Seconds without snaps before the engine auto-pauses
            (0 = never pause).

    Returns:
        Configured UniMMCore instance.
    """
    bridge = SimulationBridge(sim, factory=factory)
    if factory_kwargs:
        bridge._factory_kwargs = factory_kwargs
    set_global_bridge(bridge)
    core = UniMMCore()
    core.loadSystemConfiguration(str(cfg_path))
    _install_pixel_size(core, sim)

    # Auto-start real-time engine for continuous sims
    should_start = realtime if realtime is not None else getattr(sim, 'continuous', False)
    if should_start and hasattr(sim, 'step'):
        from virtual_microscope.engine.realtime import RealtimeEngine
        engine = RealtimeEngine(sim, time_scale=time_scale,
                                tick_hz=tick_hz, idle_timeout=idle_timeout,
                                bridge=bridge)
        engine.patch_snap_frame()
        engine.start()
        bridge._engine = engine
        logger.info("RealtimeEngine auto-started for %s (tick_hz=%d, idle_timeout=%.0fs)",
                    type(sim).__name__, tick_hz, idle_timeout)

    return core


def load_standalone(sim,
                    channels: dict[str, tuple[str, str, str]],
                    extra_devices: dict | None = None) -> UniMMCore:
    """Set up a standalone (no .cfg) backend with manual device wiring.

    Handles the common boilerplate: global bridge, Camera, Shutter,
    optional extra devices, and Channel config group.

    Args:
        sim: The simulation instance.
        channels: Mapping of config_name → (device, property, value).
            Example: ``{"brightfield": ("Channel", "Label", "brightfield")}``
        extra_devices: Optional mapping of device_name → device_instance.
            Common devices: ``GenericStateDevice``, ``ObjectiveDevice``.

    Returns:
        Configured UniMMCore instance.
    """
    from virtual_microscope.devices.camera import SimCameraDevice
    from virtual_microscope.devices.shutter import SimShutterDevice

    bridge = SimulationBridge(sim)
    set_global_bridge(bridge)

    core = UniMMCore()
    core.unloadAllDevices()

    core.loadPyDevice("Camera", SimCameraDevice())
    core.loadPyDevice("Shutter", SimShutterDevice())

    if extra_devices:
        for name, device in extra_devices.items():
            core.loadPyDevice(name, device)

    # Initialize all loaded devices
    devices = ["Camera", "Shutter"]
    if extra_devices:
        devices.extend(extra_devices)
    for dev in devices:
        core.initializeDevice(dev)

    core.setCameraDevice("Camera")
    core.setShutterDevice("Shutter")

    # Set initial state for state devices
    if extra_devices:
        for name in extra_devices:
            try:
                core.setState(name, 0)
            except Exception:
                pass

    # Build Channel config group
    if channels:
        core.defineConfigGroup("Channel")
        for config_name, (dev, prop, val) in channels.items():
            core.defineConfig("Channel", config_name, dev, prop, val)
        first = next(iter(channels))
        core.setConfig("Channel", first)

    return core
