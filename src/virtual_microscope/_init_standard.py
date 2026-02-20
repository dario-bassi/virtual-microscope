"""Standard device stack initializer — shared by all programmatic setup_* functions.

The primary entry point is ``load_cfg(sim, cfg_path)`` which:
  1. Wires the simulation into the global bridge
  2. Loads the backend .cfg (devices, channels, everything)
  3. Installs pixel-size helper

Usage:
    from virtual_microscope._init_standard import load_cfg
    core = load_cfg(sim, Path(__file__).parent / "bacteria.cfg")
"""

from pathlib import Path

from pymmcore_plus.experimental.unicore import UniMMCore

import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope.simulation_bridge import SimulationBridge


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


def load_cfg(sim, cfg_path: Path) -> UniMMCore:
    """Load a backend .cfg after pre-creating the simulation with custom params.

    This is the single entry point for programmatic setup_*_microscope() calls.
    The .cfg is the sole source of truth for devices and channel definitions.

    Args:
        sim: Simulation backend instance (already created with custom params).
        cfg_path: Path to the backend's .cfg file.

    Returns:
        Configured UniMMCore instance.
    """
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    bridge_module.bridge_ready.set()
    core = UniMMCore()
    core.loadSystemConfiguration(str(cfg_path))
    _install_pixel_size(core, sim)
    return core
