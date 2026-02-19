"""hemocytometer backend for virtual-microscope."""

from virtual_microscope.backends.hemocytometer.sim import HemocytometerSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=150, viability=0.85, cell_radius_range=(3, 7), dilution_factor=2, clump_fraction=0.0, seed=42) -> HemocytometerSim:
    """Create a hemocytometer simulation."""
    return HemocytometerSim(
        n_cells=n_cells,
        viability=viability,
        cell_radius_range=cell_radius_range,
        dilution_factor=dilution_factor,
        clump_fraction=clump_fraction,
        seed=seed,
    )


def setup_hemocytometer(n_cells=150, viability=0.85, cell_radius_range=(3, 7), dilution_factor=2, clump_fraction=0.0, seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope.devices.state import GenericStateDevice
    sim = create_sim(n_cells=n_cells, viability=viability, cell_radius_range=cell_radius_range, dilution_factor=dilution_factor, clump_fraction=clump_fraction, seed=seed)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    core.unloadAllDevices()
    from virtual_microscope.devices.camera import SimCameraDevice
    core.loadPyDevice("Camera", SimCameraDevice())
    labels = {0: "brightfield", 1: "trypan-blue"}
    core.loadPyDevice("Channel", GenericStateDevice("Channel", labels))
    for dev in ("Camera", "Channel"):
        core.initializeDevice(dev)
    core.setCameraDevice("Camera")
    core.setState("Channel", 0)
    core.defineConfigGroup("Fake")
    core.defineConfig("Fake", "brightfield", "Channel", "Label", "brightfield")
    core.defineConfig("Fake", "trypan-blue", "Channel", "Label", "trypan-blue")
    core.setConfig("Fake", "brightfield")
    return core, sim
