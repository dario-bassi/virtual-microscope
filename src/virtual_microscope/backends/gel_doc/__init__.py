"""gel_doc backend for virtual-microscope."""

from virtual_microscope.backends.gel_doc.sim import GelDocSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_lanes=8, gel_type="western", seed=42) -> GelDocSim:
    """Create a gel doc simulation."""
    return GelDocSim(n_lanes=n_lanes, gel_type=gel_type, seed=seed)


def setup_gel_doc(n_lanes=8, gel_type="western", seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope.devices.state import ObjectiveDevice
    sim = create_sim(n_lanes=n_lanes, gel_type=gel_type, seed=seed)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    core.unloadAllDevices()
    from virtual_microscope.devices.camera import SimCameraDevice
    core.loadPyDevice("Camera", SimCameraDevice())
    core.loadPyDevice("Objective", ObjectiveDevice())
    for dev in ("Camera", "Objective"):
        core.initializeDevice(dev)
    core.setCameraDevice("Camera")
    core.setState("Objective", 0)
    core.defineConfigGroup("Channel")
    core.defineConfig("Channel", "gel-image", "Objective", "Label", "10x")
    core.setConfig("Channel", "gel-image")
    return core, sim
