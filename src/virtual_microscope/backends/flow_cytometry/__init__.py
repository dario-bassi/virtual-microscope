"""flow_cytometry backend for virtual-microscope."""

from virtual_microscope.backends.flow_cytometry.sim import FlowCytometrySim
from virtual_microscope.engine.simulation_bridge import SimulationBridge
import virtual_microscope.engine.simulation_bridge as bridge_module
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_total=10000, events_per_snap=100, seed=42) -> FlowCytometrySim:
    """Create a flow cytometry simulation."""
    return FlowCytometrySim(n_total=n_total, events_per_snap=events_per_snap, seed=seed)


def setup_flow_cytometry(n_total=10000, events_per_snap=100, seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope.devices.state import GenericStateDevice, ObjectiveDevice
    sim = create_sim(n_total=n_total, events_per_snap=events_per_snap, seed=seed)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    core.unloadAllDevices()
    from virtual_microscope.devices.camera import SimCameraDevice
    from virtual_microscope.devices.shutter import SimShutterDevice
    core.loadPyDevice("Camera", SimCameraDevice())
    core.loadPyDevice("Shutter", SimShutterDevice())
    core.loadPyDevice("Objective", ObjectiveDevice())
    core.loadPyDevice("Detector", GenericStateDevice("Detector", {0: "FSC-SSC", 1: "FL1-FITC", 2: "FL2-PE"}))
    for dev in ("Camera", "Objective", "Detector", "Shutter"):
        core.initializeDevice(dev)
    core.setCameraDevice("Camera")
    core.setShutterDevice("Shutter")
    core.setState("Objective", 0)
    core.setState("Detector", 0)
    core.defineConfigGroup("Channel")
    core.defineConfig("Channel", "scatter", "Detector", "Label", "FSC-SSC")
    core.defineConfig("Channel", "FITC", "Detector", "Label", "FL1-FITC")
    core.defineConfig("Channel", "PE", "Detector", "Label", "FL2-PE")
    core.setConfig("Channel", "scatter")
    return core, sim
