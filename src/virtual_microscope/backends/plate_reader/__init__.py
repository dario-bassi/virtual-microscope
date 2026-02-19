"""plate_reader backend for virtual-microscope."""

from virtual_microscope.backends.plate_reader.sim import PlateReaderSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(assay_type="viability", seed=42) -> PlateReaderSim:
    """Create a plate reader simulation."""
    return PlateReaderSim(assay_type=assay_type, seed=seed)


def setup_plate_reader(assay_type="viability", seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope.devices.state import GenericStateDevice
    sim = create_sim(assay_type=assay_type, seed=seed)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    core.unloadAllDevices()
    from virtual_microscope.devices.camera import SimCameraDevice
    core.loadPyDevice("Camera", SimCameraDevice())
    core.initializeDevice("Camera")
    core.setCameraDevice("Camera")
    # Channel based on assay type
    from virtual_microscope.backends.plate_reader.sim import ASSAY_PRESETS
    preset = ASSAY_PRESETS.get(assay_type, ASSAY_PRESETS["viability"])
    wl_pri = preset["wavelength_primary"]
    wl_ref = preset.get("wavelength_reference")
    labels = {0: f"{wl_pri}nm" if wl_pri else "luminescence"}
    if wl_ref:
        labels[1] = f"{wl_ref}nm-ref"
    core.loadPyDevice("Channel", GenericStateDevice("Channel", labels))
    core.initializeDevice("Channel")
    core.setState("Channel", 0)
    core.defineConfigGroup("Fake")
    for idx, lbl in labels.items():
        core.defineConfig("Fake", lbl, "Channel", "Label", lbl)
    core.setConfig("Fake", labels[0])
    return core, sim
