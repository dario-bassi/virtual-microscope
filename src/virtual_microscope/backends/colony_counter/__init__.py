"""colony_counter backend for virtual-microscope."""

from virtual_microscope.backends.colony_counter.sim import ColonySim
from virtual_microscope.engine.simulation_bridge import SimulationBridge
import virtual_microscope.engine.simulation_bridge as bridge_module
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_colonies=200, plate_type="spread", seed=42, distribution="random", staining=None, blue_fraction=0.3, colony_size_range=(3, 15), satellite_fraction=0.0, dynamic=False, growth_rate=0.1, max_colony_radius=20.0) -> ColonySim:
    """Create a colony counter simulation."""
    return ColonySim(
        n_colonies=n_colonies,
        plate_type=plate_type,
        distribution=distribution,
        staining=staining,
        blue_fraction=blue_fraction,
        zone_discs=None,
        colony_size_range=colony_size_range,
        satellite_fraction=satellite_fraction,
        dynamic=dynamic,
        growth_rate=growth_rate,
        max_colony_radius=max_colony_radius,
        seed=seed,
    )


def setup_colony_counter(n_colonies=200, plate_type="spread", seed=42, **kwargs):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope.devices.state import GenericStateDevice
    sim = create_sim(n_colonies=n_colonies, plate_type=plate_type, seed=seed, **kwargs)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    core.unloadAllDevices()
    from virtual_microscope.devices.camera import SimCameraDevice
    from virtual_microscope.devices.shutter import SimShutterDevice
    core.loadPyDevice("Camera", SimCameraDevice())
    core.loadPyDevice("Shutter", SimShutterDevice())
    labels = {0: "transmitted", 1: "blue-filter", 2: "GFP-excitation"}
    core.loadPyDevice("Channel", GenericStateDevice("Channel", labels))
    for dev in ("Camera", "Channel", "Shutter"):
        core.initializeDevice(dev)
    core.setCameraDevice("Camera")
    core.setShutterDevice("Shutter")
    core.setState("Channel", 0)
    core.defineConfigGroup("Channel")
    core.defineConfig("Channel", "plate-image", "Channel", "Label", "transmitted")
    core.defineConfig("Channel", "blue-channel", "Channel", "Label", "blue-filter")
    core.defineConfig("Channel", "gfp-channel", "Channel", "Label", "GFP-excitation")
    core.setConfig("Channel", "plate-image")
    return core, sim
