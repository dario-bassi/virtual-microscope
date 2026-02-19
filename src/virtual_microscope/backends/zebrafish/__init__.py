"""zebrafish backend for virtual-microscope."""

from virtual_microscope.backends.zebrafish.sim import ZebrafishSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_rbc=30, cardiac_freq=2.5, seed=42, internal_scale=2) -> ZebrafishSim:
    """Create a zebrafish simulation."""
    return ZebrafishSim(n_rbc=n_rbc, cardiac_freq=cardiac_freq, seed=seed, internal_scale=internal_scale)


def setup_zebrafish_microscope(n_rbc=30, cardiac_freq=2.5, seed=42, internal_scale=2):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope.devices.state import AnesthesiaDevice
    sim = create_sim(n_rbc=n_rbc, cardiac_freq=cardiac_freq, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    # Override channels for zebrafish transgenic lines
    core.defineConfig("Fake", "nucleus-channel", "LED", "Label", "GREEN")
    core.defineConfig("Fake", "nucleus-channel", "Filter Wheel", "Label", "TagGFP2(483/506)")
    core.defineConfig("Fake", "membrane-channel", "LED", "Label", "ORANGE")
    core.defineConfig("Fake", "membrane-channel", "Filter Wheel", "Label", "mScarlet3(569/582)")
    core.setState("Temperature", 9)  # 28°C optimal
    core.loadPyDevice("Anesthesia", AnesthesiaDevice())
    core.initializeDevice("Anesthesia")
    core.setState("Anesthesia", 0)
    # Zebrafish pixel size: 5.0/2.5/1.25/0.625 µm/px
    _zf_px = {0: 5.0, 1: 2.5, 2: 1.25, 3: 0.625}
    core.getPixelSizeUm = lambda cached=False: _zf_px.get(core.getState("Objective"), 1.0)
    return core, sim
