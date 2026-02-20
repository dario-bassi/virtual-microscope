"""dictyostelium backend for virtual-microscope."""

from virtual_microscope.backends.dictyostelium.sim import DictyosteliumSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=100, n_pacemakers=3, seed=42, world_size=512, relay_radius=40.0, pulse_period=10, camp_reporter_gain=1.0) -> DictyosteliumSim:
    """Create a Dictyostelium simulation."""
    return DictyosteliumSim(
        n_cells=n_cells,
        n_pacemakers=n_pacemakers,
        seed=seed,
        world_size=world_size,
        relay_radius=relay_radius,
        pulse_period=pulse_period,
        camp_reporter_gain=camp_reporter_gain,
    )


def setup_dictyostelium_microscope(n_cells=100, n_pacemakers=3, seed=42, world_size=512, relay_radius=40.0, pulse_period=10, camp_reporter_gain=1.0):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, n_pacemakers=n_pacemakers, seed=seed, world_size=world_size, relay_radius=relay_radius, pulse_period=pulse_period, camp_reporter_gain=camp_reporter_gain)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
