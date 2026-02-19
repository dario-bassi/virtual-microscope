"""yeast backend for virtual-microscope."""

from virtual_microscope.backends.yeast.sim import YeastSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(world_size=512, n_cells=200, seed=42, internal_scale=4, division_time=15.0) -> YeastSim:
    """Create a yeast simulation."""
    return YeastSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
        division_time=division_time,
    )


def setup_yeast_microscope(world_size=512, n_cells=200, seed=42, internal_scale=4, division_time=15.0):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, n_cells=n_cells, seed=seed, internal_scale=internal_scale, division_time=division_time)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
