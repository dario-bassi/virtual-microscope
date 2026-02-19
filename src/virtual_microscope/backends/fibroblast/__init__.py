"""fibroblast backend for virtual-microscope."""

from virtual_microscope.backends.fibroblast.sim import FibroblastSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(world_size=512, n_cells=8, seed=42, internal_scale=4) -> FibroblastSim:
    """Create a fibroblast simulation."""
    return FibroblastSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_fibroblast_microscope(world_size=512, n_cells=8, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, n_cells=n_cells, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
