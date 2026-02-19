"""bacteria backend for virtual-microscope."""

from virtual_microscope.backends.bacteria.sim import BacteriaSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=30, seed=42, world_size=512, internal_scale=4) -> BacteriaSim:
    """Create a bacteria simulation."""
    return BacteriaSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_bacteria_microscope(n_cells=30, seed=42, world_size=512, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, seed=seed, world_size=world_size, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
