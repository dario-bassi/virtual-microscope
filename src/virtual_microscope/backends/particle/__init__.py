"""particle backend for virtual-microscope (MicroscopeSimOptmized scattered cells)."""

from virtual_microscope.core.microscope_sim_optimized import MicroscopeSimOptmized
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(nb_cells=50, world_width=1500, world_height=1500, base_radius=20.0, rng_seed=0, **kwargs) -> MicroscopeSimOptmized:
    """Create a particle (scattered cells) simulation."""
    return MicroscopeSimOptmized(
        width=world_width,
        height=world_height,
        nb_cells=nb_cells,
        base_radius=base_radius,
        rng_seed=rng_seed,
    )


def setup_particle_microscope(nb_cells=50, rng_seed=0, **kwargs):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(nb_cells=nb_cells, rng_seed=rng_seed, **kwargs)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
