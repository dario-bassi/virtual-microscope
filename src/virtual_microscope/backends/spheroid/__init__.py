"""spheroid backend for virtual-microscope."""

from virtual_microscope.backends.spheroid.sim import SpheroidSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(radius=80, n_cells=2000, necrotic_fraction=0.45, quiescent_fraction=0.20, seed=42, internal_scale=4) -> SpheroidSim:
    """Create a spheroid simulation."""
    return SpheroidSim(
        radius=radius,
        n_cells=n_cells,
        necrotic_fraction=necrotic_fraction,
        quiescent_fraction=quiescent_fraction,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_spheroid_microscope(radius=80, n_cells=2000, necrotic_fraction=0.45, quiescent_fraction=0.20, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(radius=radius, n_cells=n_cells, necrotic_fraction=necrotic_fraction, quiescent_fraction=quiescent_fraction, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
