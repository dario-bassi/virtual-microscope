"""organoid backend for virtual-microscope."""

from virtual_microscope.backends.organoid.sim import OrganoidSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(outer_radius=120, wall_thickness=20, n_cells=400, n_buds=0, seed=42, internal_scale=4, lumen_opacity=0.6) -> OrganoidSim:
    """Create an organoid simulation."""
    return OrganoidSim(
        outer_radius=outer_radius,
        wall_thickness=wall_thickness,
        n_cells=n_cells,
        n_buds=n_buds,
        seed=seed,
        internal_scale=internal_scale,
        lumen_opacity=lumen_opacity,
    )


def setup_organoid_microscope(outer_radius=120, wall_thickness=20, n_cells=400, n_buds=0, seed=42, internal_scale=4, lumen_opacity=0.6):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(outer_radius=outer_radius, wall_thickness=wall_thickness, n_cells=n_cells, n_buds=n_buds, seed=seed, internal_scale=internal_scale, lumen_opacity=lumen_opacity)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
