"""viability backend for virtual-microscope (DynamicVoronoiSim + viability staining)."""

from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=60, seed=42, live_fraction=0.85, width=512, height=512, internal_scale=4) -> DynamicVoronoiSim:
    """Create a DynamicVoronoiSim for viability staining."""
    return DynamicVoronoiSim(
        nb_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        rng_seed=seed,
        jitter=0.7,
        nucleus_fraction=0.3,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_viability_microscope(n_cells=60, seed=42, live_fraction=0.85, **kwargs):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, seed=seed, live_fraction=live_fraction)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    sim.enable_viability_staining(
        core=core,
        live_fraction=live_fraction,
        calcein_intensity=190.0,
        pi_intensity=220.0,
        calcein_dead_fraction=0.05,
        rng_seed=seed + 1000,
    )
    return core, sim
