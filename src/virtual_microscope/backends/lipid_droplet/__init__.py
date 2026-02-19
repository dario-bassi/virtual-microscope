"""lipid_droplet backend for virtual-microscope (DynamicVoronoiSim + lipid droplets)."""

from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=25, seed=42, steatotic_fraction=0.4, normal_n_range=(1, 5), steatotic_n_range=(15, 40), radius_range=(3.0, 8.0), width=512, height=512, internal_scale=4) -> DynamicVoronoiSim:
    """Create a DynamicVoronoiSim for lipid droplet imaging."""
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


def setup_lipid_droplet_microscope(n_cells=25, seed=42, steatotic_fraction=0.4, normal_n_range=(1, 5), steatotic_n_range=(15, 40), radius_range=(3.0, 8.0), **kwargs):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, seed=seed, steatotic_fraction=steatotic_fraction, normal_n_range=normal_n_range, steatotic_n_range=steatotic_n_range, radius_range=radius_range)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    sim.enable_lipid_droplets(
        core=core,
        normal_n_range=normal_n_range,
        steatotic_n_range=steatotic_n_range,
        steatotic_fraction=steatotic_fraction,
        radius_range=radius_range,
        intensity_mean=210.0,
        intensity_std=25.0,
    )
    return core, sim
