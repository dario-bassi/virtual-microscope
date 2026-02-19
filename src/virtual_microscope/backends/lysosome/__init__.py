"""lysosome backend for virtual-microscope (DynamicVoronoiSim + lysosomes)."""

from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=20, seed=42, n_lyso_min=5, n_lyso_max=20, diffusion_rate=0.3, width=512, height=512, internal_scale=4, **kwargs) -> DynamicVoronoiSim:
    """Create a DynamicVoronoiSim for lysosome imaging."""
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


def setup_lysosome_microscope(n_cells=20, seed=42, n_lyso_min=5, n_lyso_max=20, diffusion_rate=0.3, **kwargs):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, seed=seed, n_lyso_min=n_lyso_min, n_lyso_max=n_lyso_max, diffusion_rate=diffusion_rate)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    sim.enable_lysosomes(
        core=core,
        n_min=n_lyso_min,
        n_max=n_lyso_max,
        radius_min=1.5,
        radius_max=3.0,
        intensity_mean=215.0,
        intensity_std=20.0,
        diffusion_rate=diffusion_rate,
    )
    return core, sim
