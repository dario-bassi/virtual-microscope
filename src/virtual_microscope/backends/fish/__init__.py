"""fish backend for virtual-microscope (DynamicVoronoiSim + FISH probes)."""

from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_cells=40, seed=42, locus_copies=2, amplified_fraction=0.15, deleted_fraction=0.10, amplified_copies_range=(3, 6), width=512, height=512, internal_scale=4) -> DynamicVoronoiSim:
    """Create a DynamicVoronoiSim for FISH probe imaging."""
    return DynamicVoronoiSim(
        nb_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        rng_seed=seed,
        jitter=0.6,
        nucleus_fraction=0.32,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_fish_microscope(n_cells=40, seed=42, locus_copies=2, amplified_fraction=0.15, deleted_fraction=0.10, amplified_copies_range=(3, 6), **kwargs):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, seed=seed, locus_copies=locus_copies, amplified_fraction=amplified_fraction, deleted_fraction=deleted_fraction, amplified_copies_range=amplified_copies_range)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    sim.enable_fish_probes(
        core=core,
        locus_copies=locus_copies,
        amplified_fraction=amplified_fraction,
        deleted_fraction=deleted_fraction,
        amplified_copies_range=amplified_copies_range,
        probe_intensity=215.0,
        probe_intensity_std=18.0,
        fwhm_world_px=1.8,
    )
    return core, sim
