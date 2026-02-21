"""stress_granule backend for virtual-microscope (StressGranuleSim + stress granules)."""

from pathlib import Path

from virtual_microscope.backends.stress_granule.sim import StressGranuleSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=30, seed=42, foci_radius_range=(2.0, 4.0), max_foci_per_cell=12, formation_rate=0.25, dissolution_rate=0.12, heterogeneity=0.35, width=512, height=512, internal_scale=4) -> StressGranuleSim:
    """Create a StressGranuleSim for stress granule imaging."""
    return StressGranuleSim(
        nb_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        rng_seed=seed,
        jitter=0.7,
        nucleus_fraction=0.30,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_stress_granule_microscope(n_cells=30, seed=42, foci_radius_range=(2.0, 4.0), max_foci_per_cell=12, formation_rate=0.25, dissolution_rate=0.12, heterogeneity=0.35, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, foci_radius_range=foci_radius_range, max_foci_per_cell=max_foci_per_cell, formation_rate=formation_rate, dissolution_rate=dissolution_rate, heterogeneity=heterogeneity)
    core = load_cfg(sim, Path(__file__).parent / "stress_granule.cfg")
    sim.enable_stress_granules(
        core=core,
        foci_radius_range=foci_radius_range,
        max_foci_per_cell=max_foci_per_cell,
        formation_rate=formation_rate,
        dissolution_rate=dissolution_rate,
        heterogeneity=heterogeneity,
        baseline_foci=0,
    )
    return core, sim
