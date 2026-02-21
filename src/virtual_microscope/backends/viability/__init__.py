"""viability backend for virtual-microscope (ViabilitySim + viability staining)."""

from pathlib import Path

from virtual_microscope.backends.viability.sim import ViabilitySim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=60, seed=42, live_fraction=0.85, width=512, height=512, internal_scale=4) -> ViabilitySim:
    """Create a ViabilitySim for viability staining."""
    return ViabilitySim(
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
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, live_fraction=live_fraction)
    core = load_cfg(sim, Path(__file__).parent / "viability.cfg")
    sim.enable_viability_staining(
        core=core,
        live_fraction=live_fraction,
        calcein_intensity=190.0,
        pi_intensity=220.0,
        calcein_dead_fraction=0.05,
        rng_seed=seed + 1000,
    )
    return core, sim
