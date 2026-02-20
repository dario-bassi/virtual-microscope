"""voronoi backend for virtual-microscope (static VoronoiSim)."""

from pathlib import Path

from virtual_microscope.sims.voronoi.voronoi import VoronoiSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=60, width=512, height=512, seed=42, jitter=0.7, nucleus_fraction=0.3, internal_scale=4) -> VoronoiSim:
    """Create a static Voronoi tissue simulation."""
    return VoronoiSim(
        nb_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        rng_seed=seed,
        jitter=jitter,
        nucleus_fraction=nucleus_fraction,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_voronoi_microscope(n_cells=60, width=512, height=512, seed=42, jitter=0.7, nucleus_fraction=0.3, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, width=width, height=height, seed=seed, jitter=jitter, nucleus_fraction=nucleus_fraction, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "voronoi.cfg")
    return core, sim
