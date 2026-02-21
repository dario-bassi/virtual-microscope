"""particle backend for virtual-microscope (ScatteredCellSim)."""

from pathlib import Path

from virtual_microscope.sims.cell.sim import ScatteredCellSim
from virtual_microscope._init_standard import load_cfg


def create_sim(nb_cells=50, world_width=1500, world_height=1500, base_radius=20.0, rng_seed=0, **kwargs) -> ScatteredCellSim:
    """Create a particle (scattered cells) simulation."""
    return ScatteredCellSim(
        width=world_width,
        height=world_height,
        nb_cells=nb_cells,
        base_radius=base_radius,
        rng_seed=rng_seed,
    )


def setup_particle_microscope(nb_cells=50, rng_seed=0, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(nb_cells=nb_cells, rng_seed=rng_seed, **kwargs)
    core = load_cfg(sim, Path(__file__).parent / "particle.cfg")
    return core, sim
