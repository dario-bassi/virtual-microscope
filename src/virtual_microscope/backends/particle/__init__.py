"""particle backend for virtual-microscope (ScatteredCellSim)."""

BACKEND_INFO = {
    "description": "Simulates scattered vertex-based cells as generic particles for basic microscopy. Provides phase-contrast, DAPI, and membrane channels.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": True,
    "extra_devices": [],
}

from pathlib import Path

from virtual_microscope.sims.cell.sim import ScatteredCellSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=50, world_size=1500, base_radius=20.0, seed=0, **kwargs) -> ScatteredCellSim:
    """Create a particle (scattered cells) simulation."""
    return ScatteredCellSim(
        width=world_size,
        height=world_size,
        n_cells=n_cells,
        base_radius=base_radius,
        seed=seed,
    )


def setup_particle(n_cells=50, seed=0, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, **kwargs)
    core = load_cfg(sim, Path(__file__).parent / "particle.cfg")
    return core, sim
