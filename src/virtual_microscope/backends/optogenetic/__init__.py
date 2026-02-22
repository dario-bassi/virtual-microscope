"""optogenetic backend — vertex-based cells with SLM stimulation."""

BACKEND_INFO = {
    "description": "Simulates vertex-based cells with optogenetic SLM stimulation driving cell motility. Uses ScatteredCellSim with light-activated morphological responses.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": True,
    "extra_devices": ["SLM"],
}

from pathlib import Path

from virtual_microscope.sims.cell.sim import ScatteredCellSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=30, world_size=512, base_radius=20.0, seed=0, **kwargs) -> ScatteredCellSim:
    """Create an optogenetic cell-motility simulation."""
    return ScatteredCellSim(
        width=world_size,
        height=world_size,
        n_cells=n_cells,
        cell_type="optogenetic",
        base_radius=base_radius,
        seed=seed,
    )


def setup_optogenetic(n_cells=30, seed=0, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, **kwargs)
    core = load_cfg(sim, Path(__file__).parent / "optogenetic.cfg")
    return core, sim
