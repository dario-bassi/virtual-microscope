"""particle backend for virtual-microscope (ScatteredCellSim)."""

BACKEND_INFO = {
    "description": "Simulates scattered vertex-based cells as generic particles for basic microscopy. Provides phase-contrast, DAPI, and membrane channels.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": True,
    "extra_devices": [],
    "specimen": "Generic scattered cells (vertex model)",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "A generic cell simulation using the vertex-based scattered-cell model. "
        "Cells undergo cell-cycle progression with division and apoptosis. "
        "Provides phase-contrast, DAPI, and membrane channels as a baseline for "
        "testing image-analysis pipelines on simple cell populations."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_cells": "Initial cell count (default 50)",
        "world_size": "World size in pixels (default 1500)",
        "base_radius": "Base cell radius in pixels (default 20)",
    },
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
