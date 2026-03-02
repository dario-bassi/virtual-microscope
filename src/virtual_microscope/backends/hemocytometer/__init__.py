"""hemocytometer backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a hemocytometer counting chamber with trypan-blue viability staining. Supports cell clumps and configurable dilution factor.",
    "channels": ["brightfield", "trypan-blue"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Cell suspension in Neubauer counting chamber",
    "modality": "Brightfield with trypan blue",
    "experiment_guide": (
        "A hemocytometer counting chamber loaded with a cell suspension and "
        "trypan-blue viability dye. Live cells exclude the dye and appear bright; "
        "dead cells stain blue. Count cells in the grid squares to estimate "
        "concentration. Useful for training cell-counting and "
        "viability-estimation algorithms."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_cells": "Total cells in the chamber (default 150)",
        "viability": "Fraction of live cells (default 0.85)",
        "dilution_factor": "Dilution factor for concentration calculation (default 2)",
    },
}

from virtual_microscope.backends.hemocytometer.sim import HemocytometerSim


def create_sim(n_cells=150, viability=0.85, cell_radius_range=(3, 7), dilution_factor=2, clump_fraction=0.0, seed=42) -> HemocytometerSim:
    """Create a hemocytometer simulation."""
    return HemocytometerSim(
        n_cells=n_cells,
        viability=viability,
        cell_radius_range=cell_radius_range,
        dilution_factor=dilution_factor,
        clump_fraction=clump_fraction,
        seed=seed,
    )


def setup_hemocytometer(n_cells=150, viability=0.85, cell_radius_range=(3, 7), dilution_factor=2, clump_fraction=0.0, seed=42):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    from pathlib import Path
    from virtual_microscope._init_standard import load_cfg

    sim = create_sim(n_cells=n_cells, viability=viability, cell_radius_range=cell_radius_range, dilution_factor=dilution_factor, clump_fraction=clump_fraction, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "hemocytometer.cfg")
    return core, sim
