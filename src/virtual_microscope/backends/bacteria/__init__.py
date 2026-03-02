"""bacteria backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates rod-shaped bacteria with growth and division dynamics. Supports SLM-based optogenetic stimulation and temperature control.",
    "channels": ["phase-contrast", "GFP", "DAPI"],
    "continuous": True,
    "extra_devices": ["SLM", "Temperature"],
    "specimen": "E. coli (rod-shaped bacteria)",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "Seed a population of bacteria and watch them grow and divide. Use "
        "phase-contrast for morphology and GFP/DAPI for fluorescence markers. "
        "Apply SLM masks to trigger optogenetic stimulation in specific regions. "
        "Raise temperature to accelerate growth or cool to 4°C to arrest "
        "division."
    ),
    "device_effects": {
        "SLM": "Activates bPAC optogenetic tool in illuminated cells, increasing cAMP",
        "Temperature": "Growth rate scales with temperature; 4°C arrests division, 42°C is heat shock",
    },
    "key_parameters": {
        "n_cells": "Initial cell count (default 30)",
        "world_size": "World size in pixels (default 512)",
    },
}

from pathlib import Path

from virtual_microscope.backends.bacteria.sim import BacteriaSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=30, seed=42, world_size=512, internal_scale=4) -> BacteriaSim:
    """Create a bacteria simulation."""
    return BacteriaSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_bacteria(n_cells=30, seed=42, world_size=512, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, world_size=world_size, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "bacteria.cfg")
    return core, sim
