"""voronoi backend for virtual-microscope (static VoronoiSim)."""

BACKEND_INFO = {
    "description": "Simulates a static Voronoi tessellation of an epithelial tissue monolayer. Provides phase-contrast, DAPI nuclear, and membrane channels.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Epithelial tissue monolayer (static)",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "A static Voronoi tessellation representing an epithelial monolayer. "
        "Navigate the tissue with XY stage and zoom with objectives. "
        "Phase-contrast shows cell boundaries, DAPI labels nuclei, and membrane "
        "channel highlights cell–cell junctions. Useful as a simple baseline for "
        "segmentation benchmarks."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_cells": "Number of cells in the tessellation (default 60)",
        "jitter": "Voronoi jitter (randomness) 0–1 (default 0.7)",
        "nucleus_fraction": "Nucleus-to-cell area ratio (default 0.3)",
    },
}

from pathlib import Path

from virtual_microscope.sims.voronoi.voronoi import VoronoiSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=60, width=512, height=512, seed=42, jitter=0.7, nucleus_fraction=0.3, internal_scale=4) -> VoronoiSim:
    """Create a static Voronoi tissue simulation."""
    return VoronoiSim(
        n_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        jitter=jitter,
        nucleus_fraction=nucleus_fraction,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_voronoi(n_cells=60, width=512, height=512, seed=42, jitter=0.7, nucleus_fraction=0.3, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, width=width, height=height, seed=seed, jitter=jitter, nucleus_fraction=nucleus_fraction, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "voronoi.cfg")
    return core, sim
