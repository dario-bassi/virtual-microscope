"""viability backend for virtual-microscope (ViabilitySim + viability staining)."""

BACKEND_INFO = {
    "description": "Simulates live/dead viability staining with Calcein-AM and Ethidium homodimer-1. Features configurable live fraction with temperature and perfusion.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion"],
    "specimen": "Epithelial cells with live/dead viability staining",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "Assess cell viability using Calcein-AM (live, green) and Ethidium "
        "homodimer-1 (dead, red) dual staining. A configurable fraction of cells "
        "are dead at baseline. Temperature and perfusion can modulate viability "
        "over time. Useful for training live/dead classification and cytotoxicity "
        "quantification pipelines."
    ),
    "device_effects": {
        "Temperature": "Extreme temperatures (≥45°C) induce cell death over time",
        "Perfusion": "Delivers cytotoxic agents that reduce viability",
    },
    "key_parameters": {
        "n_cells": "Number of cells (default 60)",
        "live_fraction": "Initial fraction of live cells (default 0.85)",
    },
}

from pathlib import Path

from virtual_microscope.backends.viability.sim import ViabilitySim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=60, seed=42, live_fraction=0.85, width=512, height=512, internal_scale=4) -> ViabilitySim:
    """Create a ViabilitySim for viability staining."""
    return ViabilitySim(
        n_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        jitter=0.7,
        nucleus_fraction=0.3,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_viability(n_cells=60, seed=42, live_fraction=0.85, **kwargs):
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
