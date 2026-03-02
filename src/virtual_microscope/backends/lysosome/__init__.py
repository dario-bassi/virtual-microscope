"""lysosome backend for virtual-microscope (LysosomeSim + lysosomes)."""

BACKEND_INFO = {
    "description": "Simulates lysosomal trafficking with LysoTracker-stained puncta undergoing Brownian motion. Features configurable lysosome count and diffusion rate with temperature and perfusion.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion"],
    "specimen": "Epithelial cells with LysoTracker-stained lysosomes",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "Watch LysoTracker-stained lysosomes undergoing Brownian diffusion inside "
        "epithelial cells. Track individual puncta over time for single-particle "
        "analysis. Temperature modulates diffusion rate; perfusion delivers "
        "agents that alter lysosomal trafficking (e.g. chloroquine, bafilomycin)."
    ),
    "device_effects": {
        "Temperature": "Higher temperature increases lysosomal diffusion rate",
        "Perfusion": "Delivers agents that alter lysosomal pH or trafficking (chloroquine, bafilomycin)",
    },
    "key_parameters": {
        "n_cells": "Number of cells (default 20)",
        "n_lyso_min": "Minimum lysosomes per cell (default 5)",
        "n_lyso_max": "Maximum lysosomes per cell (default 20)",
        "diffusion_rate": "Brownian diffusion coefficient (default 0.3)",
    },
}

from pathlib import Path

from virtual_microscope.backends.lysosome.sim import LysosomeSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=20, seed=42, n_lyso_min=5, n_lyso_max=20, diffusion_rate=0.3, width=512, height=512, internal_scale=4, **kwargs) -> LysosomeSim:
    """Create a LysosomeSim for lysosome imaging."""
    return LysosomeSim(
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


def setup_lysosome(n_cells=20, seed=42, n_lyso_min=5, n_lyso_max=20, diffusion_rate=0.3, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, n_lyso_min=n_lyso_min, n_lyso_max=n_lyso_max, diffusion_rate=diffusion_rate)
    core = load_cfg(sim, Path(__file__).parent / "lysosome.cfg")
    sim.enable_lysosomes(
        core=core,
        n_min=n_lyso_min,
        n_max=n_lyso_max,
        radius_min=1.5,
        radius_max=3.0,
        intensity_mean=215.0,
        intensity_std=20.0,
        diffusion_rate=diffusion_rate,
    )
    return core, sim
