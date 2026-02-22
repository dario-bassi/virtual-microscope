"""lysosome backend for virtual-microscope (LysosomeSim + lysosomes)."""

BACKEND_INFO = {
    "description": "Simulates lysosomal trafficking with LysoTracker-stained puncta undergoing Brownian motion. Features configurable lysosome count and diffusion rate with temperature and perfusion.",
    "channels": ["phase-contrast", "DAPI", "membrane"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion"],
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
