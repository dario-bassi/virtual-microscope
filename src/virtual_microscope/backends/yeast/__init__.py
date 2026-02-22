"""yeast backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates budding yeast (S. cerevisiae) with cell-wall Calcofluor-White staining and GFP reporter. Supports temperature control and configurable division time.",
    "channels": ["phase-contrast", "Calcofluor-White", "GFP-reporter"],
    "continuous": False,
    "extra_devices": ["Temperature"],
}

from pathlib import Path

from virtual_microscope.backends.yeast.sim import YeastSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=512, n_cells=200, seed=42, internal_scale=4, division_time=15.0) -> YeastSim:
    """Create a yeast simulation."""
    return YeastSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
        division_time=division_time,
    )


def setup_yeast(world_size=512, n_cells=200, seed=42, internal_scale=4, division_time=15.0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, n_cells=n_cells, seed=seed, internal_scale=internal_scale, division_time=division_time)
    core = load_cfg(sim, Path(__file__).parent / "yeast.cfg")
    return core, sim
