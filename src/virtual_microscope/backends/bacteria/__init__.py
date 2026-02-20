"""bacteria backend for virtual-microscope."""

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


def setup_bacteria_microscope(n_cells=30, seed=42, world_size=512, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, world_size=world_size, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "bacteria.cfg")
    return core, sim
