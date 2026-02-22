"""fibroblast backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.fibroblast.sim import FibroblastSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=512, n_cells=8, seed=42, internal_scale=4) -> FibroblastSim:
    """Create a fibroblast simulation."""
    return FibroblastSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_fibroblast(world_size=512, n_cells=8, seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, n_cells=n_cells, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "fibroblast.cfg")
    return core, sim
