"""histology backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.histology.sim import HistologySim
from virtual_microscope._init_standard import load_cfg


def create_sim(tissue_type="glandular", world_size=512, n_nuclei=200, seed=42, internal_scale=4, grade=0) -> HistologySim:
    """Create a histology simulation."""
    return HistologySim(
        tissue_type=tissue_type,
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_nuclei=n_nuclei,
        seed=seed,
        internal_scale=internal_scale,
        grade=grade,
    )


def setup_histology_microscope(tissue_type="glandular", world_size=512, n_nuclei=200, seed=42, internal_scale=4, grade=0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(tissue_type=tissue_type, world_size=world_size, n_nuclei=n_nuclei, seed=seed, internal_scale=internal_scale, grade=grade)
    core = load_cfg(sim, Path(__file__).parent / "histology.cfg")
    return core, sim
