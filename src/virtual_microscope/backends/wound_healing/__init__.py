"""wound_healing backend for virtual-microscope (DynamicVoronoiSim with wound)."""

from pathlib import Path

from virtual_microscope.sims.voronoi.tissue_dynamics import DynamicVoronoiSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=100, seed=42, wound_width=120, migration_speed=2.0, width=512, height=512, internal_scale=4) -> DynamicVoronoiSim:
    """Create a DynamicVoronoiSim for wound healing assay."""
    sim = DynamicVoronoiSim(
        n_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        jitter=0.7,
        nucleus_fraction=0.3,
        internal_scale=internal_scale,
    )
    sim.migration_speed = migration_speed
    return sim


def setup_wound_healing(n_cells=100, seed=42, wound_width=120, migration_speed=2.0, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, wound_width=wound_width, migration_speed=migration_speed)
    core = load_cfg(sim, Path(__file__).parent / "wound_healing.cfg")
    # Pre-equilibrate tissue
    for _ in range(5):
        sim.step(dt=1.0)
    # Create vertical scratch wound
    sim.create_wound(
        shape="rectangle",
        center=(512 // 2, 512 // 2),
        size=(wound_width, 512),
        jagged=0.2,
    )
    sim._bf_full = None
    sim._nuc_full = None
    sim._mem_full = None
    return core, sim
