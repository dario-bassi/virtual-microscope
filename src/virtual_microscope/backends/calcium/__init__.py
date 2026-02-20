"""calcium backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.calcium.sim import CalciumSim
from virtual_microscope._init_standard import load_cfg


def create_sim(grid_size=512, n_sources=-1, n_cells=200, steps_per_snap=1000, seed=42, internal_scale=4, Du=5.0) -> CalciumSim:
    """Create a calcium wave simulation."""
    sim = CalciumSim(
        grid_size=grid_size,
        viewport_width=512,
        viewport_height=512,
        n_sources=n_sources,
        n_cells=n_cells,
        seed=seed,
        internal_scale=internal_scale,
        Du=Du,
    )
    sim.steps_per_snap = steps_per_snap
    return sim


def setup_calcium_microscope(grid_size=512, n_sources=-1, n_cells=200, steps_per_snap=1000, seed=42, internal_scale=4, Du=5.0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(grid_size=grid_size, n_sources=n_sources, n_cells=n_cells, steps_per_snap=steps_per_snap, seed=seed, internal_scale=internal_scale, Du=Du)
    core = load_cfg(sim, Path(__file__).parent / "calcium.cfg")
    return core, sim
