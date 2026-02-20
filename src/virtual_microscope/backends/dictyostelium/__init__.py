"""dictyostelium backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.dictyostelium.sim import DictyosteliumSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=100, n_pacemakers=3, seed=42, world_size=512, relay_radius=40.0, pulse_period=10, camp_reporter_gain=1.0) -> DictyosteliumSim:
    """Create a Dictyostelium simulation."""
    return DictyosteliumSim(
        n_cells=n_cells,
        n_pacemakers=n_pacemakers,
        seed=seed,
        world_size=world_size,
        relay_radius=relay_radius,
        pulse_period=pulse_period,
        camp_reporter_gain=camp_reporter_gain,
    )


def setup_dictyostelium_microscope(n_cells=100, n_pacemakers=3, seed=42, world_size=512, relay_radius=40.0, pulse_period=10, camp_reporter_gain=1.0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, n_pacemakers=n_pacemakers, seed=seed, world_size=world_size, relay_radius=relay_radius, pulse_period=pulse_period, camp_reporter_gain=camp_reporter_gain)
    core = load_cfg(sim, Path(__file__).parent / "dictyostelium.cfg")
    return core, sim
