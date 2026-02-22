"""spheroid backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a multicellular tumour spheroid cross-section with necrotic core and quiescent rim. Features Calcein-AM (live) and propidium-iodide (dead) viability staining.",
    "channels": ["brightfield", "Calcein-AM", "propidium-iodide"],
    "continuous": True,
    "extra_devices": [],
}

from pathlib import Path

from virtual_microscope.backends.spheroid.sim import SpheroidSim
from virtual_microscope._init_standard import load_cfg


def create_sim(radius=80, n_cells=2000, necrotic_fraction=0.45, quiescent_fraction=0.20, seed=42, internal_scale=4) -> SpheroidSim:
    """Create a spheroid simulation."""
    return SpheroidSim(
        radius=radius,
        n_cells=n_cells,
        necrotic_fraction=necrotic_fraction,
        quiescent_fraction=quiescent_fraction,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_spheroid(radius=80, n_cells=2000, necrotic_fraction=0.45, quiescent_fraction=0.20, seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(radius=radius, n_cells=n_cells, necrotic_fraction=necrotic_fraction, quiescent_fraction=quiescent_fraction, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "spheroid.cfg")
    return core, sim
