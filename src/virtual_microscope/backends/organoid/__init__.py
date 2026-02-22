"""organoid backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a 3D organoid cross-section with lumen, wall cells, and optional budding. Features E-cadherin junctional staining and DAPI nuclear label.",
    "channels": ["brightfield", "DAPI", "E-cadherin"],
    "continuous": True,
    "extra_devices": [],
}

from pathlib import Path

from virtual_microscope.backends.organoid.sim import OrganoidSim
from virtual_microscope._init_standard import load_cfg


def create_sim(outer_radius=120, wall_thickness=20, n_cells=400, n_buds=0, seed=42, internal_scale=4, lumen_opacity=0.6) -> OrganoidSim:
    """Create an organoid simulation."""
    return OrganoidSim(
        outer_radius=outer_radius,
        wall_thickness=wall_thickness,
        n_cells=n_cells,
        n_buds=n_buds,
        seed=seed,
        internal_scale=internal_scale,
        lumen_opacity=lumen_opacity,
    )


def setup_organoid(outer_radius=120, wall_thickness=20, n_cells=400, n_buds=0, seed=42, internal_scale=4, lumen_opacity=0.6):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(outer_radius=outer_radius, wall_thickness=wall_thickness, n_cells=n_cells, n_buds=n_buds, seed=seed, internal_scale=internal_scale, lumen_opacity=lumen_opacity)
    core = load_cfg(sim, Path(__file__).parent / "organoid.cfg")
    return core, sim
