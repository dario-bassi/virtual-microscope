"""organoid backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a 3D organoid cross-section with lumen, wall cells, and optional budding. Features E-cadherin junctional staining and DAPI nuclear label.",
    "channels": ["brightfield", "DAPI", "E-cadherin"],
    "continuous": True,
    "extra_devices": [],
    "specimen": "Intestinal organoid cross-section",
    "modality": "Brightfield + epifluorescence",
    "experiment_guide": (
        "View a cross-section through a 3D organoid with a central lumen, "
        "polarised wall cells, and optional budding crypts. E-cadherin highlights "
        "cell–cell junctions; DAPI labels nuclei. Track organoid growth and "
        "morphogenesis over time. Useful for training organoid segmentation and "
        "morphometric analysis pipelines."
    ),
    "device_effects": {},
    "key_parameters": {
        "outer_radius": "Organoid outer radius in pixels (default 120)",
        "n_cells": "Number of cells in the organoid wall (default 400)",
        "n_buds": "Number of crypt buds (default 0)",
    },
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
