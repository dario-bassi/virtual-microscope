"""fibroblast backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates adherent fibroblasts with actin stress fibres and focal adhesions. Supports temperature, perfusion, and mechanical stretch.",
    "channels": ["brightfield", "DAPI", "phalloidin"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion", "Stretch"],
    "specimen": "Adherent fibroblasts (primary or cell line)",
    "modality": "Brightfield + epifluorescence",
    "experiment_guide": (
        "Observe adherent fibroblasts with actin stress fibres (phalloidin) and "
        "nuclear staining (DAPI). Apply mechanical stretch to study cytoskeletal "
        "remodelling. Perfusion delivers cytochalasin-D or latrunculin-A to "
        "disrupt the actin network. Temperature affects cell spreading and "
        "migration."
    ),
    "device_effects": {
        "Temperature": "Higher temperature increases cell spreading and migration speed",
        "Perfusion": "Delivers actin-disrupting drugs (CytoD, LatA) that dissolve stress fibres",
        "Stretch": "Applies uniaxial mechanical stretch, reorienting stress fibres perpendicular to strain",
    },
    "key_parameters": {
        "n_cells": "Number of fibroblasts (default 8)",
        "world_size": "World size in pixels (default 512)",
    },
}

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
