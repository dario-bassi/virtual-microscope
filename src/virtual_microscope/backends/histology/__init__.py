"""histology backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates H&E-stained histology tissue sections with configurable tissue type and tumour grade. Supports separate hematoxylin and eosin pseudo-channels.",
    "channels": ["H-and-E", "hematoxylin", "eosin"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "H&E-stained tissue section",
    "modality": "Brightfield histopathology (RGB output)",
    "experiment_guide": (
        "A static stained tissue section. Switch between composite H&E view, "
        "isolated hematoxylin (nuclei), and eosin (cytoplasm/stroma) channels. "
        "Navigate across the slide with XY stage and zoom with objectives. Useful "
        "for training histopathology image analysis pipelines."
    ),
    "device_effects": {},
    "key_parameters": {
        "tissue_type": "Tissue morphology preset (default 'glandular')",
        "grade": "Tumour differentiation grade (default 0 = normal)",
        "n_nuclei": "Number of nuclei to generate (default 200)",
    },
}

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


def setup_histology(tissue_type="glandular", world_size=512, n_nuclei=200, seed=42, internal_scale=4, grade=0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(tissue_type=tissue_type, world_size=world_size, n_nuclei=n_nuclei, seed=seed, internal_scale=internal_scale, grade=grade)
    core = load_cfg(sim, Path(__file__).parent / "histology.cfg")
    return core, sim
