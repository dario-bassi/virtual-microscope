"""blood_smear backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a stained peripheral blood smear with RBCs, WBCs, and platelets. Supports abnormal morphologies and rouleaux formation.",
    "channels": ["wright-giemsa", "nuclei-aid", "membrane-aid"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Peripheral blood smear (Wright-Giemsa stain)",
    "modality": "Brightfield histology (RGB output)",
    "experiment_guide": (
        "A fixed, stained peripheral blood smear. Navigate the slide to find "
        "RBCs, WBCs (neutrophils, lymphocytes, monocytes), and platelets. Switch "
        "between composite Wright-Giemsa view and auxiliary channels highlighting "
        "nuclei or membranes. Useful for training automated differential "
        "white-cell counters and detecting abnormal morphologies."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_rbc": "Red blood cell count (default 800)",
        "n_wbc": "White blood cell count (default 15)",
        "abnormal_rbc": "Dict of abnormality fractions (e.g. sickle, target)",
        "rouleaux_fraction": "Fraction of RBCs in rouleaux stacks (default 0.0)",
    },
}

from pathlib import Path

from virtual_microscope.backends.blood_smear.sim import BloodSmearSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=512, n_rbc=800, n_wbc=15, n_platelets=25,
               abnormal_rbc=None, rouleaux_fraction=0.0,
               seed=42, internal_scale=4) -> BloodSmearSim:
    """Create a blood smear simulation."""
    return BloodSmearSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_rbc=n_rbc,
        n_wbc=n_wbc,
        n_platelets=n_platelets,
        seed=seed,
        abnormal_rbc=abnormal_rbc or {},
        rouleaux_fraction=rouleaux_fraction,
        internal_scale=internal_scale,
    )


def setup_blood_smear(world_size=512, n_rbc=800, n_wbc=15,
                                 n_platelets=25, abnormal_rbc=None,
                                 rouleaux_fraction=0.0,
                                 seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(
        world_size=world_size, n_rbc=n_rbc, n_wbc=n_wbc,
        n_platelets=n_platelets, abnormal_rbc=abnormal_rbc,
        rouleaux_fraction=rouleaux_fraction,
        seed=seed, internal_scale=internal_scale,
    )
    core = load_cfg(sim, Path(__file__).parent / "blood_smear.cfg")
    return core, sim
