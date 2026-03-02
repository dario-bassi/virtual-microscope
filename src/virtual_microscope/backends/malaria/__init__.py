"""malaria backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a Giemsa-stained malaria blood smear with intra-erythrocytic Plasmodium parasites. Features configurable parasitemia, stage progression, and appliqué forms.",
    "channels": ["giemsa", "chromatin-aid", "RBC-overlay"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Giemsa-stained thin blood smear with Plasmodium parasites",
    "modality": "Brightfield histology (RGB output)",
    "experiment_guide": (
        "A Giemsa-stained thin blood smear with intra-erythrocytic Plasmodium "
        "parasites at various developmental stages (ring, trophozoite, schizont). "
        "Navigate the smear to find infected RBCs and count parasitemia. Useful "
        "for training malaria diagnostic algorithms and studying parasite "
        "morphology including appliqué (accolé) forms."
    ),
    "device_effects": {},
    "key_parameters": {
        "parasitemia": "Fraction of infected RBCs (default 0.05)",
        "n_rbc": "Total red blood cell count (default 2000)",
        "hours_per_step": "Hours of parasite development per step (default 2.0)",
    },
}

from pathlib import Path

from virtual_microscope.backends.malaria.sim import MalariaSmearSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=512, n_rbc=2000, n_wbc=8, parasitemia=0.05,
               hours_per_step=2.0, n_platelets=0, applique_rate=0.35,
               seed=42, internal_scale=4) -> MalariaSmearSim:
    """Create a malaria smear simulation."""
    sim = MalariaSmearSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_rbc=n_rbc,
        n_wbc=n_wbc,
        parasitemia=parasitemia,
        n_platelets=n_platelets,
        applique_rate=applique_rate,
        seed=seed,
        internal_scale=internal_scale,
    )
    sim._hours_per_step = hours_per_step
    return sim


def setup_malaria(world_size=512, n_rbc=2000, n_wbc=8, parasitemia=0.05,
                             hours_per_step=2.0, n_platelets=0, applique_rate=0.35,
                             seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, n_rbc=n_rbc, n_wbc=n_wbc,
                     parasitemia=parasitemia, hours_per_step=hours_per_step,
                     n_platelets=n_platelets, applique_rate=applique_rate,
                     seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "malaria.cfg")
    return core, sim
