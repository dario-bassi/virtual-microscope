"""volvox backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a swimming Volvox colony with somatic cells and gonidia. Features SLM phototaxis control with chlorophyll and pherophorin channels.",
    "channels": ["brightfield", "chlorophyll", "pherophorin"],
    "continuous": True,
    "extra_devices": ["SLM", "Temperature"],
    "specimen": "Volvox carteri (colonial green alga)",
    "modality": "Brightfield + epifluorescence",
    "experiment_guide": (
        "Watch a Volvox colony swim and rotate. Somatic cells on the surface beat "
        "flagella for locomotion; large gonidia inside develop into daughter "
        "colonies. Chlorophyll autofluorescence labels all cells; pherophorin "
        "marks the extracellular matrix. SLM phototaxis steers the colony toward "
        "or away from light."
    ),
    "device_effects": {
        "SLM": "Directional light cue that steers phototactic swimming toward illuminated region",
        "Temperature": "Affects flagellar beat frequency and swimming speed",
    },
    "key_parameters": {
        "n_somatic": "Number of somatic cells (default 300)",
        "n_gonidia": "Number of reproductive gonidia (default 4)",
        "colony_radius": "Colony radius in pixels (default 60)",
        "swim_speed": "Swimming speed in px/step (default 3.0)",
    },
}

from pathlib import Path

from virtual_microscope.backends.volvox.sim import VolvoxSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_somatic=300, n_gonidia=4, colony_radius=60.0, swim_speed=3.0, rotation_speed=0.15, seed=42, internal_scale=4) -> VolvoxSim:
    """Create a Volvox simulation."""
    return VolvoxSim(
        n_somatic=n_somatic,
        n_gonidia=n_gonidia,
        colony_radius=colony_radius,
        swim_speed=swim_speed,
        rotation_speed=rotation_speed,
        world_size=512,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_volvox(n_somatic=300, n_gonidia=4, colony_radius=60.0, swim_speed=3.0, rotation_speed=0.15, seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_somatic=n_somatic, n_gonidia=n_gonidia, colony_radius=colony_radius, swim_speed=swim_speed, rotation_speed=rotation_speed, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "volvox.cfg")
    return core, sim
