"""celegans backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a crawling C. elegans nematode with sinusoidal body bending. Features pharyngeal GFP and body-wall mCherry with temperature, perfusion, and electrode control.",
    "channels": ["DIC", "GFP-pharynx", "mCherry-body"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion"],
    "specimen": "C. elegans (nematode worm)",
    "modality": "DIC + epifluorescence",
    "experiment_guide": (
        "Track a crawling C. elegans nematode with sinusoidal body bending. Use "
        "DIC for body morphology, GFP-pharynx for the feeding organ, and "
        "mCherry-body for body-wall muscle. Temperature affects crawling speed; "
        "perfusion can deliver paralytic agents (e.g. levamisole)."
    ),
    "device_effects": {
        "Temperature": "Crawling speed scales with temperature; low temperature slows locomotion",
        "Perfusion": "Delivers paralytic agents or nutrients affecting worm behaviour",
    },
    "key_parameters": {
        "worm_length": "Worm body length in pixels (default 250)",
        "worm_width": "Worm body width in pixels (default 18)",
        "speed": "Crawling speed in px/s (default 40)",
    },
}

from pathlib import Path

from virtual_microscope.backends.celegans.sim import CelegansSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=2048, worm_length=250.0, worm_width=18.0, speed=40.0, seed=42, fixed_dt=1.0) -> CelegansSim:
    """Create a C. elegans simulation."""
    return CelegansSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        worm_length=worm_length,
        worm_width=worm_width,
        speed=speed,
        seed=seed,
        fixed_dt=fixed_dt,
    )


def setup_celegans(world_size=2048, worm_length=250.0, worm_width=18.0, speed=40.0, seed=42, fixed_dt=1.0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, worm_length=worm_length, worm_width=worm_width, speed=speed, seed=seed, fixed_dt=fixed_dt)
    core = load_cfg(sim, Path(__file__).parent / "celegans.cfg")
    return core, sim
