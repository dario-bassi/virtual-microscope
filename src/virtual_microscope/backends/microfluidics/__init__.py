"""microfluidics backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates cells flowing through a microfluidic channel with optional traps and chemical gradients. Supports temperature and perfusion control.",
    "channels": ["phase-contrast", "DAPI", "fluorescein"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion"],
    "specimen": "Cells in a microfluidic channel",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "Watch cells flowing through a microfluidic channel with optional "
        "trapping posts and chemical gradients. Use perfusion to control flow "
        "speed and deliver fluorescein for gradient visualisation. Temperature "
        "affects cell viability and motility. Useful for studying shear stress "
        "effects and chemotaxis in confined geometries."
    ),
    "device_effects": {
        "Temperature": "Affects cell viability and motility in the channel",
        "Perfusion": "Controls flow speed and delivers chemical gradients",
    },
    "key_parameters": {
        "n_cells": "Number of cells in the channel (default 30)",
        "channel_width": "Channel width in pixels (default 100)",
        "flow_speed": "Flow speed in px/step (default 3.0)",
        "n_traps": "Number of trapping posts (default 0)",
    },
}

from pathlib import Path

from virtual_microscope.backends.microfluidics.sim import MicrofluidicsSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=30, channel_width=100, flow_speed=3.0, n_traps=0, gradient=False, seed=42, internal_scale=4) -> MicrofluidicsSim:
    """Create a microfluidics simulation."""
    return MicrofluidicsSim(
        n_cells=n_cells,
        channel_width=channel_width,
        flow_speed=flow_speed,
        n_traps=n_traps,
        gradient=gradient,
        world_size=512,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_microfluidics(n_cells=30, channel_width=100, flow_speed=3.0, n_traps=0, gradient=False, seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, channel_width=channel_width, flow_speed=flow_speed, n_traps=n_traps, gradient=gradient, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "microfluidics.cfg")
    return core, sim
