"""dictyostelium backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates Dictyostelium discoideum chemotaxis with cAMP wave relay. Features pacemaker-driven aggregation with SLM stimulation and temperature control.",
    "channels": ["dark-field", "GFP", "cAMP-reporter"],
    "continuous": True,
    "extra_devices": ["SLM", "Temperature"],
    "specimen": "Dictyostelium discoideum (social amoeba)",
    "modality": "Dark-field + epifluorescence",
    "experiment_guide": (
        "Watch Dictyostelium amoebae aggregate via cAMP chemotaxis. Pacemaker "
        "cells emit periodic cAMP pulses that are relayed outward as spiral "
        "waves, guiding cells toward aggregation centres. Use SLM to create "
        "artificial cAMP sources and redirect streaming. Temperature modulates "
        "relay kinetics and aggregation speed."
    ),
    "device_effects": {
        "SLM": "Creates artificial cAMP point sources in illuminated regions",
        "Temperature": "Modulates cAMP relay kinetics and cell motility speed",
    },
    "key_parameters": {
        "n_cells": "Number of amoebae (default 100)",
        "n_pacemakers": "Number of pacemaker cells (default 3)",
        "relay_radius": "cAMP relay radius in pixels (default 40)",
    },
}

from pathlib import Path

from virtual_microscope.backends.dictyostelium.sim import DictyosteliumSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=100, n_pacemakers=3, seed=42, world_size=512, relay_radius=40.0, pulse_period=10, camp_reporter_gain=1.0) -> DictyosteliumSim:
    """Create a Dictyostelium simulation."""
    return DictyosteliumSim(
        n_cells=n_cells,
        n_pacemakers=n_pacemakers,
        seed=seed,
        world_size=world_size,
        relay_radius=relay_radius,
        pulse_period=pulse_period,
        camp_reporter_gain=camp_reporter_gain,
    )


def setup_dictyostelium(n_cells=100, n_pacemakers=3, seed=42, world_size=512, relay_radius=40.0, pulse_period=10, camp_reporter_gain=1.0):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, n_pacemakers=n_pacemakers, seed=seed, world_size=world_size, relay_radius=relay_radius, pulse_period=pulse_period, camp_reporter_gain=camp_reporter_gain)
    core = load_cfg(sim, Path(__file__).parent / "dictyostelium.cfg")
    return core, sim
