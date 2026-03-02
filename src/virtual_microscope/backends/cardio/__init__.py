"""cardio backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates beating cardiomyocytes with calcium transients and arrhythmia. Supports SLM pacing, temperature control, and perfusion.",
    "channels": ["phase-contrast", "GCaMP", "cell-junctions"],
    "continuous": True,
    "extra_devices": ["SLM", "Temperature", "Perfusion"],
    "specimen": "iPSC-derived cardiomyocytes",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "Observe beating cardiomyocytes with calcium transients visualised via "
        "GCaMP. Normal cells beat synchronously; a fraction exhibit arrhythmic "
        "pacing. Use SLM to optically pace specific regions and restore "
        "synchrony. Perfusion delivers drugs (e.g. isoproterenol) that alter beat "
        "rate. Temperature affects contraction frequency."
    ),
    "device_effects": {
        "SLM": "Optically paces illuminated cells, overriding their intrinsic rhythm",
        "Temperature": "Beat frequency scales with temperature (Q10 ≈ 2)",
        "Perfusion": "Delivers chronotropic drugs that increase or decrease beat rate",
    },
    "key_parameters": {
        "n_cells": "Number of cardiomyocytes (default 300)",
        "normal_freq": "Normal beating frequency in Hz (default 1.0)",
        "arrhythmia_fraction": "Fraction of cells with arrhythmic pacing (default 0.12)",
    },
}

from pathlib import Path

from virtual_microscope.backends.cardio.sim import CardioSim
from virtual_microscope._init_standard import load_cfg


def create_sim(grid_size=512, n_cells=300, normal_freq=1.0, arrhythmia_freq=1.8, arrhythmia_fraction=0.12, coupling_strength=2.0, seed=42, internal_scale=4) -> CardioSim:
    """Create a cardiomyocyte simulation."""
    return CardioSim(
        grid_size=grid_size,
        viewport_width=512,
        viewport_height=512,
        n_cells=n_cells,
        normal_freq=normal_freq,
        arrhythmia_freq=arrhythmia_freq,
        arrhythmia_fraction=arrhythmia_fraction,
        coupling_strength=coupling_strength,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_cardio(grid_size=512, n_cells=300, normal_freq=1.0, arrhythmia_freq=1.8, arrhythmia_fraction=0.12, coupling_strength=2.0, seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(grid_size=grid_size, n_cells=n_cells, normal_freq=normal_freq, arrhythmia_freq=arrhythmia_freq, arrhythmia_fraction=arrhythmia_fraction, coupling_strength=coupling_strength, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "cardio.cfg")
    core.setState("Temperature", 4)  # 37°C
    return core, sim
