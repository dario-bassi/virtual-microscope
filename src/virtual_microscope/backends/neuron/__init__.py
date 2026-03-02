"""neuron backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates cultured neurons with branching dendrites and axons. Features MAP2-GFP and synaptophysin markers for neuronal morphology.",
    "channels": ["phase-contrast", "MAP2-GFP", "synaptophysin"],
    "continuous": True,
    "extra_devices": [],
    "specimen": "Primary cortical neurons in culture",
    "modality": "Phase-contrast + epifluorescence",
    "experiment_guide": (
        "Observe cultured neurons with branching dendrites and axons. MAP2-GFP "
        "labels dendrites; synaptophysin marks presynaptic terminals. Track "
        "neurite outgrowth and synapse formation over time. Useful for training "
        "neurite-tracing and synapse-detection algorithms."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_neurons": "Number of neurons (default 8)",
        "world_size": "World size in pixels (default 512)",
    },
}

from pathlib import Path

from virtual_microscope.backends.neuron.sim import NeuronSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_neurons=8, world_size=512, seed=42, internal_scale=4) -> NeuronSim:
    """Create a neuron simulation."""
    return NeuronSim(n_neurons=n_neurons, world_size=world_size, seed=seed, internal_scale=internal_scale)


def setup_neuron(n_neurons=8, world_size=512, seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_neurons=n_neurons, world_size=world_size, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "neuron.cfg")
    return core, sim
