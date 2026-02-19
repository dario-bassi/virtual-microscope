"""neuron backend for virtual-microscope."""

from virtual_microscope.backends.neuron.sim import NeuronSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_neurons=8, world_size=512, seed=42, internal_scale=4) -> NeuronSim:
    """Create a neuron simulation."""
    return NeuronSim(n_neurons=n_neurons, world_size=world_size, seed=seed, internal_scale=internal_scale)


def setup_neuron_microscope(n_neurons=8, world_size=512, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_neurons=n_neurons, world_size=world_size, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
