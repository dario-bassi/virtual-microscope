"""malaria backend for virtual-microscope."""

from virtual_microscope.backends.malaria.sim import MalariaSmearSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(world_size=512, n_rbc=2000, n_wbc=8, parasitemia=0.05, hours_per_step=2.0, seed=42, internal_scale=4) -> MalariaSmearSim:
    """Create a malaria smear simulation."""
    sim = MalariaSmearSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_rbc=n_rbc,
        n_wbc=n_wbc,
        parasitemia=parasitemia,
        seed=seed,
        internal_scale=internal_scale,
    )
    sim._hours_per_step = hours_per_step
    return sim


def setup_malaria_microscope(world_size=512, n_rbc=2000, n_wbc=8, parasitemia=0.05, hours_per_step=2.0, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, n_rbc=n_rbc, n_wbc=n_wbc, parasitemia=parasitemia, hours_per_step=hours_per_step, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
