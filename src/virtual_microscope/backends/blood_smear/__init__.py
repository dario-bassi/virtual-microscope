"""blood_smear backend for virtual-microscope."""

from virtual_microscope.backends.blood_smear.sim import BloodSmearSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(world_size=512, n_rbc=800, n_wbc=15, n_platelets=25, seed=42, internal_scale=4) -> BloodSmearSim:
    """Create a blood smear simulation."""
    return BloodSmearSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_rbc=n_rbc,
        n_wbc=n_wbc,
        n_platelets=n_platelets,
        seed=seed,
        abnormal_rbc={},
        internal_scale=internal_scale,
    )


def setup_blood_smear_microscope(world_size=512, n_rbc=800, n_wbc=15, n_platelets=25, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, n_rbc=n_rbc, n_wbc=n_wbc, n_platelets=n_platelets, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
