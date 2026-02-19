"""volvox backend for virtual-microscope."""

from virtual_microscope.backends.volvox.sim import VolvoxSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


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


def setup_volvox_microscope(n_somatic=300, n_gonidia=4, colony_radius=60.0, swim_speed=3.0, rotation_speed=0.15, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_somatic=n_somatic, n_gonidia=n_gonidia, colony_radius=colony_radius, swim_speed=swim_speed, rotation_speed=rotation_speed, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
