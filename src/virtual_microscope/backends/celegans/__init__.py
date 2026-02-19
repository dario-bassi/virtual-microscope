"""celegans backend for virtual-microscope."""

from virtual_microscope.backends.celegans.sim import CelegansSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


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


def setup_celegans_microscope(world_size=2048, worm_length=250.0, worm_width=18.0, speed=40.0, seed=42, fixed_dt=1.0):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, worm_length=worm_length, worm_width=worm_width, speed=speed, seed=seed, fixed_dt=fixed_dt)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
