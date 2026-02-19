"""mito backend for virtual-microscope."""

from virtual_microscope.backends.mito.sim import MitoSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(world_size=512, n_tubules=40, fragmentation=0.0, fission_rate=0.03, fusion_rate=0.03, seed=42, fixed_dt=5.0, internal_scale=4) -> MitoSim:
    """Create a mitochondria simulation."""
    return MitoSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_tubules=n_tubules,
        fragmentation=fragmentation,
        fission_rate=fission_rate,
        fusion_rate=fusion_rate,
        seed=seed,
        fixed_dt=fixed_dt,
        internal_scale=internal_scale,
    )


def setup_mito_microscope(world_size=512, n_tubules=40, fragmentation=0.0, fission_rate=0.03, fusion_rate=0.03, seed=42, fixed_dt=5.0, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, n_tubules=n_tubules, fragmentation=fragmentation, fission_rate=fission_rate, fusion_rate=fusion_rate, seed=seed, fixed_dt=fixed_dt, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
