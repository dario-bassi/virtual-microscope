"""microfluidics backend for virtual-microscope."""

from virtual_microscope.backends.microfluidics.sim import MicrofluidicsSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


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


def setup_microfluidics_microscope(n_cells=30, channel_width=100, flow_speed=3.0, n_traps=0, gradient=False, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_cells=n_cells, channel_width=channel_width, flow_speed=flow_speed, n_traps=n_traps, gradient=gradient, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
