"""reaction_diffusion backend for virtual-microscope."""

from virtual_microscope.backends.reaction_diffusion.sim import ReactionDiffusionSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from virtual_microscope.devices.state import GenericStateDevice
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(grid_size=512, preset="waves", F=None, K=None, steps_per_snap=200, seed=42) -> ReactionDiffusionSim:
    """Create a reaction-diffusion simulation."""
    return ReactionDiffusionSim(
        grid_size=grid_size,
        viewport_width=512,
        viewport_height=512,
        preset=preset,
        F=F,
        K=K,
        steps_per_snap=steps_per_snap,
        seed=seed,
    )


def setup_rd_microscope(grid_size=512, preset="waves", F=None, K=None, steps_per_snap=200, seed=42):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(grid_size=grid_size, preset=preset, F=F, K=K, steps_per_snap=steps_per_snap, seed=seed)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    core.loadPyDevice("SLM-Mode", GenericStateDevice("SLM-Mode", {0: "excite", 1: "inhibit"}))
    core.initializeDevice("SLM-Mode")
    core.setState("SLM-Mode", 0)
    return core, sim


# Alias for load_backend("reaction_diffusion") compatibility
setup_reaction_diffusion_microscope = setup_rd_microscope
