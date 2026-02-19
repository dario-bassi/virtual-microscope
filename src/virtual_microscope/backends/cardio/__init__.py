"""cardio backend for virtual-microscope."""

from virtual_microscope.backends.cardio.sim import CardioSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


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


def setup_cardio_microscope(grid_size=512, n_cells=300, normal_freq=1.0, arrhythmia_freq=1.8, arrhythmia_fraction=0.12, coupling_strength=2.0, seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(grid_size=grid_size, n_cells=n_cells, normal_freq=normal_freq, arrhythmia_freq=arrhythmia_freq, arrhythmia_fraction=arrhythmia_fraction, coupling_strength=coupling_strength, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    core.setState("Temperature", 4)  # 37°C
    return core, sim
