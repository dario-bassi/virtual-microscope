"""spt backend for virtual-microscope."""

from virtual_microscope.backends.spt.sim import SPTSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(n_free=20, n_confined=10, n_directed=5, D_free=0.1, D_confined=0.01, D_directed=0.1, confinement_radius=0.5, directed_speed=0.5, blink_rate=0.05, recovery_rate=0.30, bleach_rate=0.002, seed=42) -> SPTSim:
    """Create an SPT simulation."""
    return SPTSim(
        n_free=n_free,
        n_confined=n_confined,
        n_directed=n_directed,
        D_free=D_free,
        D_confined=D_confined,
        D_directed=D_directed,
        confinement_radius=confinement_radius,
        directed_speed=directed_speed,
        blink_rate=blink_rate,
        recovery_rate=recovery_rate,
        bleach_rate=bleach_rate,
        seed=seed,
    )


def setup_spt_microscope(n_free=20, n_confined=10, n_directed=5, D_free=0.1, D_confined=0.01, D_directed=0.1, confinement_radius=0.5, directed_speed=0.5, blink_rate=0.05, recovery_rate=0.30, bleach_rate=0.002, seed=42):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(n_free=n_free, n_confined=n_confined, n_directed=n_directed, D_free=D_free, D_confined=D_confined, D_directed=D_directed, confinement_radius=confinement_radius, directed_speed=directed_speed, blink_rate=blink_rate, recovery_rate=recovery_rate, bleach_rate=bleach_rate, seed=seed)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    core.defineConfig("Fake", "spt-channel", "LED", "Label", "GREEN")
    core.defineConfig("Fake", "spt-channel", "Filter Wheel", "Label", "TagGFP2(483/506)")
    core.setConfig("Fake", "spt-channel")
    return core, sim
