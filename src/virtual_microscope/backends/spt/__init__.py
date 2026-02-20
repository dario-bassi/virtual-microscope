"""spt backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.spt.sim import SPTSim
from virtual_microscope._init_standard import load_cfg


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
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_free=n_free, n_confined=n_confined, n_directed=n_directed, D_free=D_free, D_confined=D_confined, D_directed=D_directed, confinement_radius=confinement_radius, directed_speed=directed_speed, blink_rate=blink_rate, recovery_rate=recovery_rate, bleach_rate=bleach_rate, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "spt.cfg")
    core.setConfig("Channel", "TIRF")
    return core, sim
