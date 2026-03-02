"""spt backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates single-particle tracking with free, confined, and directed diffusion modes. Features blinking, bleaching, and widefield/TIRF illumination.",
    "channels": ["widefield", "TIRF"],
    "continuous": True,
    "extra_devices": [],
    "specimen": "Fluorescent nanoparticles / single molecules",
    "modality": "Widefield + TIRF epifluorescence",
    "experiment_guide": (
        "Track single fluorescent particles undergoing free diffusion, confined "
        "diffusion, or directed transport. Particles exhibit stochastic blinking "
        "and irreversible photobleaching. Compare widefield vs. TIRF illumination "
        "for signal-to-noise. Useful for benchmarking single-particle tracking "
        "algorithms."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_free": "Free-diffusion particles (default 20)",
        "n_confined": "Confined-diffusion particles (default 10)",
        "n_directed": "Directed-transport particles (default 5)",
        "D_free": "Free diffusion coefficient (default 0.1)",
    },
}

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


def setup_spt(n_free=20, n_confined=10, n_directed=5, D_free=0.1, D_confined=0.01, D_directed=0.1, confinement_radius=0.5, directed_speed=0.5, blink_rate=0.05, recovery_rate=0.30, bleach_rate=0.002, seed=42):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_free=n_free, n_confined=n_confined, n_directed=n_directed, D_free=D_free, D_confined=D_confined, D_directed=D_directed, confinement_radius=confinement_radius, directed_speed=directed_speed, blink_rate=blink_rate, recovery_rate=recovery_rate, bleach_rate=bleach_rate, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "spt.cfg")
    core.setConfig("Channel", "TIRF")
    return core, sim
