"""reaction_diffusion backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates Gray-Scott reaction-diffusion Turing patterns with activator and inhibitor species. Supports SLM-based perturbation with configurable F/K parameters.",
    "channels": ["both-species", "activator-U", "inhibitor-V"],
    "continuous": True,
    "extra_devices": ["SLM", "SLM-Mode"],
    "specimen": "Gray–Scott reaction–diffusion system",
    "modality": "Pseudo-fluorescence (activator / inhibitor concentration maps)",
    "experiment_guide": (
        "Watch Turing patterns emerge from a Gray–Scott reaction–diffusion "
        "system. The activator (U) and inhibitor (V) form spots, stripes, or "
        "waves depending on feed/kill parameters (F, K). Use SLM masks to locally "
        "perturb concentrations and seed new pattern domains. SLM-Mode toggles "
        "between adding activator and adding inhibitor."
    ),
    "device_effects": {
        "SLM": "Locally perturbs species concentrations in illuminated regions",
        "SLM-Mode": "Toggles perturbation target between activator (U) and inhibitor (V)",
    },
    "key_parameters": {
        "grid_size": "Simulation grid size in pixels (default 512)",
        "preset": "Pattern preset: 'waves', 'spots', 'stripes', etc. (default 'waves')",
        "F": "Feed rate (default depends on preset)",
        "K": "Kill rate (default depends on preset)",
    },
}

from pathlib import Path

from virtual_microscope.backends.reaction_diffusion.sim import ReactionDiffusionSim
from virtual_microscope._init_standard import load_cfg


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


def setup_reaction_diffusion(grid_size=512, preset="waves", F=None, K=None, steps_per_snap=200, seed=42):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(grid_size=grid_size, preset=preset, F=F, K=K, steps_per_snap=steps_per_snap, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "reaction_diffusion.cfg")
    return core, sim

