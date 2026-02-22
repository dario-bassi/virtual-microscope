"""reaction_diffusion backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates Gray-Scott reaction-diffusion Turing patterns with activator and inhibitor species. Supports SLM-based perturbation with configurable F/K parameters.",
    "channels": ["both-species", "activator-U", "inhibitor-V"],
    "continuous": True,
    "extra_devices": ["SLM", "SLM-Mode"],
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

