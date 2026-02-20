"""reaction_diffusion backend for virtual-microscope."""

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


def setup_rd_microscope(grid_size=512, preset="waves", F=None, K=None, steps_per_snap=200, seed=42):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(grid_size=grid_size, preset=preset, F=F, K=K, steps_per_snap=steps_per_snap, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "reaction_diffusion.cfg")
    return core, sim


# Alias for load_backend("reaction_diffusion") compatibility
setup_reaction_diffusion_microscope = setup_rd_microscope
