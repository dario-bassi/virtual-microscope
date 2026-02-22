"""fish backend for virtual-microscope (FishSim + FISH probes)."""

from pathlib import Path

from virtual_microscope.backends.fish.sim import FishSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=40, seed=42, locus_copies=2, amplified_fraction=0.15, deleted_fraction=0.10, amplified_copies_range=(3, 6), width=512, height=512, internal_scale=4) -> FishSim:
    """Create a FishSim for FISH probe imaging."""
    return FishSim(
        n_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        jitter=0.6,
        nucleus_fraction=0.32,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_fish(n_cells=40, seed=42, locus_copies=2, amplified_fraction=0.15, deleted_fraction=0.10, amplified_copies_range=(3, 6), **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, locus_copies=locus_copies, amplified_fraction=amplified_fraction, deleted_fraction=deleted_fraction, amplified_copies_range=amplified_copies_range)
    core = load_cfg(sim, Path(__file__).parent / "fish.cfg")
    sim.enable_fish_probes(
        core=core,
        locus_copies=locus_copies,
        amplified_fraction=amplified_fraction,
        deleted_fraction=deleted_fraction,
        amplified_copies_range=amplified_copies_range,
        probe_intensity=215.0,
        probe_intensity_std=18.0,
        fwhm_world_px=1.8,
    )
    return core, sim
