"""fucci backend for virtual-microscope (FucciSim + FUCCI reporter)."""

BACKEND_INFO = {
    "description": "Simulates FUCCI cell-cycle reporter with mCherry-Cdt1 (G1) and mVenus-Geminin (S/G2/M). Features configurable phase durations with temperature and perfusion.",
    "channels": ["phase-contrast", "mCherry-Cdt1", "mVenus-Geminin"],
    "continuous": True,
    "extra_devices": ["Temperature", "Perfusion"],
}

from pathlib import Path

from virtual_microscope.backends.fucci.sim import FucciSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_cells=60, seed=42, g1_duration=30, s_duration=15, g2_duration=10, m_duration=8, division_on_m=True, width=512, height=512, internal_scale=4) -> FucciSim:
    """Create a FucciSim for FUCCI cell cycle imaging."""
    return FucciSim(
        n_cells=n_cells,
        width=width,
        height=height,
        viewport_width=512,
        viewport_height=512,
        seed=seed,
        jitter=0.7,
        nucleus_fraction=0.30,
        internal_scale=internal_scale,
        textured_nuclei=True,
    )


def setup_fucci(n_cells=60, seed=42, g1_duration=30, s_duration=15, g2_duration=10, m_duration=8, division_on_m=True, **kwargs):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_cells=n_cells, seed=seed, g1_duration=g1_duration, s_duration=s_duration, g2_duration=g2_duration, m_duration=m_duration, division_on_m=division_on_m)
    core = load_cfg(sim, Path(__file__).parent / "fucci.cfg")
    sim.enable_fucci_reporter(
        g1_duration=g1_duration,
        s_duration=s_duration,
        g2_duration=g2_duration,
        m_duration=m_duration,
        division_on_m=division_on_m,
        core=core,
    )
    return core, sim
