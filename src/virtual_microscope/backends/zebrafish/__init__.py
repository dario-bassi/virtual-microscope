"""zebrafish backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.zebrafish.sim import ZebrafishSim
from virtual_microscope._init_standard import load_cfg


def create_sim(n_rbc=30, cardiac_freq=2.5, seed=42, internal_scale=2) -> ZebrafishSim:
    """Create a zebrafish simulation."""
    return ZebrafishSim(n_rbc=n_rbc, cardiac_freq=cardiac_freq, seed=seed, internal_scale=internal_scale)


def setup_zebrafish_microscope(n_rbc=30, cardiac_freq=2.5, seed=42, internal_scale=2):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(n_rbc=n_rbc, cardiac_freq=cardiac_freq, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "zebrafish.cfg")
    core.setState("Temperature", 9)  # 28°C optimal
    # Zebrafish pixel size: 5.0/2.5/1.25/0.625 µm/px
    _zf_px = {0: 5.0, 1: 2.5, 2: 1.25, 3: 0.625}
    core.getPixelSizeUm = lambda cached=False: _zf_px.get(core.getState("Objective"), 1.0)
    return core, sim
