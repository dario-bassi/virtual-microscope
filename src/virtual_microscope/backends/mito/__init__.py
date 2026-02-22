"""mito backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates mitochondrial networks with tubules undergoing fission and fusion. Features MitoTracker staining with configurable fragmentation dynamics.",
    "channels": ["brightfield", "DAPI", "MitoTracker"],
    "continuous": True,
    "extra_devices": [],
}

from pathlib import Path

from virtual_microscope.backends.mito.sim import MitoSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=512, n_tubules=40, fragmentation=0.0, fission_rate=0.03, fusion_rate=0.03, seed=42, fixed_dt=5.0, internal_scale=4) -> MitoSim:
    """Create a mitochondria simulation."""
    return MitoSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_tubules=n_tubules,
        fragmentation=fragmentation,
        fission_rate=fission_rate,
        fusion_rate=fusion_rate,
        seed=seed,
        fixed_dt=fixed_dt,
        internal_scale=internal_scale,
    )


def setup_mito(world_size=512, n_tubules=40, fragmentation=0.0, fission_rate=0.03, fusion_rate=0.03, seed=42, fixed_dt=5.0, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, n_tubules=n_tubules, fragmentation=fragmentation, fission_rate=fission_rate, fusion_rate=fusion_rate, seed=seed, fixed_dt=fixed_dt, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "mito.cfg")
    return core, sim
