"""malaria backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.malaria.sim import MalariaSmearSim
from virtual_microscope._init_standard import load_cfg


def create_sim(world_size=512, n_rbc=2000, n_wbc=8, parasitemia=0.05,
               hours_per_step=2.0, n_platelets=0, applique_rate=0.35,
               seed=42, internal_scale=4) -> MalariaSmearSim:
    """Create a malaria smear simulation."""
    sim = MalariaSmearSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_rbc=n_rbc,
        n_wbc=n_wbc,
        parasitemia=parasitemia,
        n_platelets=n_platelets,
        applique_rate=applique_rate,
        seed=seed,
        internal_scale=internal_scale,
    )
    sim._hours_per_step = hours_per_step
    return sim


def setup_malaria(world_size=512, n_rbc=2000, n_wbc=8, parasitemia=0.05,
                             hours_per_step=2.0, n_platelets=0, applique_rate=0.35,
                             seed=42, internal_scale=4):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, n_rbc=n_rbc, n_wbc=n_wbc,
                     parasitemia=parasitemia, hours_per_step=hours_per_step,
                     n_platelets=n_platelets, applique_rate=applique_rate,
                     seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "malaria.cfg")
    return core, sim
