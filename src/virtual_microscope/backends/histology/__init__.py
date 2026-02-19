"""histology backend for virtual-microscope."""

from virtual_microscope.backends.histology.sim import HistologySim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(tissue_type="glandular", world_size=512, n_nuclei=200, seed=42, internal_scale=4, grade=0) -> HistologySim:
    """Create a histology simulation."""
    return HistologySim(
        tissue_type=tissue_type,
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        n_nuclei=n_nuclei,
        seed=seed,
        internal_scale=internal_scale,
        grade=grade,
    )


def setup_histology_microscope(tissue_type="glandular", world_size=512, n_nuclei=200, seed=42, internal_scale=4, grade=0):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(tissue_type=tissue_type, world_size=world_size, n_nuclei=n_nuclei, seed=seed, internal_scale=internal_scale, grade=grade)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
