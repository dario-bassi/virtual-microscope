"""plant_cell backend for virtual-microscope."""

from virtual_microscope.backends.plant_cell.sim import PlantCellSim
from virtual_microscope.simulation_bridge import SimulationBridge
import virtual_microscope.simulation_bridge as bridge_module
from virtual_microscope._init_standard import init_standard_devices
from pymmcore_plus.experimental.unicore import UniMMCore


def create_sim(world_size=512, cell_length_range=(150, 250), cell_width_range=(30, 50), wall_thickness=7.0, staining="iodine", seed=42, internal_scale=4) -> PlantCellSim:
    """Create a plant cell simulation."""
    return PlantCellSim(
        world_size=world_size,
        viewport_width=512,
        viewport_height=512,
        cell_length_range=cell_length_range,
        cell_width_range=cell_width_range,
        wall_thickness=wall_thickness,
        staining=staining,
        seed=seed,
        internal_scale=internal_scale,
    )


def setup_plant_cell_microscope(world_size=512, cell_length_range=(150, 250), cell_width_range=(30, 50), wall_thickness=7.0, staining="iodine", seed=42, internal_scale=4):
    """Programmatic setup (no .cfg needed)."""
    sim = create_sim(world_size=world_size, cell_length_range=cell_length_range, cell_width_range=cell_width_range, wall_thickness=wall_thickness, staining=staining, seed=seed, internal_scale=internal_scale)
    bridge_module.GLOBAL_BRIDGE = SimulationBridge(sim)
    core = UniMMCore()
    init_standard_devices(core, sim)
    return core, sim
