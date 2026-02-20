"""plant_cell backend for virtual-microscope."""

from pathlib import Path

from virtual_microscope.backends.plant_cell.sim import PlantCellSim
from virtual_microscope._init_standard import load_cfg


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
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    sim = create_sim(world_size=world_size, cell_length_range=cell_length_range, cell_width_range=cell_width_range, wall_thickness=wall_thickness, staining=staining, seed=seed, internal_scale=internal_scale)
    core = load_cfg(sim, Path(__file__).parent / "plant_cell.cfg")
    return core, sim
