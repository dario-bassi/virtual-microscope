"""hemocytometer backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a hemocytometer counting chamber with trypan-blue viability staining. Supports cell clumps and configurable dilution factor.",
    "channels": ["brightfield", "trypan-blue"],
    "continuous": False,
    "extra_devices": [],
}

from virtual_microscope.backends.hemocytometer.sim import HemocytometerSim


def create_sim(n_cells=150, viability=0.85, cell_radius_range=(3, 7), dilution_factor=2, clump_fraction=0.0, seed=42) -> HemocytometerSim:
    """Create a hemocytometer simulation."""
    return HemocytometerSim(
        n_cells=n_cells,
        viability=viability,
        cell_radius_range=cell_radius_range,
        dilution_factor=dilution_factor,
        clump_fraction=clump_fraction,
        seed=seed,
    )


def setup_hemocytometer(n_cells=150, viability=0.85, cell_radius_range=(3, 7), dilution_factor=2, clump_fraction=0.0, seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope._init_standard import load_standalone
    from virtual_microscope.devices.state import GenericStateDevice

    sim = create_sim(n_cells=n_cells, viability=viability, cell_radius_range=cell_radius_range, dilution_factor=dilution_factor, clump_fraction=clump_fraction, seed=seed)
    labels = {0: "brightfield", 1: "trypan-blue"}
    core = load_standalone(sim,
        channels={
            "brightfield": ("Channel", "Label", "brightfield"),
            "trypan-blue": ("Channel", "Label", "trypan-blue"),
        },
        extra_devices={"Channel": GenericStateDevice("Channel", labels)},
    )
    return core, sim
