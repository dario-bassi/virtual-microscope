"""colony_counter backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates bacterial colony plates for colony counting assays. Supports spread/streak plate types with optional blue-white screening and GFP.",
    "channels": ["plate-image", "blue-channel", "gfp-channel"],
    "continuous": False,
    "extra_devices": [],
}

from virtual_microscope.backends.colony_counter.sim import ColonySim


def create_sim(n_colonies=200, plate_type="spread", seed=42, distribution="random", staining=None, blue_fraction=0.3, colony_size_range=(3, 15), satellite_fraction=0.0, dynamic=False, growth_rate=0.1, max_colony_radius=20.0) -> ColonySim:
    """Create a colony counter simulation."""
    return ColonySim(
        n_colonies=n_colonies,
        plate_type=plate_type,
        distribution=distribution,
        staining=staining,
        blue_fraction=blue_fraction,
        zone_discs=None,
        colony_size_range=colony_size_range,
        satellite_fraction=satellite_fraction,
        dynamic=dynamic,
        growth_rate=growth_rate,
        max_colony_radius=max_colony_radius,
        seed=seed,
    )


def setup_colony_counter(n_colonies=200, plate_type="spread", seed=42, **kwargs):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope._init_standard import load_standalone
    from virtual_microscope.devices.state import GenericStateDevice

    sim = create_sim(n_colonies=n_colonies, plate_type=plate_type, seed=seed, **kwargs)
    labels = {0: "transmitted", 1: "blue-filter", 2: "GFP-excitation"}
    core = load_standalone(sim,
        channels={
            "plate-image": ("Channel", "Label", "transmitted"),
            "blue-channel": ("Channel", "Label", "blue-filter"),
            "gfp-channel": ("Channel", "Label", "GFP-excitation"),
        },
        extra_devices={"Channel": GenericStateDevice("Channel", labels)},
    )
    return core, sim
