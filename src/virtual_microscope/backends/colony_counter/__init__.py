"""colony_counter backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates bacterial colony plates for colony counting assays. Supports spread/streak plate types with optional blue-white screening and GFP.",
    "channels": ["plate-image", "blue-channel", "gfp-channel"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Bacterial colony plate (agar)",
    "modality": "Brightfield plate imaging",
    "experiment_guide": (
        "A bacterial colony plate viewed from above. Count colonies on spread or "
        "streak plates, with optional blue-white screening (lacZ) or GFP "
        "fluorescence. Useful for training colony-counting algorithms and "
        "studying plating efficiency, satellite colonies, and zone-of-inhibition "
        "assays."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_colonies": "Number of colonies (default 200)",
        "plate_type": "Plate type: 'spread' or 'streak' (default 'spread')",
        "staining": "Optional staining: 'blue-white', 'gfp', or None",
    },
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
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    from pathlib import Path
    from virtual_microscope._init_standard import load_cfg

    sim = create_sim(n_colonies=n_colonies, plate_type=plate_type, seed=seed, **kwargs)
    core = load_cfg(sim, Path(__file__).parent / "colony_counter.cfg")
    return core, sim
