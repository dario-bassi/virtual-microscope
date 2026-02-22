"""plate_reader backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a microplate reader producing well-based absorbance or fluorescence heatmaps. Channels vary by assay preset (viability, ELISA, Bradford, etc.).",
    "channels": [],
    "continuous": False,
    "extra_devices": [],
}

from virtual_microscope.backends.plate_reader.sim import PlateReaderSim


def create_sim(assay_type="viability", seed=42) -> PlateReaderSim:
    """Create a plate reader simulation."""
    return PlateReaderSim(assay_type=assay_type, seed=seed)


def setup_plate_reader(assay_type="viability", seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope._init_standard import load_standalone
    from virtual_microscope.devices.state import GenericStateDevice
    from virtual_microscope.backends.plate_reader.sim import ASSAY_PRESETS

    sim = create_sim(assay_type=assay_type, seed=seed)

    # Build channel labels + configs from assay preset
    preset = ASSAY_PRESETS.get(assay_type, ASSAY_PRESETS["viability"])
    wl_pri = preset["wavelength_primary"]
    wl_ref = preset.get("wavelength_reference")
    labels = {0: f"{wl_pri}nm" if wl_pri else "luminescence"}
    if wl_ref:
        labels[1] = f"{wl_ref}nm-ref"

    channels = {lbl: ("Channel", "Label", lbl) for lbl in labels.values()}
    core = load_standalone(sim,
        channels=channels,
        extra_devices={"Channel": GenericStateDevice("Channel", labels)},
    )
    return core, sim
