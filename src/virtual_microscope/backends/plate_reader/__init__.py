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
    """Programmatic setup — .cfg is single source of truth for devices/channels.

    Channel configs are added dynamically after loading the .cfg because
    the available wavelengths depend on the assay preset.
    """
    from pathlib import Path
    from virtual_microscope._init_standard import load_cfg
    from virtual_microscope.backends.plate_reader.sim import ASSAY_PRESETS

    sim = create_sim(assay_type=assay_type, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "plate_reader.cfg")

    # Add dynamic channel configs based on assay preset
    preset = ASSAY_PRESETS.get(assay_type, ASSAY_PRESETS["viability"])
    wl_pri = preset["wavelength_primary"]
    wl_ref = preset.get("wavelength_reference")

    core.defineConfigGroup("Channel")
    pri_label = f"{wl_pri}nm" if wl_pri else "luminescence"
    core.defineConfig("Channel", pri_label, "Channel", "Label", pri_label)
    if wl_ref:
        ref_label = f"{wl_ref}nm-ref"
        core.defineConfig("Channel", ref_label, "Channel", "Label", ref_label)
    core.setConfig("Channel", pri_label)

    return core, sim
