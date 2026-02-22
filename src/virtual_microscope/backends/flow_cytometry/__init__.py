"""flow_cytometry backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a flow cytometer producing scatter and fluorescence dot-plot images. Generates multi-population event data with FITC and PE channels.",
    "channels": ["scatter", "FITC", "PE"],
    "continuous": False,
    "extra_devices": [],
}

from virtual_microscope.backends.flow_cytometry.sim import FlowCytometrySim


def create_sim(n_total=10000, events_per_snap=100, seed=42) -> FlowCytometrySim:
    """Create a flow cytometry simulation."""
    return FlowCytometrySim(n_total=n_total, events_per_snap=events_per_snap, seed=seed)


def setup_flow_cytometry(n_total=10000, events_per_snap=100, seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope._init_standard import load_standalone
    from virtual_microscope.devices.state import GenericStateDevice, ObjectiveDevice

    sim = create_sim(n_total=n_total, events_per_snap=events_per_snap, seed=seed)
    core = load_standalone(sim,
        channels={
            "scatter": ("Detector", "Label", "FSC-SSC"),
            "FITC": ("Detector", "Label", "FL1-FITC"),
            "PE": ("Detector", "Label", "FL2-PE"),
        },
        extra_devices={
            "Objective": ObjectiveDevice(),
            "Detector": GenericStateDevice("Detector", {0: "FSC-SSC", 1: "FL1-FITC", 2: "FL2-PE"}),
        },
    )
    return core, sim
