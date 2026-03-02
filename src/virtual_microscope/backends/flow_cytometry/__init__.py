"""flow_cytometry backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a flow cytometer producing scatter and fluorescence dot-plot images. Generates multi-population event data with FITC and PE channels.",
    "channels": ["scatter", "FITC", "PE"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Cell suspension (multi-population)",
    "modality": "Flow cytometry scatter + fluorescence",
    "experiment_guide": (
        "Simulates a flow cytometer acquiring scatter and fluorescence events "
        "from a mixed cell population. Each snap adds a batch of events to the "
        "dot-plot image. Observe distinct populations in FSC/SSC scatter and "
        "FITC/PE fluorescence channels. Useful for training gating and clustering "
        "algorithms."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_total": "Total events to acquire (default 10000)",
        "events_per_snap": "Events added per snap (default 100)",
    },
}

from virtual_microscope.backends.flow_cytometry.sim import FlowCytometrySim


def create_sim(n_total=10000, events_per_snap=100, seed=42) -> FlowCytometrySim:
    """Create a flow cytometry simulation."""
    return FlowCytometrySim(n_total=n_total, events_per_snap=events_per_snap, seed=seed)


def setup_flow_cytometry(n_total=10000, events_per_snap=100, seed=42):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    from pathlib import Path
    from virtual_microscope._init_standard import load_cfg

    sim = create_sim(n_total=n_total, events_per_snap=events_per_snap, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "flow_cytometry.cfg")
    return core, sim
