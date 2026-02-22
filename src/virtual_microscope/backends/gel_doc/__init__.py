"""gel_doc backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a gel electrophoresis documentation system for western or DNA gels. Renders multi-lane band patterns with configurable lane count and gel type.",
    "channels": ["gel-image"],
    "continuous": False,
    "extra_devices": [],
}

from virtual_microscope.backends.gel_doc.sim import GelDocSim


def create_sim(n_lanes=8, gel_type="western", seed=42) -> GelDocSim:
    """Create a gel doc simulation."""
    return GelDocSim(n_lanes=n_lanes, gel_type=gel_type, seed=seed)


def setup_gel_doc(n_lanes=8, gel_type="western", seed=42):
    """Programmatic setup (no .cfg needed)."""
    from virtual_microscope._init_standard import load_standalone
    from virtual_microscope.devices.state import ObjectiveDevice

    sim = create_sim(n_lanes=n_lanes, gel_type=gel_type, seed=seed)
    core = load_standalone(sim,
        channels={"gel-image": ("Objective", "Label", "10x")},
        extra_devices={"Objective": ObjectiveDevice()},
    )
    return core, sim
