"""gel_doc backend for virtual-microscope."""

BACKEND_INFO = {
    "description": "Simulates a gel electrophoresis documentation system for western or DNA gels. Renders multi-lane band patterns with configurable lane count and gel type.",
    "channels": ["gel-image"],
    "continuous": False,
    "extra_devices": [],
    "specimen": "Electrophoresis gel (western, DNA agarose, or Coomassie)",
    "modality": "Gel documentation (transillumination / chemiluminescence)",
    "experiment_guide": (
        "A gel electrophoresis documentation image. View multi-lane band patterns "
        "from western blots, DNA agarose gels, or Coomassie-stained protein gels. "
        "Navigate the gel with XY stage and zoom with objectives. Useful for "
        "training lane-detection and band-quantification pipelines."
    ),
    "device_effects": {},
    "key_parameters": {
        "n_lanes": "Number of gel lanes (default 8)",
        "gel_type": "Gel type: 'western', 'dna', or 'coomassie' (default 'western')",
    },
}

from virtual_microscope.backends.gel_doc.sim import GelDocSim


def create_sim(n_lanes=8, gel_type="western", seed=42) -> GelDocSim:
    """Create a gel doc simulation."""
    return GelDocSim(n_lanes=n_lanes, gel_type=gel_type, seed=seed)


def setup_gel_doc(n_lanes=8, gel_type="western", seed=42):
    """Programmatic setup — .cfg is single source of truth for devices/channels."""
    from pathlib import Path
    from virtual_microscope._init_standard import load_cfg

    sim = create_sim(n_lanes=n_lanes, gel_type=gel_type, seed=seed)
    core = load_cfg(sim, Path(__file__).parent / "gel_doc.cfg")
    return core, sim
