"""virtual_microscope — Simulation backends for pymmcore-plus.

Load via programmatic API:
    from virtual_microscope import setup_bacteria
    core, sim = setup_bacteria(n_cells=50, seed=42)

Or discover and load dynamically:
    from virtual_microscope import load_backend, list_backends
    print(list_backends())
    core, sim = load_backend("bacteria", n_cells=50)

Or via .cfg file (requires pymmcore-plus >= 0.17.0):
    core.loadSystemConfiguration("/path/to/bacteria/bacteria.cfg")
"""

# ── Core infrastructure ──────────────────────────────────────────────────────
from virtual_microscope.engine.simulation_bridge import SimulationBridge, GLOBAL_BRIDGE
from virtual_microscope.engine.realtime import RealtimeEngine
from virtual_microscope.backends import load_backend, list_backends

# ── Backend setup functions ──────────────────────────────────────────────────
# Import each backend's setup function lazily on first access.
# These are the main entry points for programmatic use.

def __getattr__(name: str):
    """Lazy import of setup_* functions from backend modules."""
    if not name.startswith("setup_"):
        raise AttributeError(f"module 'virtual_microscope' has no attribute {name!r}")

    import importlib

    # Map function name to backend module
    fn_to_mod = {
        "setup_particle": "particle",
        "setup_voronoi": "voronoi",
        "setup_bacteria": "bacteria",
        "setup_celegans": "celegans",
        "setup_reaction_diffusion": "reaction_diffusion",
        "setup_fibroblast": "fibroblast",
        "setup_mito": "mito",
        "setup_calcium": "calcium",
        "setup_cardio": "cardio",
        "setup_blood_smear": "blood_smear",
        "setup_yeast": "yeast",
        "setup_histology": "histology",
        "setup_flow_cytometry": "flow_cytometry",
        "setup_spheroid": "spheroid",
        "setup_organoid": "organoid",
        "setup_zebrafish": "zebrafish",
        "setup_neuron": "neuron",
        "setup_microfluidics": "microfluidics",
        "setup_volvox": "volvox",
        "setup_malaria": "malaria",
        "setup_plant_cell": "plant_cell",
        "setup_plate_reader": "plate_reader",
        "setup_gel_doc": "gel_doc",
        "setup_colony_counter": "colony_counter",
        "setup_hemocytometer": "hemocytometer",
        "setup_lysosome": "lysosome",
        "setup_lipid_droplet": "lipid_droplet",
        "setup_stress_granule": "stress_granule",
        "setup_viability": "viability",
        "setup_wound_healing": "wound_healing",
        "setup_dictyostelium": "dictyostelium",
        "setup_fucci": "fucci",
        "setup_spt": "spt",
        "setup_fish": "fish",
        "setup_optogenetic": "optogenetic",
    }

    mod_name = fn_to_mod.get(name)
    if mod_name is None:
        raise AttributeError(f"module 'virtual_microscope' has no attribute {name!r}")

    mod = importlib.import_module(f"virtual_microscope.backends.{mod_name}")
    fn = getattr(mod, name, None)
    if fn is None:
        raise AttributeError(
            f"Backend 'virtual_microscope.backends.{mod_name}' "
            f"has no function {name!r}"
        )
    # Cache in module globals for subsequent accesses
    globals()[name] = fn
    return fn


__all__ = [
    # Infrastructure
    "SimulationBridge",
    "GLOBAL_BRIDGE",
    "RealtimeEngine",
    "load_backend",
    "list_backends",
    # Setup functions (all lazily imported)
    "setup_particle",
    "setup_voronoi",
    "setup_bacteria",
    "setup_celegans",
    "setup_reaction_diffusion",
    "setup_fibroblast",
    "setup_mito",
    "setup_calcium",
    "setup_cardio",
    "setup_blood_smear",
    "setup_yeast",
    "setup_histology",
    "setup_flow_cytometry",
    "setup_spheroid",
    "setup_organoid",
    "setup_zebrafish",
    "setup_neuron",
    "setup_microfluidics",
    "setup_volvox",
    "setup_malaria",
    "setup_plant_cell",
    "setup_plate_reader",
    "setup_gel_doc",
    "setup_colony_counter",
    "setup_hemocytometer",
    "setup_lysosome",
    "setup_lipid_droplet",
    "setup_stress_granule",
    "setup_viability",
    "setup_wound_healing",
    "setup_dictyostelium",
    "setup_fucci",
    "setup_spt",
    "setup_fish",
    "setup_optogenetic",
]
