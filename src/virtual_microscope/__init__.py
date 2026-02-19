"""virtual_microscope — Simulation backends for pymmcore-plus.

Load via programmatic API:
    from virtual_microscope import setup_bacteria_microscope
    core, sim = setup_bacteria_microscope(n_cells=50, seed=42)

Or discover and load dynamically:
    from virtual_microscope import load_backend, list_backends
    print(list_backends())
    core, sim = load_backend("bacteria", n_cells=50)

Or via .cfg file (requires pymmcore-plus >= 0.17.0):
    core.loadSystemConfiguration("/path/to/bacteria/bacteria.cfg")
"""

# ── Core infrastructure ──────────────────────────────────────────────────────
from virtual_microscope.simulation_bridge import SimulationBridge, GLOBAL_BRIDGE
from virtual_microscope.realtime import RealtimeEngine
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
    # Most follow: setup_<backend>_microscope → backends.<backend>
    # Special aliases: setup_plate_reader → backends.plate_reader
    #                  setup_gel_doc → backends.gel_doc
    #                  setup_colony_counter → backends.colony_counter
    #                  setup_hemocytometer → backends.hemocytometer
    fn_to_mod = {
        "setup_particle_microscope": "particle",
        "setup_voronoi_microscope": "voronoi",
        "setup_bacteria_microscope": "bacteria",
        "setup_celegans_microscope": "celegans",
        "setup_rd_microscope": "reaction_diffusion",
        "setup_reaction_diffusion_microscope": "reaction_diffusion",
        "setup_fibroblast_microscope": "fibroblast",
        "setup_mito_microscope": "mito",
        "setup_calcium_microscope": "calcium",
        "setup_cardio_microscope": "cardio",
        "setup_blood_smear_microscope": "blood_smear",
        "setup_yeast_microscope": "yeast",
        "setup_histology_microscope": "histology",
        "setup_flow_cytometry_microscope": "flow_cytometry",
        "setup_spheroid_microscope": "spheroid",
        "setup_organoid_microscope": "organoid",
        "setup_zebrafish_microscope": "zebrafish",
        "setup_neuron_microscope": "neuron",
        "setup_microfluidics_microscope": "microfluidics",
        "setup_volvox_microscope": "volvox",
        "setup_malaria_microscope": "malaria",
        "setup_plant_cell_microscope": "plant_cell",
        "setup_plate_reader": "plate_reader",
        "setup_gel_doc": "gel_doc",
        "setup_colony_counter": "colony_counter",
        "setup_hemocytometer": "hemocytometer",
        "setup_lysosome_microscope": "lysosome",
        "setup_lipid_droplet_microscope": "lipid_droplet",
        "setup_stress_granule_microscope": "stress_granule",
        "setup_viability_microscope": "viability",
        "setup_wound_healing_microscope": "wound_healing",
        "setup_dictyostelium_microscope": "dictyostelium",
        "setup_fucci_microscope": "fucci",
        "setup_spt_microscope": "spt",
        "setup_fish_microscope": "fish",
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
    "setup_particle_microscope",
    "setup_voronoi_microscope",
    "setup_bacteria_microscope",
    "setup_celegans_microscope",
    "setup_rd_microscope",
    "setup_fibroblast_microscope",
    "setup_mito_microscope",
    "setup_calcium_microscope",
    "setup_cardio_microscope",
    "setup_blood_smear_microscope",
    "setup_yeast_microscope",
    "setup_histology_microscope",
    "setup_flow_cytometry_microscope",
    "setup_spheroid_microscope",
    "setup_organoid_microscope",
    "setup_zebrafish_microscope",
    "setup_neuron_microscope",
    "setup_microfluidics_microscope",
    "setup_volvox_microscope",
    "setup_malaria_microscope",
    "setup_plant_cell_microscope",
    "setup_plate_reader",
    "setup_gel_doc",
    "setup_colony_counter",
    "setup_hemocytometer",
    "setup_lysosome_microscope",
    "setup_lipid_droplet_microscope",
    "setup_stress_granule_microscope",
    "setup_viability_microscope",
    "setup_wound_healing_microscope",
    "setup_dictyostelium_microscope",
    "setup_fucci_microscope",
    "setup_spt_microscope",
    "setup_fish_microscope",
]
