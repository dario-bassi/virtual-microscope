"""
Bridge between Voronoi tissue backend and pymmcore-plus API.

This is now a thin wrapper around setup_microscope.setup_voronoi_microscope().
The VoronoiSim plugs directly into SimulationBridge — no monkey-patching.

Usage:
    core, tissue = setup_voronoi_microscope(n_cells=80, seed=42)
    core.setConfig("Channel", "brightfield")
    core.snapImage()
    bf_image = core.getImage()
"""

from virtual_microscope.backends.voronoi import setup_voronoi_microscope

__all__ = ["setup_voronoi_microscope"]
