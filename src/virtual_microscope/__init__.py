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

import importlib

# ── Core infrastructure ──────────────────────────────────────────────────────
from virtual_microscope.engine.simulation_bridge import SimulationBridge, GLOBAL_BRIDGE
from virtual_microscope.engine.realtime import RealtimeEngine
from virtual_microscope.backends import load_backend, list_backends, describe_backend, describe_backends

_STATIC_NAMES = [
    "SimulationBridge",
    "GLOBAL_BRIDGE",
    "RealtimeEngine",
    "load_backend",
    "list_backends",
    "describe_backend",
    "describe_backends",
]

# ── Backend setup functions ──────────────────────────────────────────────────
# Auto-discovered: any backend directory with setup_<name>() is accessible
# as ``from virtual_microscope import setup_<name>``.


def __getattr__(name: str):
    """Lazy auto-discovery of setup_* functions from backend modules."""
    if not name.startswith("setup_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    backend_name = name[len("setup_"):]  # "setup_bacteria" → "bacteria"
    try:
        mod = importlib.import_module(f"virtual_microscope.backends.{backend_name}")
    except ModuleNotFoundError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    fn = getattr(mod, name, None)
    if fn is None:
        raise AttributeError(
            f"Backend {backend_name!r} has no function {name!r}"
        )
    # Cache in module globals for subsequent accesses
    globals()[name] = fn
    return fn


def __dir__():
    """Include auto-discovered setup_* functions in dir()."""
    return [*_STATIC_NAMES, *(f"setup_{b}" for b in list_backends())]
