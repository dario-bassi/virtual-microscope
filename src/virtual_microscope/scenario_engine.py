"""
Scenario Engine — Config-driven scenario creation.

Turns a declarative config (dict or JSON file) into a ready-to-use
simulation with ground truth. Eliminates the ~100-line boilerplate
in each challenge_XXX.py script.

Supports:
  - Voronoi tissue backend (static or dynamic)
  - N-channel marker system
  - Per-channel optical pipeline overrides
  - Automatic ground truth extraction
  - Dynamics (wound healing, migration, division)

Usage:
    from scenario_engine import ScenarioEngine

    # From dict config
    engine = ScenarioEngine({
        "tissue": {"n_cells": 30, "width": 512, "height": 512, "seed": 42},
        "markers": {"nucleus": {"fraction": 0.7}, "membrane": {"fraction": 1.0}},
    })
    gt, config, core = engine.build()

    # From JSON file
    engine = ScenarioEngine.from_file("scenarios/wound_healing.json")
    gt, config, core = engine.build()

    # Post as challenge
    engine.post(title="...", description="...", objectives=[...], ...)
"""

import json
import numpy as np
from pathlib import Path

# Lazy imports to avoid import-time failures
_VENV = Path(__file__).resolve().parent.parent
_LOGS = _VENV.parent / "logs"


def _ensure_paths():
    """Ensure import paths are set up."""
    import sys
    for p in [str(_VENV), str(_VENV / "src"), str(_LOGS)]:
        if p not in sys.path:
            sys.path.insert(0, p)


class ScenarioEngine:
    """Builds a simulation from a declarative config."""

    def __init__(self, config: dict):
        """
        Config schema:
        {
            "tissue": {
                "backend": "voronoi",       # "voronoi" (default) or "dynamic"
                "n_cells": 60,
                "width": 512,
                "height": 512,
                "seed": 42,
                "jitter": 0.7,
                "nucleus_fraction": 0.3,
                "viewport_width": 512,      # optional, defaults to 512
                "viewport_height": 512,
            },
            "markers": {
                "nucleus": {
                    "fraction": 0.7,
                    "pattern": "random",
                    "intensity": {"mode": "uniform", "range": [0.5, 1.0]},
                },
                "membrane": {
                    "fraction": 1.0,
                    "pattern": "all",
                    "intensity": {"mode": "uniform", "range": [0.6, 0.9]},
                },
            },
            "optics": {                     # optional per-channel overrides
                "bf": {"psf_sigma": 0, "noise": {"photon_scale": 1.0, "read_std": 8.0},
                       "vignette": 0.35},
                "nucleus": {"psf_sigma": 2.5, "noise": {"read_std": 6.0}},
                "membrane": {"psf_sigma": 1.5},
            },
            "dynamics": {                   # optional — uses DynamicVoronoiSim
                "wound": {"shape": "rectangle", "center": [512, 512], "size": [200, 1024]},
                "migration_speed": 2.0,
                "random_motility": 0.5,
                "division_rate": 0.0,
                "apoptosis_rate": 0.0,
                "n_steps": 10,
                "dt": 1.0,
            },
            "stage": {                      # optional
                "x": 256, "y": 256,
                "channel": "brightfield",   # initial channel
            },
            "cell_overrides": [             # optional per-cell modifications
                {"indices": [0, 3, 7], "has_membrane_marker": false,
                 "nucleus_intensity": [0.9, 0.95, 0.92]},
            ],
        }
        """
        self.config = config
        self.sim = None
        self.core = None
        self._ground_truth = None

    @classmethod
    def from_file(cls, path: str) -> "ScenarioEngine":
        """Load config from a JSON file."""
        with open(path) as f:
            config = json.load(f)
        return cls(config)

    def build(self) -> tuple:
        """Build the simulation.

        Returns:
            (ground_truth: dict, config: dict, core: UniMMCore)
        """
        _ensure_paths()
        from pymmcore_plus.experimental.unicore import UniMMCore
        from virtual_microscope.simulation_bridge import SimulationBridge
        from setup_microscope import _set_global_bridge, _init_devices

        tissue = self.config.get("tissue", {})
        backend = tissue.get("backend", "voronoi")
        seed = tissue.get("seed", 42)
        n_cells = tissue.get("n_cells", 60)
        width = tissue.get("width", 512)
        height = tissue.get("height", 512)
        jitter = tissue.get("jitter", 0.7)
        nuc_frac = tissue.get("nucleus_fraction", 0.3)
        vp_w = tissue.get("viewport_width", 512)
        vp_h = tissue.get("viewport_height", 512)
        textured = tissue.get("textured_nuclei", False)
        ruffle = tissue.get("membrane_ruffle", 0.0)

        # Create sim
        dynamics = self.config.get("dynamics")
        if backend == "dynamic" or dynamics:
            from tissue_dynamics import DynamicVoronoiSim
            dyn_cfg = dynamics or {}
            self.sim = DynamicVoronoiSim(
                width=width, height=height, nb_cells=n_cells,
                viewport_width=vp_w, viewport_height=vp_h,
                rng_seed=seed, jitter=jitter, nucleus_fraction=nuc_frac,
                textured_nuclei=textured, membrane_ruffle=ruffle,
                migration_speed=dyn_cfg.get("migration_speed", 2.0),
                random_motility=dyn_cfg.get("random_motility", 0.5),
                division_rate=dyn_cfg.get("division_rate", 0.0),
                apoptosis_rate=dyn_cfg.get("apoptosis_rate", 0.0),
            )
        else:
            from voronoi_sim import VoronoiSim
            self.sim = VoronoiSim(
                width=width, height=height, nb_cells=n_cells,
                viewport_width=vp_w, viewport_height=vp_h,
                rng_seed=seed, jitter=jitter, nucleus_fraction=nuc_frac,
                textured_nuclei=textured, membrane_ruffle=ruffle,
            )

        # Set tissue Z position (for Z-stack / defocus challenges)
        tissue_z = tissue.get("tissue_z")
        if tissue_z is not None:
            self.sim.tissue_z = float(tissue_z)

        # Apply markers
        marker_cfg = self.config.get("markers")
        if marker_cfg:
            rng = np.random.default_rng(seed)
            self.sim.apply_marker_config(marker_cfg, rng)

        # Apply cell overrides
        overrides = self.config.get("cell_overrides", [])
        for ov in overrides:
            indices = ov.get("indices", [])
            for key in ("has_nucleus_marker", "has_membrane_marker"):
                if key in ov:
                    val = ov[key]
                    for idx in indices:
                        getattr(self.sim, key)[idx] = val
            if "nucleus_intensity" in ov:
                vals = ov["nucleus_intensity"]
                for i, idx in enumerate(indices):
                    self.sim.nucleus_intensity[idx] = vals[i] if isinstance(vals, list) else vals
            if "membrane_intensity" in ov:
                vals = ov["membrane_intensity"]
                for i, idx in enumerate(indices):
                    self.sim.membrane_intensity[idx] = vals[i] if isinstance(vals, list) else vals

        # Apply optical pipeline overrides
        optics = self.config.get("optics")
        if optics:
            from optical_pipeline import OpticalPipeline
            channel_map = {"bf": "_bf_pipeline", "nucleus": "_nuc_pipeline",
                           "membrane": "_mem_pipeline"}
            for ch_name, cfg in optics.items():
                attr = channel_map.get(ch_name)
                if attr:
                    pipeline = OpticalPipeline(
                        psf_sigma=cfg.get("psf_sigma", 0),
                        noise=cfg.get("noise"),
                        vignette=cfg.get("vignette", 0),
                        illumination_unevenness=cfg.get("illumination_unevenness", 0),
                        photobleach_rate=cfg.get("photobleach_rate", 0),
                        focus_drift_rate=cfg.get("focus_drift_rate", 0),
                        debris=cfg.get("debris"),
                        rng_seed=cfg.get("rng_seed", seed + hash(ch_name) % 10000),
                    )
                    setattr(self.sim, attr, pipeline)

        # Apply spectral crosstalk
        crosstalk = self.config.get("crosstalk")
        if crosstalk:
            self.sim.crosstalk = crosstalk

        # Re-render after overrides
        if overrides or optics or crosstalk:
            self.sim._bf_full = None
            self.sim._nuc_full = None
            self.sim._mem_full = None
            self.sim._render_full_tissue()

        # Apply dynamics
        if dynamics:
            wound_cfg = dynamics.get("wound")
            if wound_cfg:
                self.sim.create_wound(
                    shape=wound_cfg.get("shape", "rectangle"),
                    center=tuple(wound_cfg.get("center", [width // 2, height // 2])),
                    size=tuple(wound_cfg.get("size", [200, height])),
                )
            n_steps = dynamics.get("n_steps", 0)
            dt = dynamics.get("dt", 1.0)
            for _ in range(n_steps):
                self.sim.step(dt)

        # Set up bridge and core
        self.core = UniMMCore()
        _set_global_bridge(SimulationBridge(self.sim))
        _init_devices(self.core)

        # Stage position
        stage = self.config.get("stage", {})
        sx = stage.get("x", width / 2)
        sy = stage.get("y", height / 2)
        self.core.setXYPosition(sx, sy)
        channel = stage.get("channel", "brightfield")
        self.core.setConfig("Fake", channel)

        # Compute ground truth
        self._ground_truth = self._compute_ground_truth()

        # Build output config
        out_config = {
            "backend": backend if not dynamics else "dynamic_voronoi",
            "n_cells": n_cells,
            "world_size": [width, height],
            "rng_seed": seed,
        }
        if optics:
            out_config["custom_optics"] = True
        if dynamics:
            out_config["dynamics"] = True

        return self._ground_truth, out_config, self.core

    def _compute_ground_truth(self) -> dict:
        """Extract standard ground truth from the simulation."""
        sim = self.sim
        gt = sim.get_ground_truth()

        tissue = self.config.get("tissue", {})
        result = {
            "total_cells": gt["n_cells"],
            "n_nucleus_positive": gt["n_nucleus_positive"],
            "n_membrane_positive": gt["n_membrane_positive"],
            "mean_area": gt["mean_area"],
            "backend": tissue.get("backend", "voronoi"),
            "world_size": [tissue.get("width", 512), tissue.get("height", 512)],
        }

        # Per-cell data
        cells = []
        for i in range(sim.nb_cells):
            cell = {
                "idx": i,
                "centroid_x": round(float(sim.cell_centroids[i][0]), 1),
                "centroid_y": round(float(sim.cell_centroids[i][1]), 1),
                "area": round(float(sim.cell_areas[i]), 1),
                "has_nucleus": bool(sim.has_nucleus_marker[i]),
                "has_membrane": bool(sim.has_membrane_marker[i]),
            }
            cells.append(cell)
        result["cells"] = cells

        # Dynamics-specific GT
        dynamics = self.config.get("dynamics")
        if dynamics and hasattr(sim, "get_wound_area"):
            result["wound_area"] = round(sim.get_wound_area(), 1)
            result["migration_front"] = sim.get_migration_front()
            result["n_steps"] = dynamics.get("n_steps", 0)

        return result

    def post(self, title: str, description: str, objectives: list,
             grading_criteria: list, difficulty: str = "intermediate",
             extra_gt: dict | None = None) -> int:
        """Build and post as a challenge.

        Args:
            title: Challenge title
            description: Full challenge description
            objectives: List of objective strings
            grading_criteria: List of grading criteria strings
            difficulty: "beginner", "intermediate", "advanced"
            extra_gt: Additional ground truth fields to merge

        Returns:
            Challenge ID
        """
        _ensure_paths()
        from comms.messaging import create_challenge

        if self._ground_truth is None:
            self.build()

        gt = self._ground_truth.copy()
        if extra_gt:
            gt.update(extra_gt)

        cid, cdir = create_challenge(
            title=title,
            description=description,
            objectives=objectives,
            grading_criteria=grading_criteria,
            ground_truth=gt,
            difficulty=difficulty,
            config={
                "backend": self.config.get("tissue", {}).get("backend", "voronoi"),
                "n_cells": self.config.get("tissue", {}).get("n_cells", 60),
                "world_size": [
                    self.config.get("tissue", {}).get("width", 512),
                    self.config.get("tissue", {}).get("height", 512),
                ],
                "rng_seed": self.config.get("tissue", {}).get("seed", 42),
            },
        )
        return cid
