# Virtual Microscope

> **Note:** This project is in active development. APIs and backends may change without notice.

A fully simulated microscope platform that generates realistic microscopy images — no hardware required. Each backend simulates a different biological specimen with brightfield, fluorescence, and specialty imaging modes, all compatible with the [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) device interface. Samples can be perturbed via photo-activation (SLM), temperature shifts, and drug perfusion, making the platform suitable for testing and training closed-loop feedback workflows. See the [feedback control notebook](examples/03_feedback_control.ipynb) for a complete example.

## Gallery
**[See the full gallery and description of all 35 backends →](docs/gallery.md)**

<table>
  <tr>
    <td align="center" width="33%"><img src="docs/gallery/frames/fucci_04.png" width="100%"><br><b>fucci</b></td>
    <td align="center" width="33%"><img src="docs/gallery/frames/histology_03.png" width="100%"><br><b>histology</b></td>
    <td align="center" width="33%"><img src="docs/gallery/frames/lipid_droplet_04.png" width="100%"><br><b>lipid_droplet</b></td>
  </tr>
  <tr>
    <td align="center" width="33%"><img src="docs/gallery/frames/volvox_01.png" width="100%"><br><b>volvox</b></td>
    <td align="center" width="33%"><img src="docs/gallery/frames/neuron_04.png" width="100%"><br><b>neuron</b></td>
    <td align="center" width="33%"><img src="docs/gallery/frames/plant_cell_01.png" width="100%"><br><b>plant_cell</b></td>
  </tr>
</table>

## Installation

```bash
pip install -e .
```

## Quick Start

### Via pymmcore-plus device API

```python
from virtual_microscope.backends import load_backend

core, sim = load_backend("bacteria", n_cells=50, seed=42)

# Use like a real microscope
core.setXYPosition(300, 400)
core.setConfig("Channel", "phase-contrast")
img = core.snap()  # numpy array
```

### Direct sim access (no device layer)

```python
from virtual_microscope.backends.bacteria import create_sim

sim = create_sim(n_cells=30, seed=42)
sim.mode = 0  # 0=phase-contrast, 1=GFP, 2=DAPI
img = sim.snap_frame(exposure=50.0, intensity=1.0)

# Dynamic sims advance their state each step
sim.step(dt=1.0)
```

### Core components

| Component | Description |
|-----------|-------------|
| **`SimBase`** | ABC providing shared microscope state and rendering pipeline — objectives, FOV cropping, defocus blur, exposure model, photobleaching, z-drift. All microscope-style backends inherit from it. |
| **`SimulationBridge`** | Adapter between pymmcore device calls and the sim. Manages stage, focus, SLM masks, snap dispatch. Supports factory-based simulation recreation. |
| **`RealtimeEngine`** | Background thread driving `sim.step(dt)` at configurable tick rate. Thread-safe snap via mutex. Auto-starts for sims with `continuous = True`. Supports idle timeout and wake-on-snap. |
| **`SLMProcessor`** | Centralized SLM mask handling. Maps viewport masks to world coordinates, maintains an exponential-decay stimulation field, supports spatial-filter illumination mode. |
| **`OpticalPipeline`** | Per-channel post-processing: PSF convolution, Poisson-Gaussian noise, vignetting, photobleaching, debris overlay. |
| **`NewExperimentDevice`** | pymmcore device that triggers soft reset (`sim.reset()`) or full recreation via factory. |
| **`load_cfg()`** | Wires sim → bridge → core, loads `.cfg` device config, auto-starts `RealtimeEngine` for continuous sims. |


### How a snap works

1. `core.snap()` calls `SimulationBridge.snap(exposure, brightness)`
2. Bridge passes SLM mask + device state to `sim.snap_frame()`
3. Sim renders at `internal_scale x world_size` (typically 4x)
4. `SimBase._crop_fov()` extracts the viewport region for the current objective and resizes to 512×512
5. `SimBase._apply_defocus()` applies Gaussian blur based on focal distance
6. `SimBase._apply_exposure()` scales brightness for BF vs fluorescence
7. `OpticalPipeline.apply()` adds noise, PSF, vignetting, photobleaching
8. Result is returned as `(512, 512)` uint8 (or `(512, 512, 3)` for stain-based backends like blood_smear)

### Real-time simulation

Backends with `continuous = True` automatically get a `RealtimeEngine` that advances the simulation in a background thread between snaps:

```python
core, sim = load_backend("bacteria")
# Engine is already running — sim.step(dt) called ~10x/sec in the background

import time
time.sleep(5)        # bacteria grow and divide for 5 seconds

img = core.snap()    # snapshot of the current state (thread-safe)
```

The engine supports:
- **Time scaling**: `bridge._engine.time_scale = 5.0` for 5x speed
- **Idle timeout**: auto-pauses after 30s without snaps, resumes on next snap
- **Pause/resume**: `bridge._engine.pause()` / `bridge._engine.resume()`

### SLM stimulation

```python
import numpy as np

core, sim = load_backend("calcium")
mask = np.zeros((512, 512), dtype=np.uint8)
mask[200:300, 200:300] = 1             # illuminate a square region
core.setSLMImage("SLM", mask)          # triggers calcium wave at target

# The stimulation field decays exponentially over time
```

### Simulation reset

```python
# Soft reset — same parameters, new random state
core.setProperty("NewExperiment", "Seed", "42")
core.setProperty("NewExperiment", "Action", "Reset")

# Hard reset — recreate from factory with fresh parameters
core.setProperty("NewExperiment", "Action", "New")
```

## Backends

Each of the 35 backends embeds rich documentation in its `BACKEND_INFO` dict (specimen, modality, experiment guide, device effects, key parameters). Access it programmatically:

```python
from virtual_microscope.backends import describe_backends

info = describe_backends()["bacteria"]
print(info["experiment_guide"])  # multi-sentence experiment walkthrough
print(info["device_effects"])    # {"SLM": "...", "Temperature": "..."}
print(info["key_parameters"])    # {"n_cells": "Initial cell count (default 30)", ...}
```

## Adding a Backend

1. **Create `sim.py`** inheriting from `SimBase`:

```python
from virtual_microscope.base import SimBase

class MySim(SimBase):
    continuous = True  # set True if the sim has dynamics

    def __init__(self, n_cells=50, seed=42, **kwargs):
        super().__init__(
            width=512, height=512, seed=seed,
            internal_scale=4,
            mode_map={
                ("TagGFP2(483/506)", "GREEN"): 1,
                ("SCFP2(434/474)", "UV"): 2,
            },
        )
        # ... your specimen-specific state ...

    def _render_for_mode(self, mode):
        # render your specimen into a (self._ih, self._iw) buffer
        # mode is the integer channel index from mode_map
        # return a numpy array (grayscale or RGB)
        return frame

    def step(self, dt=1.0):
        # advance dynamics (called by RealtimeEngine if continuous=True)
        ...
```

2. **Create `__init__.py`** with `BACKEND_INFO` and setup function:

```python
BACKEND_INFO = {
    "description": "One-line summary of the backend.",
    "channels": ["brightfield", "GFP", "DAPI"],
    "continuous": True,
    "extra_devices": ["SLM", "Temperature"],
    "specimen": "Cell type or sample description",
    "modality": "Imaging modality (e.g. Phase-contrast + epifluorescence)",
    "experiment_guide": "Multi-sentence guide explaining what the user can do.",
    "device_effects": {
        "SLM": "What the SLM does in this backend",
        "Temperature": "What temperature control does",
    },
    "key_parameters": {
        "n_cells": "Initial cell count (default 50)",
    },
}

from pathlib import Path
from virtual_microscope._init_standard import load_cfg

def create_sim(**kwargs):
    from .sim import MySim
    return MySim(**kwargs)

def setup_my_backend(**kwargs):
    sim = create_sim(**kwargs)
    core = load_cfg(sim, Path(__file__).parent / "my_backend.cfg")
    return core, sim
```

3. **Create `my_backend.cfg`** with device and channel definitions (see any existing backend for the template).

4. **Create `showcase.py`** with `create_showcase_images(seed=0)` returning a list of `(512, 512, 3)` uint8 RGB images.

5. **Verify**: `python -c "from virtual_microscope.backends import load_backend; core, sim = load_backend('my_backend'); print(core.snap().shape)"`

## Project Structure

```
src/virtual_microscope/
├── base/
│   └── sim_base.py              # SimBase ABC: shared microscope state & pipeline
├── sims/
│   ├── voronoi/
│   │   ├── voronoi.py           # VoronoiSim: Voronoi tessellation tissue
│   │   └── tissue_dynamics.py   # DynamicVoronoiSim: migration, division, wound healing, hooks
│   └── cell/
│       ├── sim.py               # ScatteredCellSim (particle backend)
│       ├── renderer.py          # Cell cycle rendering (brightfield, nucleus, membrane)
│       ├── chromatin.py         # Chromatin/chromosome drawing functions
│       ├── apoptosis.py         # Apoptosis phase rendering
│       ├── cycle_manager.py     # Division/death population dynamics
│       ├── cycle.py             # CellCycleNormal: cell cycle state machine
│       ├── cell.py              # CellBase: core physics (Numba-accelerated)
│       ├── normal.py            # NormalCell: default cell type
│       ├── drug.py              # DrugResponseCell
│       ├── optogenetic.py       # OptogeneticCell
│       └── spatial_grid.py      # Spatial indexing for collision detection
├── engine/
│   ├── simulation_bridge.py     # SimulationBridge: sim ↔ device adapter
│   ├── realtime.py              # RealtimeEngine: background sim thread
│   ├── slm_processor.py         # SLMProcessor: mask mapping & decay field
│   └── multi_field_sim.py       # MultiFieldBridge: multi-position experiments
├── pipeline/
│   ├── optical_pipeline.py      # OpticalPipeline: noise, PSF, vignetting
│   ├── debris_overlay.py        # Debris/dirt overlay effects
│   └── nuclear_texture.py       # Textured nuclear rendering
├── backends/
│   ├── __init__.py              # load_backend(), list_backends(), describe_backends()
│   ├── bacteria/                # one directory per backend (35 total)
│   │   ├── __init__.py          #   BACKEND_INFO + create_sim() + setup_bacteria()
│   │   ├── sim.py               #   BacteriaSim(SimBase)
│   │   ├── bacteria.cfg         #   device & channel config
│   │   └── showcase.py          #   gallery image generator
│   └── .../                     # 34 more backends
├── devices/
│   ├── sim_server.py            # SimServer: main pymmcore device adapter
│   ├── new_experiment.py        # NewExperimentDevice: reset/recreate
│   ├── camera.py                # Camera device adapter
│   ├── shutter.py               # Shutter device adapter
│   ├── slm.py                   # SLM device adapter
│   ├── stage.py                 # XY stage device adapter
│   ├── z_stage.py               # Z (focus) stage device adapter
│   └── state.py                 # State device (objective, temperature, etc.)
├── _init_standard.py            # load_cfg() — wires everything together
└── _showcase_utils.py           # Shared helpers for showcase image generation
```
