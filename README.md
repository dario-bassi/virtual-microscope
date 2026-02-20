# Virtual Microscope

A fully simulated microscope platform that generates realistic microscopy images — no hardware required. Each backend simulates a different biological specimen with brightfield, fluorescence, and specialty imaging modes, all compatible with the [pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus) device interface.

## Gallery

| | | |
|:---:|:---:|:---:|
| ![bacteria](docs/gallery/bacteria.png) | ![neuron](docs/gallery/neuron.png) | ![calcium](docs/gallery/calcium.png) |
| **bacteria** | **neuron** | **calcium** |
| ![blood_smear](docs/gallery/blood_smear.png) | ![wound_healing](docs/gallery/wound_healing.png) | ![malaria](docs/gallery/malaria.png) |
| **blood_smear** | **wound_healing** | **malaria** |

See the [full gallery](docs/gallery/gallery.md) for all 34 backends.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from virtual_microscope.backends import load_backend

# Load a backend — returns (core, sim) compatible with pymmcore-plus
core, sim = load_backend("bacteria", n_cells=50, seed=42)

# Take a brightfield snapshot
core.setConfig("Fake", "brightfield")
core.snapImage()
img = core.getImage()  # numpy array (512, 512), uint8

# Switch to fluorescence
core.setConfig("Fake", "nucleus-channel")
core.snapImage()
fl_img = core.getImage()
```

Or use a backend directly without the device layer:

```python
from virtual_microscope.backends.bacteria import create_sim

sim = create_sim(n_cells=30, seed=42)
sim.mode = 0  # 0=brightfield, 1=GFP, 2=DAPI
img = sim.snap_frame(exposure=50.0, intensity=1.0)
```

## Backends

| Backend | Description | Dynamic | Channels |
|---------|-------------|:-------:|----------|
| `bacteria` | Rod-shaped bacteria, growth & division | yes | BF, GFP, DAPI |
| `blood_smear` | Wright-Giemsa stained peripheral blood | no | BF, nucleus, membrane |
| `calcium` | Calcium wave propagation (FitzHugh-Nagumo) | yes | BF, GCaMP, E-cadherin |
| `cardio` | Cardiac tissue with calcium transients | yes | BF, GCaMP, membrane |
| `celegans` | C. elegans with sinusoidal locomotion | yes | BF, GFP-pharynx, mCherry |
| `colony_counter` | Bacterial colony plates | optional | transmitted, blue-filter, GFP |
| `dictyostelium` | Dictyostelium aggregation, cAMP waves | yes | dark-field, GFP, cAMP reporter |
| `fibroblast` | Fibroblasts with stress fibers | no | BF, DAPI, phalloidin |
| `fish` | FISH probes on tissue sections | yes | BF, FISH nucleus, cytoplasm |
| `flow_cytometry` | Flow cytometry scatter & fluorescence | no | FSC-SSC, FL1-FITC, FL2-PE |
| `fucci` | FUCCI cell cycle reporter | yes | BF, RFP-Cdt1, GFP-Geminin |
| `gel_doc` | Gel electrophoresis (western/agarose/Coomassie) | no | gel image |
| `hemocytometer` | Neubauer chamber with trypan blue | no | BF, trypan blue |
| `histology` | H&E stained tissue sections | no | BF (H&E) |
| `lipid_droplet` | Hepatocytes with lipid droplets | yes | BF, nucleus, Nile Red |
| `lysosome` | LysoTracker-stained lysosomes | yes | BF, nucleus, LysoTracker |
| `malaria` | Giemsa smear with Plasmodium parasites | no | BF, chromatin, membrane |
| `microfluidics` | Microfluidic channel with flowing cells | yes | BF, DAPI, gradient |
| `mito` | Mitochondrial network, fission/fusion | yes | BF, DAPI, MitoTracker |
| `neuron` | Neurons with dendrites & synapses | no | BF, MAP2, synaptophysin |
| `organoid` | 3D organoid cross-section | no | BF, DAPI, E-cadherin |
| `particle` | Basic particle simulation | no | BF, nucleus, membrane |
| `plant_cell` | Plant cells with cell walls | no | BF (iodine), DAPI, Calcofluor White |
| `plate_reader` | 96-well microplate assays | no | primary λ, reference λ |
| `reaction_diffusion` | Gray-Scott Turing patterns | yes | BF, activator, inhibitor |
| `spheroid` | Multicellular tumor spheroid | no | BF, calcein-AM, PI |
| `spt` | Single-particle tracking | yes | epifluorescence, TIRF |
| `stress_granule` | Stress granules (G3BP1) | yes | BF, nucleus, G3BP1 |
| `viability` | Live/dead viability staining | yes | BF, calcein-AM, PI |
| `volvox` | Volvox colonial algae | yes | BF, chlorophyll |
| `voronoi` | Base Voronoi tissue simulation | no | BF, nucleus, membrane |
| `wound_healing` | Scratch wound healing assay | yes | BF, nucleus, membrane |
| `yeast` | Budding yeast (S. cerevisiae) | yes | phase, Calcofluor White, GFP |
| `zebrafish` | Zebrafish embryo, transgenic reporters | yes | BF, flk1:GFP, myl7:mCherry |

## Architecture

```
load_backend("bacteria")
    │
    ▼
┌──────────────┐     ┌───────────────────┐     ┌──────────┐
│  UniMMCore   │────▶│ SimulationBridge   │────▶│  Sim     │
│  (device API)│     │ (snap/stage/focus) │     │ (render) │
└──────────────┘     └───────────────────┘     └──────────┘
```

- **`load_backend(name)`** returns `(core, sim)` — `core` is a pymmcore-plus `UniMMCore` instance with virtual devices, `sim` is the simulation backend
- **Channel switching** via `core.setConfig("Fake", "brightfield" | "nucleus-channel" | "membrane-channel")`
- **Stage movement** via `core.setXYPosition(x, y)` — shifts the viewport over the simulated specimen
- **Focus control** via `core.setPosition("Z", z)` — adjusts defocus blur
- Each sim renders at `internal_scale × world_size` resolution, then crops a 512×512 viewport

## Adding a Backend

1. Create `src/virtual_microscope/backends/your_backend/__init__.py`:
   - `create_sim(**params)` — returns a sim object
   - `setup_your_backend_microscope(**params)` — returns `(core, sim)`
2. Create `src/virtual_microscope/backends/your_backend/sim.py` implementing:
   - `snap_frame(mask=None, exposure=50.0, intensity=1.0)` → `np.ndarray` (grayscale uint8)
   - `camera_offset`, `state_devices`, `viewport_width`, `viewport_height`
   - `set_focal_plane(z)`, `mode` (channel selector)
3. Add device configs in `.cfg` and `.sim.json` files
4. Create `src/virtual_microscope/backends/your_backend/showcase.py`:
   - `create_showcase_images(seed=0)` → list of 4 `(512, 512, 3) uint8` RGB images
5. Run `python scripts/generate_gallery.py --backends your_backend` to verify
