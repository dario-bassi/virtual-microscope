"""Example: Load a backend via .cfg file (requires pymmcore-plus >= 0.17.0).

This demonstrates the napari-micromanager compatible workflow where the
virtual microscope is configured entirely through a .cfg file.
"""

from pathlib import Path
from pymmcore_plus.experimental.unicore import UniMMCore

# Find the bacteria cfg
cfg_path = Path(__file__).parent.parent / "src/virtual_microscope/backends/bacteria/bacteria.cfg"
print(f"Loading config: {cfg_path}")

core = UniMMCore()
core.loadSystemConfiguration(str(cfg_path))

# Take a snapshot
core.setConfig("Fake", "brightfield")
core.snapImage()
img = core.getImage()
print(f"Image via .cfg: {img.shape}, dtype={img.dtype}")
print("Done!")
