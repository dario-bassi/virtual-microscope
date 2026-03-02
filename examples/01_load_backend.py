"""Example: Load a backend programmatically and take a snapshot."""

from virtual_microscope import setup_bacteria, list_backends

# Show all available backends
print("Available backends:", list_backends())

# Set up the bacteria microscope
core, sim = setup_bacteria(n_cells=30, seed=42)

# Take a snapshot in phase-contrast
core.setConfig("Channel", "phase-contrast")
core.snapImage()
img = core.getImage()
print(f"Phase-contrast image: {img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

# Switch to fluorescence and snap
core.setConfig("Channel", "GFP")
core.snapImage()
img_fl = core.getImage()
print(f"Fluorescence image: {img_fl.shape}, dtype={img_fl.dtype}")

# Move stage and snap
core.setXYPosition(300, 256)
core.snapImage()
img_moved = core.getImage()
print(f"Stage-moved image: {img_moved.shape}")

print("Done!")
