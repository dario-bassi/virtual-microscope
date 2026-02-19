"""Example: Load a backend programmatically and take a snapshot."""

from virtual_microscope import setup_bacteria_microscope, list_backends

# Show all available backends
print("Available backends:", list_backends())

# Set up the bacteria microscope
core, sim = setup_bacteria_microscope(n_cells=30, seed=42)

# Take a snapshot in brightfield
core.setConfig("Fake", "brightfield")
core.snapImage()
img = core.getImage()
print(f"Brightfield image: {img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")

# Switch to fluorescence and snap
core.setConfig("Fake", "nucleus-channel")
core.snapImage()
img_fl = core.getImage()
print(f"Fluorescence image: {img_fl.shape}, dtype={img_fl.dtype}")

# Move stage and snap
core.setXYPosition(300, 256)
core.snapImage()
img_moved = core.getImage()
print(f"Stage-moved image: {img_moved.shape}")

print("Done!")
