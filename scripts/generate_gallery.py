#!/usr/bin/env python
"""Generate a visual gallery of all virtual-microscope backends.

Usage:
    python scripts/generate_gallery.py [--output-dir docs/gallery] [--backends bacteria,neuron]
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

from virtual_microscope._showcase_utils import make_montage
from virtual_microscope.backends import list_backends

# Backend descriptions for the gallery markdown
DESCRIPTIONS: dict[str, str] = {
    "bacteria": "Rod-shaped bacteria with phase contrast, GFP fluorescence, and colony growth dynamics.",
    "blood_smear": "Wright-Giemsa stained peripheral blood smear with RBCs, WBCs, and platelets.",
    "calcium": "FitzHugh-Nagumo calcium wave propagation with GCaMP fluorescence imaging.",
    "cardio": "Cardiac tissue with calcium transients, contraction, and arrhythmia modeling.",
    "celegans": "C. elegans nematode with sinusoidal locomotion and transgenic fluorescence.",
    "colony_counter": "Bacterial colony plates with spread/streak patterns and blue-white screening.",
    "dictyostelium": "Dictyostelium aggregation with cAMP waves and chemotactic streaming.",
    "fibroblast": "Elongated fibroblasts with stress fibers, focal adhesions, and drug responses.",
    "fish": "FISH (fluorescence in situ hybridization) probes on tissue sections.",
    "flow_cytometry": "Flow cytometry with FSC/SSC scatter and multi-color fluorescence channels.",
    "fucci": "FUCCI cell cycle reporter with G1-red (Cdt1) and S/G2/M-green (Geminin).",
    "gel_doc": "Gel electrophoresis documentation: western blots, agarose gels, Coomassie staining.",
    "hemocytometer": "Neubauer hemocytometer with trypan blue viability staining and grid overlay.",
    "histology": "H&E stained tissue histology sections with glandular architecture.",
    "lipid_droplet": "Hepatocytes with lipid droplets (Nile Red) modeling steatosis.",
    "lysosome": "LysoTracker-stained lysosomes with dynamic intracellular movement.",
    "malaria": "Giemsa-stained blood smear with Plasmodium falciparum parasites at various stages.",
    "microfluidics": "Microfluidic channel with flowing cells, traps, and chemical gradients.",
    "mito": "Mitochondrial network with MitoTracker staining, fission, and fusion dynamics.",
    "neuron": "Primary neurons with MAP2 dendrites and synaptophysin synaptic puncta.",
    "organoid": "3D organoid cross-section with lumen, epithelial layer, and E-cadherin staining.",
    "particle": "Basic particle simulation with scattered fluorescent cells.",
    "plant_cell": "Plant cells with cell walls, chloroplasts, and iodine/Calcofluor White staining.",
    "plate_reader": "96-well microplate reader with viability, ELISA, and luminescence assays.",
    "reaction_diffusion": "Gray-Scott reaction-diffusion patterns (Turing patterns, waves, spots).",
    "spheroid": "Multicellular tumor spheroid with necrotic core and live/dead staining.",
    "spt": "Single-particle tracking with free, confined, and directed diffusion modes.",
    "stress_granule": "Stress granules (G3BP1) with dynamic formation and dissolution.",
    "viability": "Live/dead cell viability assay with calcein-AM and propidium iodide.",
    "volvox": "Volvox colonial algae with somatic cells and gonidia, chlorophyll fluorescence.",
    "voronoi": "Basic Voronoi tissue simulation with brightfield, nucleus, and membrane channels.",
    "wound_healing": "Scratch wound healing assay with cell migration and gap closure dynamics.",
    "yeast": "Budding yeast (S. cerevisiae) with Calcofluor White bud scars and GFP reporter.",
    "zebrafish": "Zebrafish embryo with transgenic vascular (flk1:GFP) and cardiac (myl7:mCherry) reporters.",
}


def run_showcase(name: str, seed: int = 0) -> list[np.ndarray] | None:
    """Import and run a backend's create_showcase_images, returning images or None."""
    try:
        mod = importlib.import_module(f"virtual_microscope.backends.{name}.showcase")
        images = mod.create_showcase_images(seed=seed)
        if not isinstance(images, list) or len(images) != 4:
            print(f"  WARNING: {name} returned {len(images) if isinstance(images, list) else type(images)} instead of 4 images")
            return None
        for i, img in enumerate(images):
            if img.shape[:2] != (512, 512):
                print(f"  WARNING: {name} image {i} has shape {img.shape}, expected (512, 512, ...)")
                return None
        return images
    except ImportError:
        print(f"  SKIP: {name} — no showcase.py module")
        return None
    except Exception:
        print(f"  ERROR: {name}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate virtual-microscope backend gallery")
    parser.add_argument("--output-dir", default="docs/gallery", help="Output directory for gallery")
    parser.add_argument("--backends", default=None, help="Comma-separated list of backends (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for showcase images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_backends = list_backends()
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]
        invalid = set(backends) - set(all_backends)
        if invalid:
            print(f"Unknown backends: {invalid}")
            sys.exit(1)
    else:
        backends = all_backends

    results: list[tuple[str, str]] = []  # (name, png_filename)

    for name in backends:
        print(f"[{backends.index(name) + 1}/{len(backends)}] {name}...")
        images = run_showcase(name, seed=args.seed)
        if images is None:
            continue

        montage = make_montage(images, gap=2)
        png_path = output_dir / f"{name}.png"
        # Convert RGB to BGR for cv2
        cv2.imwrite(str(png_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
        results.append((name, f"{name}.png"))
        print(f"  OK: {png_path}")

        # Clean up memory
        del images, montage
        gc.collect()

    # Generate gallery.md
    md_lines = ["# Virtual Microscope — Backend Gallery\n"]
    md_lines.append(f"Generated showcase images for **{len(results)}** of {len(all_backends)} backends.\n")
    md_lines.append("Each row shows 4 representative images arranged in a horizontal strip.\n")

    for name, png_file in results:
        desc = DESCRIPTIONS.get(name, "")
        md_lines.append(f"## {name}\n")
        if desc:
            md_lines.append(f"{desc}\n")
        md_lines.append(f"![{name}]({png_file})\n")

    md_path = output_dir / "gallery.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\nGallery written to {md_path}")
    print(f"Total: {len(results)}/{len(backends)} backends rendered successfully")


if __name__ == "__main__":
    main()
