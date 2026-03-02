#!/usr/bin/env python
"""Generate a visual gallery of all virtual-microscope backends.

Usage:
    python scripts/generate_gallery.py [--output-dir docs/gallery] [--backends bacteria,neuron]
    python scripts/generate_gallery.py --md-only          # regenerate gallery.md without re-rendering images
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import traceback
from pathlib import Path

import numpy as np

from virtual_microscope.backends import list_backends, describe_backends


# ── Badge helpers ────────────────────────────────────────────────────────────

def _shields_escape(text: str) -> str:
    """Escape text for shields.io URL path: - → --, _ → __, space → _."""
    return text.replace("-", "--").replace("_", "__").replace(" ", "_")


def _badge(label: str, color: str) -> str:
    """Return a shields.io markdown badge image."""
    slug = _shields_escape(label)
    return f"![{label}](https://img.shields.io/badge/{slug}-{color})"


def _channel_badge(ch: str) -> str:
    return _badge(ch, "ADD3FF")


def _device_badge(dev: str) -> str:
    return _badge(dev, "DBC8FF")


def _dynamics_badge(continuous: bool) -> str:
    if continuous:
        return _badge("continuous", "AFE2BD")
    return _badge("static", "CBCBCC")


# ── Gallery markdown generation ─────────────────────────────────────────────

def generate_md(
    results: list[tuple[str, list[str]]],
    backend_info: dict[str, dict],
) -> str:
    """Build gallery.md content from render results and BACKEND_INFO."""
    lines = ["# Virtual Microscope — Backend Gallery\n"]
    lines.append(
        f"Showcase images and short description for **{len(results)}** backends.\n"
    )
    # Legend
    lines.append(
        f"Dynamics: {_dynamics_badge(True)} {_dynamics_badge(False)}\n"
        f"Available imaging channels: {_channel_badge('channel')}\n"
        f"Extra hardware devices: {_device_badge('device')}\n"
    )
    lines.append("---\n")

    for name, frame_files in results:
        info = backend_info.get(name, {})
        desc = info.get("description", "")
        channels = info.get("channels", [])
        continuous = info.get("continuous", False)
        extra = info.get("extra_devices", [])
        specimen = info.get("specimen", "")
        modality = info.get("modality", "")
        guide = info.get("experiment_guide", "")
        device_effects = info.get("device_effects", {})
        key_params = info.get("key_parameters", {})

        # Title
        lines.append(f"## {name}\n")

        # Specimen + modality subtitle
        if specimen or modality:
            parts = [p for p in (specimen, modality) if p]
            lines.append(f'*{" — ".join(parts)}*\n')

        # Description
        if desc:
            lines.append(f"{desc}\n")

        # Image grid
        imgs = " ".join(
            f'<img src="gallery/frames/{f}" width="24%">'
            for f in frame_files
        )
        lines.append(f'<p style="display:flex;gap:4px">{imgs}</p>\n')

        # Badges row
        badges: list[str] = []
        badges.append(_dynamics_badge(continuous))
        for ch in channels:
            badges.append(_channel_badge(ch))
        for dev in extra:
            badges.append(_device_badge(dev))

        if badges:
            lines.append(" ".join(badges) + "\n")

        # Experiment guide
        if guide:
            lines.append(f"**Experiment guide:** {guide}\n")

        # Device effects
        if device_effects:
            effects = " | ".join(
                f"{dev} — {eff}" for dev, eff in device_effects.items()
            )
            lines.append(f"**Devices:** {effects}\n")

        # Key parameters
        if key_params:
            params = " · ".join(
                f"`{k}` ({v})" for k, v in key_params.items()
            )
            lines.append(f"**Parameters:** {params}\n")

    return "\n".join(lines)


# ── Showcase runner ──────────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate virtual-microscope backend gallery")
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--output-dir", default=str(_repo_root / "docs" / "gallery"), help="Output directory for gallery")
    parser.add_argument("--backends", default=None, help="Comma-separated list of backends (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for showcase images")
    parser.add_argument("--md-only", action="store_true", help="Regenerate gallery.md without re-rendering images")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_backends = list_backends()
    backend_info = describe_backends()

    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]
        invalid = set(backends) - set(all_backends)
        if invalid:
            print(f"Unknown backends: {invalid}")
            sys.exit(1)
    else:
        backends = all_backends

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.md_only:
        # Discover existing frames
        results: list[tuple[str, list[str]]] = []
        for name in backends:
            frame_files = sorted(
                f.name for f in frames_dir.glob(f"{name}_*.png")
            )
            if len(frame_files) >= 4:
                results.append((name, frame_files[:4]))
        print(f"Found existing frames for {len(results)}/{len(backends)} backends")
    else:
        import cv2

        results = []
        for idx, name in enumerate(backends, 1):
            print(f"[{idx}/{len(backends)}] {name}...")
            images = run_showcase(name, seed=args.seed)
            if images is None:
                continue

            frame_files = []
            for i, img in enumerate(images):
                # Resize to 256x256
                rgb = img if img.ndim == 3 else np.stack([img]*3, axis=-1)
                small = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
                fname = f"{name}_{i+1:02d}.png"
                cv2.imwrite(str(frames_dir / fname), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
                frame_files.append(fname)

            results.append((name, frame_files))
            print(f"  OK: {frames_dir / name}_*.png")

            # Clean up memory
            del images
            gc.collect()

    # Generate gallery.md
    md_content = generate_md(results, backend_info)
    md_path = output_dir.parent / "gallery.md"
    md_path.write_text(md_content)
    print(f"\nGallery written to {md_path}")
    print(f"Total: {len(results)}/{len(backends)} backends")


if __name__ == "__main__":
    main()
