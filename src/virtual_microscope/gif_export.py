"""
gif_export.py — Timelapse GIF export utility.

Simple functions to export lists of frames as animated GIFs.
Works with any backend's output (numpy uint8 arrays).

Usage:
    from gif_export import save_gif, save_timelapse_gif

    # Basic: list of numpy frames
    save_gif(frames, "/tmp/my_movie.gif", duration=200)

    # From a simulation: snap N frames and export
    save_timelapse_gif(core, n_frames=30, path="/tmp/timelapse.gif",
                       channel="nucleus-channel")
"""

import numpy as np
from pathlib import Path


def save_gif(
    frames: list,
    path: str,
    duration: int = 200,
    loop: int = 0,
    resize: tuple = None,
    add_frame_numbers: bool = False,
) -> str:
    """Save a list of numpy arrays as an animated GIF.

    Args:
        frames: List of numpy arrays (uint8, grayscale or RGB).
        path: Output file path.
        duration: Frame duration in ms.
        loop: Number of loops (0 = infinite).
        resize: Optional (width, height) to resize frames.
        add_frame_numbers: If True, burn frame number into corner.

    Returns:
        Path to saved GIF.
    """
    from PIL import Image, ImageDraw

    pil_frames = []
    for i, frame in enumerate(frames):
        if frame.ndim == 3 and frame.shape[2] == 3:
            img = Image.fromarray(frame, mode="RGB")
        elif frame.ndim == 2:
            img = Image.fromarray(frame, mode="L")
        else:
            img = Image.fromarray(frame)

        if resize:
            img = img.resize(resize, Image.LANCZOS)

        if add_frame_numbers:
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), f"t={i}", fill=255 if frame.ndim == 2 else (255, 255, 255))

        pil_frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    return path


def save_timelapse_gif(
    core,
    n_frames: int,
    path: str,
    channel: str = "nucleus-channel",
    duration: int = 200,
    add_frame_numbers: bool = True,
) -> str:
    """Snap N frames from a core and save as animated GIF.

    Args:
        core: UniMMCore instance.
        n_frames: Number of frames to acquire.
        path: Output file path.
        channel: Channel to acquire ("nucleus-channel", "brightfield", etc.).
        duration: Frame duration in ms.
        add_frame_numbers: Burn frame index into corner.

    Returns:
        Path to saved GIF.
    """
    channel_configs = {
        "nucleus-channel": ("mScarlet3(569/582)", "ORANGE"),
        "membrane-channel": ("miRFP670(642/670)", "RED"),
        "gcamp": ("TagGFP2(483/506)", "GREEN"),
        "brightfield": (None, None),
    }

    config = channel_configs.get(channel, (None, None))

    frames = []
    for i in range(n_frames):
        if channel == "brightfield":
            core.setConfig("Channel", "brightfield")
        else:
            filt, led = config
            if filt:
                core.setState("Filter Wheel", filt)
            if led:
                core.setState("LED", led)

        frame = core.snap()
        frames.append(frame)

    return save_gif(frames, path, duration=duration,
                    add_frame_numbers=add_frame_numbers)


def frames_from_scenario(scenario_module, channel="nucleus-channel",
                         n_frames=30):
    """Generate frames from a scenario's generate() function.

    Args:
        scenario_module: Module with generate() function.
        channel: Channel to snap.
        n_frames: Number of frames.

    Returns:
        List of numpy arrays.
    """
    gt, _, core = scenario_module.generate()

    channel_configs = {
        "nucleus-channel": ("mScarlet3(569/582)", "ORANGE"),
        "membrane-channel": ("miRFP670(642/670)", "RED"),
        "gcamp": ("TagGFP2(483/506)", "GREEN"),
        "brightfield": (None, None),
    }

    config = channel_configs.get(channel, (None, None))

    frames = []
    for i in range(n_frames):
        if channel == "brightfield":
            core.setConfig("Channel", "brightfield")
        else:
            filt, led = config
            if filt:
                core.setState("Filter Wheel", filt)
            if led:
                core.setState("LED", led)
        frames.append(core.snap())

    return frames, gt
