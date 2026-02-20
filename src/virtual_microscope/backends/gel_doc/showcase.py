"""Showcase images for the gel_doc backend."""

from __future__ import annotations

import numpy as np
from virtual_microscope._showcase_utils import gray_to_rgb, snap_channel


def create_showcase_images(seed: int = 0) -> list[np.ndarray]:
    """Return 4 visually compelling 512x512 RGB images of the gel_doc backend.

    Native RGB gel images — different gel types.
    1. Western blot — chemiluminescent bands
    2. Agarose gel — EtBr-stained DNA bands
    3. Coomassie stain — protein gel
    4. Western blot with different seed
    """
    from virtual_microscope.backends.gel_doc import create_sim

    sim_w = create_sim(n_lanes=8, gel_type="western", seed=seed)
    img1 = gray_to_rgb(snap_channel(sim_w, mode=0, exposure=50.0), stretch=False)

    sim_a = create_sim(n_lanes=8, gel_type="agarose", seed=seed)
    img2 = gray_to_rgb(snap_channel(sim_a, mode=0, exposure=50.0), stretch=False)

    sim_c = create_sim(n_lanes=8, gel_type="coomassie", seed=seed)
    img3 = gray_to_rgb(snap_channel(sim_c, mode=0, exposure=50.0), stretch=False)

    sim_w2 = create_sim(n_lanes=8, gel_type="western", seed=seed + 5)
    img4 = gray_to_rgb(snap_channel(sim_w2, mode=0, exposure=50.0), stretch=False)

    del sim_w, sim_a, sim_c, sim_w2
    return [img1, img2, img3, img4]
