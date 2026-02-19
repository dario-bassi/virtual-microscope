"""
Timelapse Sim — Z-axis-as-time wrapper for DynamicVoronoiSim.

Pre-computes tissue at N timepoints, stores snapshots of all channels.
The Z-stage position selects which timepoint to serve:
  Z=0 → t=0, Z=1 → t=1, ..., Z=N-1 → t=N-1

This allows agents to "scrub through time" using standard Z-stage controls,
observe wound healing dynamics, cell migration, division, etc.

Usage:
    from tissue_dynamics import DynamicVoronoiSim
    from timelapse_sim import TimelapseSim

    dyn = DynamicVoronoiSim(width=1024, height=1024, nb_cells=200,
                            migration_speed=2.0, rng_seed=42)
    dyn.create_wound(shape="rectangle", center=(512, 512), size=(200, 1024))

    tl = TimelapseSim(dyn, n_frames=20, steps_per_frame=5, dt=1.0)
    # tl is a VoronoiSim-compatible object — plug into SimulationBridge

    bridge = SimulationBridge(tl)
    # Agent sets Z=0 → first frame, Z=10 → 11th frame, etc.
"""

import numpy as np
import cv2


class TimelapseSim:
    """Z-axis-as-time wrapper for dynamic tissue simulations.

    Pre-computes all timepoints eagerly, then serves them based on
    the focal_plane (Z-stage) position. Quacks like VoronoiSim for
    SimulationBridge compatibility.
    """

    def __init__(self, dynamic_sim, n_frames: int = 10,
                 steps_per_frame: int = 1, dt: float = 1.0):
        """
        Args:
            dynamic_sim: A DynamicVoronoiSim (or any sim with step() and snap_frame())
            n_frames: Number of timepoints to pre-compute
            steps_per_frame: Dynamics steps between each captured frame
            dt: Timestep for each dynamics step
        """
        self._base_sim = dynamic_sim
        self.n_frames = n_frames
        self.steps_per_frame = steps_per_frame
        self.dt = dt

        # Forward sim attributes needed by SimulationBridge
        self.camera_offset = dynamic_sim.camera_offset
        self.focal_plane = 0.0
        self.viewport_width = dynamic_sim.viewport_width
        self.viewport_height = dynamic_sim.viewport_height
        self.state_devices = dynamic_sim.state_devices
        self.width = dynamic_sim.width
        self.height = dynamic_sim.height

        # Pre-compute snapshots
        self._snapshots = []  # list of {bf, nuc, mem, gt, extra_channels}
        self._ground_truths = []
        self._precompute()

        # Current frame index (from Z position)
        self._current_frame = 0

    def _precompute(self):
        """Run dynamics and capture all timepoints."""
        sim = self._base_sim

        for frame_idx in range(self.n_frames):
            # Ensure tissue is rendered
            if sim._bf_full is None:
                sim._render_full_tissue()

            # Capture rendered images (full-resolution, before viewport crop)
            snapshot = {
                "bf": sim._bf_full.copy() if sim._bf_full is not None else None,
                "nuc": sim._nuc_full.copy() if sim._nuc_full is not None else None,
                "mem": sim._mem_full.copy() if sim._mem_full is not None else None,
            }

            # Capture extra channels if any
            if hasattr(sim, '_extra_channels'):
                snapshot["extra"] = {}
                for mode_id, ch_info in sim._extra_channels.items():
                    snapshot["extra"][mode_id] = ch_info["image"].copy()

            self._snapshots.append(snapshot)

            # Capture ground truth at this timepoint
            gt = sim.get_ground_truth()
            gt["frame"] = frame_idx
            gt["time_step"] = frame_idx * self.steps_per_frame
            if hasattr(sim, 'get_wound_area'):
                gt["wound_area"] = round(sim.get_wound_area(), 1)
            if hasattr(sim, 'get_migration_front'):
                gt["migration_front"] = sim.get_migration_front()
            # Store cell centroids for tracking
            gt["centroids"] = [
                [round(float(sim.centers[i][0]), 1),
                 round(float(sim.centers[i][1]), 1)]
                for i in range(sim.nb_cells) if sim.alive[i]
            ]
            self._ground_truths.append(gt)

            # Advance dynamics (except after last frame)
            if frame_idx < self.n_frames - 1:
                for _ in range(self.steps_per_frame):
                    sim.step(self.dt)

    # ---- VoronoiSim-compatible interface ----

    def set_focal_plane(self, z: float):
        """Z position selects timepoint. Z=0 → frame 0, Z=1 → frame 1, etc."""
        self.focal_plane = z
        self._current_frame = int(np.clip(round(z), 0, self.n_frames - 1))

    def snap_frame(self, mask=None, exposure=50.0, intensity=1.0, **kwargs) -> np.ndarray:
        """Return the pre-computed frame at the current Z/time position."""
        # Update mode from state devices
        self._base_sim.state_devices = self.state_devices
        self._base_sim._update_mode()
        self._base_sim._update_objectif()

        snap = self._snapshots[self._current_frame]

        # Select channel based on current mode
        mode = self._base_sim.mode
        if mode == 0:
            full_img = snap["bf"]
        elif mode == 1:
            full_img = snap["nuc"]
        elif mode == 2:
            full_img = snap["mem"]
        elif "extra" in snap and mode in snap["extra"]:
            full_img = snap["extra"][mode]
        else:
            full_img = snap["bf"]

        if full_img is None:
            s = getattr(self._base_sim, 'internal_scale', 1)
            ih = self.height * s
            iw = self.width * s
            full_img = np.zeros((ih, iw, 3), dtype=np.uint8)

        # Use _crop_fov from base sim (handles internal_scale)
        self._base_sim.mode = mode
        viewport = self._base_sim._crop_fov(full_img)

        # Exposure scaling (BF uses 2× base for transmitted light)
        if mode == 0:
            scale = min(intensity * 0.02 * exposure, 2.0)
        else:
            scale = intensity * 0.01 * exposure
        viewport = (viewport.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)

        return cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)

    def update(self, dt: float = 0.016):
        """No-op — dynamics are pre-computed."""
        pass

    # ---- Ground truth access ----

    def get_ground_truth(self) -> dict:
        """Get ground truth for current timepoint."""
        return self._ground_truths[self._current_frame]

    def get_all_ground_truths(self) -> list:
        """Get ground truth for all timepoints."""
        return self._ground_truths

    def get_timelapse_summary(self) -> dict:
        """Summary ground truth across all timepoints."""
        gts = self._ground_truths
        return {
            "n_frames": self.n_frames,
            "steps_per_frame": self.steps_per_frame,
            "dt": self.dt,
            "cell_counts": [gt["n_cells"] for gt in gts],
            "wound_areas": [gt.get("wound_area", 0) for gt in gts],
            "z_to_frame": {i: i for i in range(self.n_frames)},
        }

    # ---- Forward remaining attributes to base sim ----

    def __getattr__(self, name):
        """Forward attribute access to the base sim for compatibility."""
        if name.startswith('_') and name != '_extra_channels':
            raise AttributeError(name)
        return getattr(self._base_sim, name)
