"""Multi-field simulation — independent tissue samples at different stage positions.

Simulates a well plate or multi-position experiment where each stage position
reveals a different simulation (e.g., different cell density, wound state,
marker pattern, drug concentration).

Usage:
    from multi_field_sim import MultiFieldBridge

    sims = {
        (1000, 1000): sim_control,    # Control well
        (3000, 1000): sim_drug_low,   # Low dose
        (5000, 1000): sim_drug_high,  # High dose
    }
    bridge = MultiFieldBridge(sims)

When the agent moves the stage to (3000, 1000), the bridge activates
sim_drug_low and renders from it. Each simulation is independent with
its own cells, dynamics, and rendering state.
"""

import numpy as np
from virtual_microscope.engine.simulation_bridge import SimulationBridge


class MultiFieldBridge(SimulationBridge):
    """SimulationBridge that routes to different simulations based on stage position.

    Each "field" is an independent simulation placed at a known stage coordinate.
    When the stage moves near a field center, that field's simulation becomes active.
    All SimulationBridge methods delegate to the currently-active simulation.
    """

    def __init__(self, field_sims: dict, snap_radius: float = 400.0):
        """
        Args:
            field_sims: Dict mapping (x_center, y_center) → simulation object.
                Each simulation must implement the snap_frame / update_state
                interface expected by SimulationBridge.
            snap_radius: Maximum distance from field center to activate it.
                If the stage is further than this from all fields, the nearest
                field is used anyway.
        """
        # Initialize parent with the first simulation
        positions = list(field_sims.keys())
        self._fields = field_sims
        self._positions = np.array(positions, dtype=float)  # (N, 2)
        self._field_list = list(field_sims.values())
        self._snap_radius = snap_radius
        self._active_idx = 0

        # Track last-known Z-stage position for field switches
        self._last_z = 0.0

        # Set the first field as default
        first_sim = self._field_list[0]
        super().__init__(first_sim)

    @property
    def active_field_index(self) -> int:
        """Index of the currently active field."""
        return self._active_idx

    @property
    def active_field_position(self) -> tuple:
        """Stage position of the currently active field."""
        return tuple(self._positions[self._active_idx])

    @property
    def n_fields(self) -> int:
        """Number of fields."""
        return len(self._field_list)

    def get_field_sim(self, idx: int):
        """Get the simulation at field index."""
        return self._field_list[idx]

    def set_focus(self, z: float) -> None:
        """Track Z-position and propagate to active sim."""
        self._last_z = z
        super().set_focus(z)

    def set_stage(self, x: float, y: float) -> None:
        """Move stage — selects the nearest field simulation."""
        self._stage_position = (x, y)

        # Find nearest field
        dists = np.sqrt(
            (self._positions[:, 0] - x) ** 2
            + (self._positions[:, 1] - y) ** 2
        )
        nearest = int(np.argmin(dists))

        if nearest != self._active_idx:
            # Copy current state_devices to the new sim so channel/objective
            # settings survive the field switch (core devices cache their state
            # internally and won't re-send it if unchanged).
            old_state = self._sim.state_devices
            self._active_idx = nearest
            self._sim = self._field_list[nearest]
            if old_state:
                self._sim.state_devices.update(old_state)
            # Propagate current Z-stage to new field's focal plane
            if hasattr(self._sim, 'set_focal_plane'):
                self._sim.set_focal_plane(self._last_z)

        # Set camera offset relative to the field's center
        fx, fy = self._positions[nearest]
        # Offset within the field: how far the stage is from field center
        local_x = x - fx + self._BASE_HALF
        local_y = y - fy + self._BASE_HALF
        self._sim.camera_offset = np.array([
            local_x - self._BASE_HALF,
            local_y - self._BASE_HALF,
        ])

    # ── Realtime dynamics support ───────────────────────────────────────────

    @property
    def auto_step(self):
        return False

    @auto_step.setter
    def auto_step(self, value):
        for sim in self._field_list:
            sim.auto_step = value

    @property
    def fixed_dt(self):
        return 0.0

    @fixed_dt.setter
    def fixed_dt(self, value):
        for sim in self._field_list:
            if hasattr(sim, 'fixed_dt'):
                sim.fixed_dt = value

    def step(self, dt: float = 1.0) -> None:
        """Step ALL field simulations (not just active) so non-viewed wells
        continue evolving while the agent looks elsewhere."""
        for sim in self._field_list:
            sim.step(dt)

    def step_autonomous(self, dt: float = 1.0) -> None:
        """Autonomous step for all fields (skips SLM effects)."""
        for sim in self._field_list:
            if hasattr(sim, 'step_autonomous'):
                sim.step_autonomous(dt)
            else:
                sim.step(dt)

    def snap_frame(self, *args, **kwargs):
        """Delegate to the active field's snap_frame.

        This method exists so RealtimeEngine can patch it with a lock,
        and snap() routes through it (keeping rendering thread-safe
        across field switches).
        """
        return self._sim.snap_frame(*args, **kwargs)

    def snap(self, exposure: float, brightness: float, gain: float = 1.0,
             **kwargs) -> np.ndarray:
        """Override parent snap() to route through our snap_frame().

        This ensures the RealtimeEngine's lock (patched onto self.snap_frame)
        protects rendering even after field switches.
        """
        mask = self.get_slm_mask()
        img = self.snap_frame(mask=mask, exposure=exposure,
                              intensity=brightness, **kwargs)
        if gain != 1.0:
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(
                np.uint8)
        return img

    # ── Ground truth ─────────────────────────────────────────────────────

    def get_all_ground_truth(self) -> list:
        """Get ground truth from all fields."""
        results = []
        for i, sim in enumerate(self._field_list):
            pos = tuple(self._positions[i])
            if hasattr(sim, "get_ground_truth"):
                gt = sim.get_ground_truth()
            else:
                gt = {}
            results.append({"position": pos, "field_index": i, **gt})
        return results
