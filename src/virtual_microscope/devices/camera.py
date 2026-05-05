import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Callable, Optional

from pymmcore_plus import PropertyType
import numpy as np
from numpy.typing import DTypeLike
from pymmcore_plus.experimental.unicore import CameraDevice, UniMMCore
from pymmcore_plus.experimental.unicore import pymm_property
import virtual_microscope.engine.simulation_bridge as bridge_module

class SimCameraDevice(CameraDevice):
    """
    pymmcore_camera_sim.py

    A virtual camera device for pymmcore that generates images using the microscope_sim.py simulation.
    """
    _exposure: float = 50.0
    _brightness: float = 1.0
    _gain: float = 1.0
    _binning: int = 2 # default 2x2
    _mask: Optional[np.ndarray] = None
    _led_channel: str = None
    _filter_wheel_channel: str = None
    _last_sim_time: float = 0.0

    def __init__(self) -> None:

        super().__init__()
        self.bridge = bridge_module.GLOBAL_BRIDGE
        self._mask = None
        # change limits of binning
        self.set_property_limits("Binning", (0, 20))

    def _get_bridge(self):
        """Always use the current global bridge (may be swapped at runtime)."""
        return bridge_module.GLOBAL_BRIDGE

    def get_exposure(self) -> float:
        return self._exposure

    def set_exposure(self, exposure: float) -> None:
        self._exposure = float(exposure)
        self.core.events.exposureChanged.emit(self.get_label(), float(exposure))

    def shape(self) -> tuple[int, ...]:
        bridge = self._get_bridge()
        if bridge is None:
            return 512, 512  # Default fallback dimensions
        sim = bridge._sim
        h, w = sim.viewport_height, sim.viewport_width
        if getattr(sim, 'rgb_mode', False):
            return h, w, 3
        return h, w

    def dtype(self) -> DTypeLike:
        bridge = self._get_bridge()
        if bridge is not None:
            return getattr(bridge._sim, 'pixel_dtype', np.uint8)
        return np.uint8

    def set_mask(self, mask: Optional[np.ndarray]) -> None:
        self._mask = mask

    # Minimum interval between frames (seconds) to avoid starving the Qt main
    # thread.  The rendering holds the GIL, so without a gap the UI freezes.
    _MIN_FRAME_INTERVAL: float = 0.08  # ~12 FPS max

    def start_sequence(
        self,
        n: int | None,
        get_buffer: Callable[[Sequence[int], DTypeLike], np.ndarray],
    ) -> Iterator[Mapping]:

        count = 0
        while n is None or count < n:
            t0 = time.perf_counter()
            bridge = self._get_bridge()
            self._mask = bridge.get_slm_mask() # type: ignore
            surf = bridge.snap(brightness=self._brightness, exposure=self._exposure,
                               gain=self._gain)  # type: ignore

            # Record simulation time for the SimTime property
            sim = bridge._sim if bridge else None
            if sim is not None:
                self._last_sim_time = getattr(sim, '_time', 0.0)

            buf = get_buffer(surf.shape, self.dtype())
            buf[:] = surf

            yield {
                "data": buf,
                "timestamp": time.time()
                }
            count += 1

            # Throttle: ensure minimum interval so the main thread gets GIL time
            elapsed = time.perf_counter() - t0
            remaining = self._MIN_FRAME_INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # define property brightness
    @pymm_property(
        limits=(0.0,100.0),
        sequence_max_length=100,
        name="brightness",
        property_type=PropertyType.Float
    )
    def brightness(self) -> float:
        """
        Get the brightness of the virtual camera.
        """
        return self._brightness

    # setter methods
    @brightness.setter
    def brightness(self, value: float) -> None:
        """
        Send the values to the virtual hardware to update the brightness.
        """
        self._brightness = value

    @brightness.sequence_loader
    def _load_brightness_sequence(self, sequence: Sequence[float]) -> None:
        pass

    @brightness.sequence_starter
    def _start_brightness_sequence(self) -> None:
        pass

    @pymm_property(
        limits=(0.0, 1e9),
        name="SimTime",
        property_type=PropertyType.Float,
    )
    def sim_time(self) -> float:
        """Simulation time (seconds) at the moment of the last snap.

        Returns the sim's internal clock (``_time`` attribute) captured
        when ``start_sequence`` yields a frame.  For backends without an
        explicit clock, returns 0.0.
        """
        return self._last_sim_time

    @sim_time.setter
    def sim_time(self, value: float) -> None:
        pass  # read-only; ignore writes

    @pymm_property(
        limits=(1.0, 32.0),
        sequence_max_length=100,
        name="Gain",
        property_type=PropertyType.Float,
    )
    def gain(self) -> float:
        """Analog gain multiplier (1.0 = no amplification, 32.0 = max).

        Higher gain amplifies the signal but also amplifies noise,
        reducing the effective dynamic range. Use higher gain for dim
        samples when increasing exposure is not possible (e.g. fast
        timelapse, motion blur concerns, photobleaching-sensitive).
        """
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = max(1.0, min(32.0, value))

    def get_binning(self) -> int:
        """
        Return the current binning of the virtual camera.
        """
        return self._binning

    def set_binning(self, binning: int) -> None:
        """
        Set the current binning of the virtual camera.
        """
        self._binning = binning

    def get_roi(self) -> tuple[int, int, int, int]:
        h, w = self.shape()[:2]
        return (0, 0, w, h)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        # Virtual camera generates fixed-size images; ROI is accepted but not applied.
        pass