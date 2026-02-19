"""Virtual microscope device adapters for pymmcore-plus.

All device classes can be loaded via core.loadPyDevice() or via .cfg files
using the #py pyDevice syntax (pymmcore-plus >= 0.17.0).
"""

from virtual_microscope.devices.camera import SimCameraDevice
from virtual_microscope.devices.stage import SimStageDevice
from virtual_microscope.devices.z_stage import SimZStageDevice
from virtual_microscope.devices.shutter import SimShutterDevice
from virtual_microscope.devices.slm import SimSLMDevice
from virtual_microscope.devices.state import (
    GenericStateDevice,
    LEDDevice,
    FilterWheelDevice,
    ObjectiveDevice,
    TemperatureControllerDevice,
    PerfusionPumpDevice,
    StretchDevice,
    AnesthesiaDevice,
    ElectrodeDevice,
)
from virtual_microscope.devices.sim_server import SimServer

__all__ = [
    "SimCameraDevice",
    "SimStageDevice",
    "SimZStageDevice",
    "SimShutterDevice",
    "SimSLMDevice",
    "GenericStateDevice",
    "LEDDevice",
    "FilterWheelDevice",
    "ObjectiveDevice",
    "TemperatureControllerDevice",
    "PerfusionPumpDevice",
    "StretchDevice",
    "AnesthesiaDevice",
    "ElectrodeDevice",
    "SimServer",
]
