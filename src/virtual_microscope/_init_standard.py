"""Standard device stack initializer — shared by all programmatic setup_* functions.

Replaces the old _init_devices() in setup_microscope.py.
Call this after setting GLOBAL_BRIDGE to load all standard devices.

Usage:
    from virtual_microscope._init_standard import init_standard_devices
    init_standard_devices(core, sim)
"""

from pymmcore_plus.experimental.unicore import UniMMCore

from virtual_microscope.devices.camera import SimCameraDevice
from virtual_microscope.devices.stage import SimStageDevice
from virtual_microscope.devices.z_stage import SimZStageDevice
from virtual_microscope.devices.shutter import SimShutterDevice
from virtual_microscope.devices.slm import SimSLMDevice
from virtual_microscope.devices.state import (
    LEDDevice,
    FilterWheelDevice,
    ObjectiveDevice,
    TemperatureControllerDevice,
    PerfusionPumpDevice,
    StretchDevice,
    ElectrodeDevice,
)


def _install_pixel_size(core: UniMMCore, sim) -> None:
    """Install getPixelSizeUm() returning correct µm/camera-px for current objective.

    Camera always outputs 512×512 px regardless of objective.
    FOV in world px: 10x=512, 20x=256, 40x=128, 100x=64.
    """
    world_px_um = getattr(sim, 'world_pixel_size_um', 1.0)
    _obj_factors = {0: 1, 1: 2, 2: 4, 3: 8}

    def _get_pixel_size_um(cached=False):
        obj_state = core.getState("Objective")
        factor = _obj_factors.get(obj_state, 1)
        return world_px_um / factor

    core.getPixelSizeUm = _get_pixel_size_um


def init_standard_devices(core: UniMMCore, sim=None) -> None:
    """Load and configure the full standard device stack.

    Loads: Camera, XYStage, ZStage, LED, Filter Wheel, Shutter, Objective,
           SLM, Temperature, Perfusion, Stretch, Electrode.
    Defines channel config group "Fake" with brightfield/nucleus/membrane.

    Args:
        core: UniMMCore instance to load devices into.
        sim: Simulation backend (used only for pixel size installation).
    """
    core.unloadAllDevices()

    core.loadPyDevice("Camera", SimCameraDevice())
    core.loadPyDevice("XYStage", SimStageDevice())
    core.loadPyDevice("ZStage", SimZStageDevice())
    core.loadPyDevice("LED", LEDDevice())
    core.loadPyDevice("Filter Wheel", FilterWheelDevice())
    core.loadPyDevice("Shutter", SimShutterDevice())
    core.loadPyDevice("Objective", ObjectiveDevice())
    core.loadPyDevice("SLM", SimSLMDevice())
    core.loadPyDevice("Temperature", TemperatureControllerDevice())
    core.loadPyDevice("Perfusion", PerfusionPumpDevice())
    core.loadPyDevice("Stretch", StretchDevice())
    core.loadPyDevice("Electrode", ElectrodeDevice())

    for dev in ("Camera", "XYStage", "ZStage", "LED", "Filter Wheel",
                "Shutter", "Objective", "SLM", "Temperature", "Perfusion",
                "Stretch", "Electrode"):
        core.initializeDevice(dev)

    core.setCameraDevice("Camera")
    core.setXYStageDevice("XYStage")
    core.setFocusDevice("ZStage")
    core.setShutterDevice("Shutter")
    core.setSLMDevice("SLM")

    core.setState("LED", 0)
    core.setState("Filter Wheel", 0)
    core.setState("Objective", 0)
    core.setState("Temperature", 0)   # 20°C (room temp)
    core.setState("Perfusion", 0)     # Off
    core.setState("Electrode", 0)     # Off (no electric field)

    core.defineConfigGroup("Fake")
    core.defineConfig("Fake", "nucleus-channel", "LED", "Label", "ORANGE")
    core.defineConfig("Fake", "nucleus-channel", "Filter Wheel", "Label",
                      "mScarlet3(569/582)")
    core.defineConfig("Fake", "membrane-channel", "LED", "Label", "RED")
    core.defineConfig("Fake", "membrane-channel", "Filter Wheel", "Label",
                      "miRFP670(642/670)")
    core.defineConfig("Fake", "brightfield", "LED", "Label", "CYAN")
    core.defineConfig("Fake", "brightfield", "Filter Wheel", "Label",
                      "Electra1(402/454)")

    core.setConfig("Fake", "brightfield")

    if sim is not None:
        _install_pixel_size(core, sim)
