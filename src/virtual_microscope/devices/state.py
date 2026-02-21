from pymmcore_plus.experimental.unicore import StateDevice
import virtual_microscope.engine.simulation_bridge as bridge_module


class GenericStateDevice(StateDevice):
    """A state device with caller-defined labels and name.

    All state devices in the simulation use this base. The three
    microscope-specific subclasses (LED, Filter Wheel, Objective)
    just supply default label dicts.

    Example:
        dev = GenericStateDevice("Channel", {0: "570nm", 1: "690nm-ref"})
    """

    def __init__(self, name: str, labels: dict[int, str]) -> None:
        super().__init__(labels)
        self._current_state = 0
        self._current_label = self._state_to_label.get(self._current_state)
        self._name = name
        if bridge_module.GLOBAL_BRIDGE is not None:
            self.update_microscope_simulation()

    def initialize(self) -> None:
        """Push current state to bridge after SimServer has set GLOBAL_BRIDGE.

        initializeAllDevices() runs all devices in parallel threads, so we
        wait for the bridge_ready event (set by SimServer.initialize()) before
        pushing state. Timeout of 5s prevents hangs if SimServer fails.
        """
        bridge_module.bridge_ready.wait(timeout=5.0)
        self.update_microscope_simulation()

    def get_state(self) -> int:
        return self._current_state

    def set_state(self, position: int | str) -> None:
        if isinstance(position, str):
            position = int(position)
        self._current_state = position
        self._current_label = self._state_to_label.get(self._current_state)
        self.update_microscope_simulation()

    def update_microscope_simulation(self) -> None:
        bridge = bridge_module.GLOBAL_BRIDGE
        if bridge is not None:
            bridge.update_state({
                self._name: {
                    "state": str(self._current_state),
                    "label": self._current_label,
                }
            })


class FilterWheelDevice(GenericStateDevice):
    """Fluorescence microscope filter wheel (7 emission filters)."""

    def __init__(self) -> None:
        super().__init__("Filter Wheel", {
            0: "Electra1(402/454)",
            1: "SCFP2(434/474)",
            2: "TagGFP2(483/506)",
            3: "obeYFP(514/528)",
            4: "mRFP1-Q667(549/570)",
            5: "mScarlet3(569/582)",
            6: "miRFP670(642/670)",
        })


class LEDDevice(GenericStateDevice):
    """Fluorescence microscope LED excitation source (7 wavelengths)."""

    def __init__(self) -> None:
        super().__init__("LED", {
            0: "UV",
            1: "BLUE",
            2: "CYAN",
            3: "GREEN",
            4: "YELLOW",
            5: "ORANGE",
            6: "RED",
        })


class ObjectiveDevice(GenericStateDevice):
    """Microscope objective turret (4 magnifications)."""

    def __init__(self) -> None:
        super().__init__("Objective", {
            0: "10x",
            1: "20x",
            2: "40x",
            3: "100x",
        })


class TemperatureControllerDevice(GenericStateDevice):
    """Stage-top incubator / temperature controller.

    Provides discrete temperature presets covering key biological ranges:
    cold storage (4°C), room temp (20°C), standard incubation (25-37°C),
    and heat shock (42°C).  Backends that support temperature modulation
    read the current label from ``self.state_devices["Temperature"]``
    and adjust dynamics accordingly.
    """

    # Mapping: state index → label string (°C value embedded)
    TEMPS = {
        0: "20",   # room temperature (default)
        1: "4",    # cold — stops most activity
        2: "25",   # C. elegans / yeast optimal
        3: "30",   # many microbes
        4: "37",   # mammalian body temperature
        5: "42",   # heat shock
        6: "15",   # cool — C. elegans slow but active
        7: "22",   # standard room temp
        8: "10",   # cold — minimal activity
        9: "28",   # zebrafish optimal
    }

    def __init__(self) -> None:
        super().__init__("Temperature", self.TEMPS)


class StretchDevice(GenericStateDevice):
    """Uniaxial substrate stretch device (flexible PDMS membrane).

    Applies cyclic uniaxial strain to the culture substrate.
    Fibroblasts reorient perpendicular to the stretch axis over time.
    Stretch direction is horizontal (x-axis).

    State 0 = off, states 1-4 = increasing strain magnitudes.
    Backends read ``self.state_devices["Stretch"]`` to get strain %.
    """

    MODES = {
        0: "Off",     # no stretch
        1: "5%",      # gentle
        2: "10%",     # moderate
        3: "15%",     # strong
        4: "20%",     # high
    }

    def __init__(self) -> None:
        super().__init__("Stretch", self.MODES)


class PerfusionPumpDevice(GenericStateDevice):
    """Microfluidic perfusion pump controller.

    Controls flow speed and drug delivery in microfluidic backends.
    State 0 = off (no flow), states 1-3 = flow speeds, state 4 = drug delivery.
    Backends read ``self.state_devices["Perfusion"]`` to get flow parameters.
    """

    MODES = {
        0: "Off",       # no flow
        1: "Slow",      # 1 px/s
        2: "Medium",    # 3 px/s (default)
        3: "Fast",      # 10 px/s
        4: "Drug",      # 3 px/s + drug enabled
    }

    def __init__(self) -> None:
        super().__init__("Perfusion", self.MODES)


class AnesthesiaDevice(GenericStateDevice):
    """Tricaine (MS-222) anesthesia delivery for zebrafish.

    Controls anesthesia concentration in the fish water.
    State 0 = none, states 1-3 = increasing Tricaine concentration.
    Higher concentrations suppress cardiac activity and blood flow.
    Backends read ``self.state_devices["Anesthesia"]`` to get suppression level.
    """

    MODES = {
        0: "None",       # no anesthesia
        1: "0.01%",      # light — mild bradycardia (~50% HR reduction)
        2: "0.02%",      # standard — strong bradycardia (~80% HR reduction)
        3: "0.04%",      # deep — near cardiac arrest (~95% HR reduction)
    }

    def __init__(self) -> None:
        super().__init__("Anesthesia", self.MODES)


class ElectrodeDevice(GenericStateDevice):
    """DC electric field electrode for galvanotaxis experiments.

    Applies a directional electric field to the culture dish.
    Epithelial and fibroblast cells undergo galvanotaxis — directed migration
    toward the cathode (negative electrode).  This device controls which
    direction is the cathode, and therefore in which direction cells migrate.

    State 0 = Off (no field).
    States 1-4 encode the cathode direction:
      1 = "+X"  → cathode on right side  → cells migrate right
      2 = "-X"  → cathode on left side   → cells migrate left
      3 = "+Y"  → cathode on bottom      → cells migrate down
      4 = "-Y"  → cathode on top         → cells migrate up

    Backends that support galvanotaxis read
    ``self.state_devices["Electrode"]["label"]`` during ``step()`` to add
    a directed migration bias proportional to ``galvanotaxis_strength``.
    """

    MODES = {
        0: "Off",   # no electric field
        1: "+X",    # cathode right  → cells migrate right (+X)
        2: "-X",    # cathode left   → cells migrate left  (-X)
        3: "+Y",    # cathode bottom → cells migrate down  (+Y)
        4: "-Y",    # cathode top    → cells migrate up    (-Y)
    }

    def __init__(self) -> None:
        super().__init__("Electrode", self.MODES)


class SLMModeDevice(GenericStateDevice):
    """SLM excitation mode for reaction-diffusion backend.

    States: 0=excite (increase activator), 1=inhibit (suppress activator).
    No-arg constructor so it can be loaded via #py pyDevice in .cfg files.
    """

    MODES = {0: "excite", 1: "inhibit"}

    def __init__(self) -> None:
        super().__init__("SLM-Mode", self.MODES)


class HemocytometerChannelDevice(GenericStateDevice):
    """Channel selector for hemocytometer backend (brightfield / trypan-blue)."""

    MODES = {0: "brightfield", 1: "trypan-blue"}

    def __init__(self) -> None:
        super().__init__("Channel", self.MODES)


class ColonyCounterChannelDevice(GenericStateDevice):
    """Channel selector for colony counter backend (transmitted / blue-filter / GFP)."""

    MODES = {0: "transmitted", 1: "blue-filter", 2: "GFP-excitation"}

    def __init__(self) -> None:
        super().__init__("Channel", self.MODES)


class FlowCytometerDetectorDevice(GenericStateDevice):
    """Detector channel for flow cytometry backend (FSC-SSC / FL1-FITC / FL2-PE)."""

    MODES = {0: "FSC-SSC", 1: "FL1-FITC", 2: "FL2-PE"}

    def __init__(self) -> None:
        super().__init__("Detector", self.MODES)
