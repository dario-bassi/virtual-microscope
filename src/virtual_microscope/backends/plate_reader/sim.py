"""
PlateReaderSim — 96-well plate reader simulator.

Simulates ELISA / colorimetric / luminescence plate reader output as an
8-row x 12-column uint16 camera.  Each pixel is one well.

Encoding:
  - Absorbance assays: OD x 10000  (e.g. OD 1.234 → 12340)
  - Luminescence:      raw RLU clamped to uint16 range

Channels (LED/filter):
  - mode 0 (primary):   primary-wavelength plate
  - mode 1 (reference): reference-wavelength plate (low background)

For luminescence the reference channel returns the same as primary
(single-channel instrument).

No stage, no Z, no objective, no SLM — just Camera + LED + Filter Wheel.

Usage via SimulationBridge:
    sim = PlateReaderSim(assay_type='elisa', seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np


# Well plate geometry
N_ROWS = 8      # A-H
N_COLS = 12     # 1-12
ROW_LABELS = [chr(65 + i) for i in range(N_ROWS)]  # A-H

# Assay presets
ASSAY_PRESETS = {
    "elisa": {
        "name": "TNF-alpha ELISA",
        "readout": "absorbance",
        "wavelength_primary": 450,
        "wavelength_reference": 620,
        "units": "OD",
        "blank_value": 0.05,
        "max_value": 2.5,
        "noise_cv": 0.06,
        "layout": "dose_response",
    },
    "viability": {
        "name": "MTT Viability Assay",
        "readout": "absorbance",
        "wavelength_primary": 570,
        "wavelength_reference": 690,
        "units": "OD",
        "blank_value": 0.08,
        "max_value": 1.8,
        "noise_cv": 0.08,
        "layout": "dose_response",
    },
    "fluorescence": {
        "name": "CellTiter-Glo Luminescence",
        "readout": "luminescence",
        "wavelength_primary": None,
        "wavelength_reference": None,
        "units": "RLU",
        "blank_value": 50,
        "max_value": 50000,
        "noise_cv": 0.05,
        "layout": "dose_response",
    },
}

# Dose-response layout (standard 96-well drug screen):
# Col 1-2: Positive control (no drug, max signal)
# Col 3-10: Drug concentrations (8-point dilution, 3-fold)
# Col 11-12: Negative control (blank, media only)
#
# Rows A-D: Compound 1 (4 replicates)
# Rows E-H: Compound 2 (4 replicates)

DEFAULT_EXPERIMENT = {
    "name": "Dual compound screen",
    "compounds": [
        {
            "name": "Staurosporine",
            "rows": [0, 1, 2, 3],  # A-D
            "ic50": 0.5,      # uM
            "hill": 1.2,
            "max_effect": 0.95,  # max inhibition fraction
        },
        {
            "name": "Doxorubicin",
            "rows": [4, 5, 6, 7],  # E-H
            "ic50": 2.0,
            "hill": 0.8,
            "max_effect": 0.85,
        },
    ],
    "concentrations": [100, 33.3, 11.1, 3.7, 1.23, 0.41, 0.14, 0.046],  # uM
    "pos_ctrl_cols": [0, 1],     # columns 1-2 (0-indexed)
    "neg_ctrl_cols": [10, 11],   # columns 11-12 (0-indexed)
    "drug_cols": list(range(2, 10)),  # columns 3-10 (0-indexed)
}


class PlateReaderSim:
    """96-well plate reader simulator.

    Output: 8x12 uint16 array.  Each pixel = one well.

    Parameters
    ----------
    assay_type : str
        'elisa', 'viability', or 'fluorescence'.
    experiment : dict, optional
        Experiment definition. Defaults to dual compound screen.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        assay_type: str = "viability",
        experiment: dict = None,
        seed: int = 42,
    ):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.assay_type = assay_type

        # Camera geometry — 8 rows x 12 columns, grayscale uint16
        self.viewport_width = N_COLS   # 12
        self.viewport_height = N_ROWS  # 8
        self.rgb_mode = False
        self.pixel_dtype = np.uint16

        # SimulationBridge interface
        self.mode = 0
        self.camera_offset = [0, 0]
        self.state_devices = {}

        # Assay parameters
        preset = ASSAY_PRESETS.get(assay_type, ASSAY_PRESETS["viability"])
        self._assay_name = preset["name"]
        self._readout = preset["readout"]
        self._wavelength_primary = preset["wavelength_primary"]
        self._wavelength_reference = preset["wavelength_reference"]
        self._units = preset["units"]
        self._blank = preset["blank_value"]
        self._max_val = preset["max_value"]
        self._noise_cv = preset["noise_cv"]

        # Experiment
        self._experiment = experiment or DEFAULT_EXPERIMENT
        self._concentrations = self._experiment["concentrations"]
        self._compounds = self._experiment["compounds"]
        self._pos_cols = self._experiment["pos_ctrl_cols"]
        self._neg_cols = self._experiment["neg_ctrl_cols"]
        self._drug_cols = self._experiment["drug_cols"]

        # Generate plate data (float64 OD / RLU values)
        self._plate = np.zeros((N_ROWS, N_COLS), dtype=np.float64)
        self._generate_plate()

        # Introduce outlier wells (2-3 random wells with abnormal values)
        self._outliers = []
        self._add_outliers()

        # Pre-compute encoded plates
        self._plate_primary = self._encode_plate(self._plate)
        self._plate_reference = self._make_reference_plate()

    def _hill_equation(self, conc, ic50, hill, max_effect):
        """4-parameter Hill equation for dose-response.

        Returns fraction of signal remaining (1.0 = no effect, 0 = complete inhibition).
        """
        if conc <= 0:
            return 1.0
        response = 1.0 - max_effect * (conc ** hill) / (ic50 ** hill + conc ** hill)
        return float(max(0, response))

    def _generate_plate(self):
        """Generate plate data from experiment definition."""
        rng = self._rng

        for compound in self._compounds:
            rows = compound["rows"]
            ic50 = compound["ic50"]
            hill = compound["hill"]
            max_effect = compound["max_effect"]

            for row in rows:
                # Positive control: max signal (no drug)
                for col in self._pos_cols:
                    val = self._max_val * (1.0 + rng.normal(0, self._noise_cv))
                    self._plate[row, col] = max(self._blank, val)

                # Negative control: blank (media only)
                for col in self._neg_cols:
                    val = self._blank * (1.0 + rng.normal(0, self._noise_cv * 2))
                    self._plate[row, col] = max(0, val)

                # Drug concentrations
                for ci, col in enumerate(self._drug_cols):
                    conc = self._concentrations[ci]
                    response = self._hill_equation(conc, ic50, hill, max_effect)
                    val = self._max_val * response * (1.0 + rng.normal(0, self._noise_cv))
                    self._plate[row, col] = max(self._blank * 0.5, val)

    def _add_outliers(self):
        """Add 2-3 outlier wells (pipetting errors, bubbles, etc.)."""
        rng = self._rng
        n_outliers = rng.integers(2, 4)

        for _ in range(n_outliers):
            row = rng.integers(0, N_ROWS)
            col = rng.integers(0, N_COLS)
            # Random outlier: either very high or very low
            if rng.random() < 0.5:
                self._plate[row, col] = self._max_val * rng.uniform(1.3, 1.8)
            else:
                self._plate[row, col] = self._blank * rng.uniform(0.2, 0.8)
            self._outliers.append((row, col))

    def _encode_plate(self, plate: np.ndarray) -> np.ndarray:
        """Encode float plate values to uint16.

        Absorbance: OD * 10000 (e.g. 1.234 OD → 12340)
        Luminescence: raw RLU clamped to 0-65535
        """
        if self._readout == "luminescence":
            return np.clip(plate, 0, 65535).astype(np.uint16)
        else:
            return np.clip(plate * 10000, 0, 65535).astype(np.uint16)

    def _make_reference_plate(self) -> np.ndarray:
        """Generate reference wavelength plate.

        Reference wavelength has low uniform background (no dose-response
        pattern) — used for path-length correction.  For luminescence
        instruments there is no reference, so return the primary plate.
        """
        if self._readout == "luminescence":
            return self._plate_primary.copy()

        rng = np.random.default_rng(self._seed + 100)
        # Low OD background with slight noise
        ref_od = rng.uniform(0.04, 0.12, (N_ROWS, N_COLS))
        return self._encode_plate(ref_od)

    # -- SimulationBridge interface --

    def _update_mode(self):
        ch = self.state_devices.get("Channel", {})
        ch_label = ch.get("label", ch.get("Label", ""))

        # "ref" in label → reference wavelength (mode 1)
        if "ref" in ch_label.lower():
            self.mode = 1
        else:
            self.mode = 0

    def set_focal_plane(self, z):
        pass

    def snap_frame(self, mask=None, exposure=50, intensity=100, **kwargs):
        """Return an 8x12 uint16 plate reading.

        mode 0: primary wavelength
        mode 1: reference wavelength
        """
        self._update_mode()

        if self.mode == 1:
            return self._plate_reference.copy()
        return self._plate_primary.copy()

    def get_ground_truth(self):
        """Return ground truth for grading."""
        pos_vals = [self._plate[r, c] for c in self._pos_cols for r in range(N_ROWS)]
        neg_vals = [self._plate[r, c] for c in self._neg_cols for r in range(N_ROWS)]

        encoding = "OD_x_10000" if self._readout != "luminescence" else "raw_RLU"

        return {
            "assay_name": self._assay_name,
            "assay_type": self.assay_type,
            "units": self._units,
            "encoding": encoding,
            "pixel_dtype": "uint16",
            "plate_shape": [N_ROWS, N_COLS],
            "wavelength_primary": self._wavelength_primary,
            "wavelength_reference": self._wavelength_reference,
            "plate_values": self._plate.round(4).tolist(),
            "concentrations": self._concentrations,
            "compounds": [
                {
                    "name": c["name"],
                    "rows": c["rows"],
                    "ic50": c["ic50"],
                    "hill": c["hill"],
                    "max_effect": c["max_effect"],
                }
                for c in self._compounds
            ],
            "pos_ctrl_mean": round(float(np.mean(pos_vals)), 4),
            "neg_ctrl_mean": round(float(np.mean(neg_vals)), 4),
            "outlier_wells": [(int(r), int(c)) for r, c in self._outliers],
            "layout": {
                "pos_ctrl_cols": self._pos_cols,
                "neg_ctrl_cols": self._neg_cols,
                "drug_cols": self._drug_cols,
            },
        }
