"""
FlowCytometrySim — Flow cytometry simulation backend.

Simulates cells flowing through a laser beam in a flow cytometer.
Each cell produces scatter (FSC, SSC) and fluorescence signals.
The agent collects events by snapping stream images, detects cells,
measures their properties, and builds its own scatter plots.

All three channels show the same cells in the same positions for each
snap — the agent measures cell size (FSC), contrast (SSC), and
fluorescence (FL1, FL2) from the images, then accumulates events.

Channels:
  - mode 0 (brightfield): Stream view — cells sized by FSC, darkened by SSC
  - mode 1 (nucleus):     FL1-FITC stream — cells bright by CD3-FITC signal
  - mode 2 (membrane):    FL2-PE stream — cells bright by CD14-PE signal

Stage position has no effect (no spatial scanning).
Objective changes collection rate: 10x=100, 20x=200, 40x=500 events/snap.

Cell populations (blood sample):
  - Lymphocytes:  low FSC, low SSC,  CD3+ (some), CD14-
  - Monocytes:    med FSC, med SSC,  CD3-,         CD14+
  - Granulocytes: high FSC, high SSC, CD3-,        CD14-
  - Debris:       very low FSC/SSC

Usage via SimulationBridge:
    sim = FlowCytometrySim(seed=42)
    bridge = SimulationBridge(sim)
"""

import numpy as np
import cv2


# Population definitions
DEFAULT_POPULATIONS = [
    {
        "name": "lymphocyte",
        "fraction": 0.35,
        "fsc": (200, 40),   # (mean, std)
        "ssc": (120, 30),
        "fl1": (400, 80),   # CD3-FITC (T cells: ~70% of lymphocytes)
        "fl2": (50, 20),    # CD14-PE (negative)
        "fl1_positive_frac": 0.70,
        "fl2_positive_frac": 0.0,
    },
    {
        "name": "monocyte",
        "fraction": 0.10,
        "fsc": (350, 50),
        "ssc": (250, 50),
        "fl1": (50, 20),    # CD3 negative
        "fl2": (500, 100),  # CD14-PE positive
        "fl1_positive_frac": 0.0,
        "fl2_positive_frac": 0.95,
    },
    {
        "name": "granulocyte",
        "fraction": 0.45,
        "fsc": (400, 60),
        "ssc": (500, 80),
        "fl1": (40, 15),    # CD3 negative
        "fl2": (40, 15),    # CD14 negative
        "fl1_positive_frac": 0.0,
        "fl2_positive_frac": 0.0,
    },
    {
        "name": "debris",
        "fraction": 0.10,
        "fsc": (50, 30),
        "ssc": (40, 25),
        "fl1": (20, 10),
        "fl2": (20, 10),
        "fl1_positive_frac": 0.0,
        "fl2_positive_frac": 0.0,
    },
]


class FlowCytometrySim:
    """Flow cytometry simulation.

    Parameters
    ----------
    n_total : int
        Total number of cells in the sample tube.
    populations : list of dict, optional
        Population definitions. Defaults to a standard blood sample.
    events_per_snap : int
        Base events collected per snap (at 10x).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_total: int = 10000,
        populations: list = None,
        events_per_snap: int = 100,
        seed: int = 42,
        viewport_width: int = 512,
        viewport_height: int = 512,
    ):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.n_total = n_total
        self.events_per_snap = events_per_snap
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        # SimulationBridge interface
        self.mode = 0
        self.camera_offset = [0, 0]
        self.state_devices = {}
        self.current_objectiv = 10
        self._objectif_dict = {"10x": 10, "20x": 20, "40x": 40}

        # Generate the full cell sample
        self._populations = populations or DEFAULT_POPULATIONS
        self._generate_sample()

        # Collected events (accumulate over snaps)
        self._collected_fsc = []
        self._collected_ssc = []
        self._collected_fl1 = []
        self._collected_fl2 = []
        self._collected_pop = []  # population index per event
        self._n_collected = 0
        self._snap_count = 0

        # Current batch: positions + indices for co-registered channels
        self._batch_start = 0
        self._batch_end = 0
        self._batch_positions = []  # list of (x, y) per cell in batch

    def _generate_sample(self):
        """Generate the full cell sample with all scatter/fluorescence values."""
        rng = self._rng
        all_fsc, all_ssc, all_fl1, all_fl2, all_pop = [], [], [], [], []

        for pop_idx, pop in enumerate(self._populations):
            n = int(self.n_total * pop["fraction"])
            fsc_mean, fsc_std = pop["fsc"]
            ssc_mean, ssc_std = pop["ssc"]
            fl1_mean, fl1_std = pop["fl1"]
            fl2_mean, fl2_std = pop["fl2"]

            fsc = np.abs(rng.normal(fsc_mean, fsc_std, n))
            ssc = np.abs(rng.normal(ssc_mean, ssc_std, n))

            # Fluorescence: positive fraction gets high signal, rest gets background
            fl1 = np.abs(rng.normal(30, 10, n))  # background
            fl1_pos = rng.random(n) < pop["fl1_positive_frac"]
            fl1[fl1_pos] = np.abs(rng.normal(fl1_mean, fl1_std, fl1_pos.sum()))

            fl2 = np.abs(rng.normal(30, 10, n))  # background
            fl2_pos = rng.random(n) < pop["fl2_positive_frac"]
            fl2[fl2_pos] = np.abs(rng.normal(fl2_mean, fl2_std, fl2_pos.sum()))

            all_fsc.append(fsc)
            all_ssc.append(ssc)
            all_fl1.append(fl1)
            all_fl2.append(fl2)
            all_pop.append(np.full(n, pop_idx, dtype=int))

        self._sample_fsc = np.concatenate(all_fsc)
        self._sample_ssc = np.concatenate(all_ssc)
        self._sample_fl1 = np.concatenate(all_fl1)
        self._sample_fl2 = np.concatenate(all_fl2)
        self._sample_pop = np.concatenate(all_pop)

        # Shuffle
        perm = rng.permutation(len(self._sample_fsc))
        self._sample_fsc = self._sample_fsc[perm]
        self._sample_ssc = self._sample_ssc[perm]
        self._sample_fl1 = self._sample_fl1[perm]
        self._sample_fl2 = self._sample_fl2[perm]
        self._sample_pop = self._sample_pop[perm]

    # ── SimulationBridge interface ──

    def _update_mode(self):
        """Update rendering mode from state_devices.

        Supports both:
          - New Detector device: FSC-SSC / FL1-FITC / FL2-PE
          - Legacy LED/Filter Wheel: CYAN / ORANGE / RED
        """
        # New-style: Detector device
        det = self.state_devices.get("Detector", {})
        det_label = det.get("label", det.get("Label", ""))
        if det_label:
            if "FL2" in det_label or "PE" in det_label:
                self.mode = 2
            elif "FL1" in det_label or "FITC" in det_label:
                self.mode = 1
            else:
                self.mode = 0
            return

        # Legacy: LED + Filter Wheel
        led = self.state_devices.get("LED", {})
        led_label = led.get("label", led.get("Label", ""))
        fw = self.state_devices.get("Filter Wheel", {})
        fw_label = fw.get("label", fw.get("Label", ""))
        combined = led_label + " " + fw_label

        if "ORANGE" in combined or "mScarlet3" in combined:
            self.mode = 1
        elif "RED" in combined or "miRFP670" in combined:
            self.mode = 2
        else:
            self.mode = 0

    def _update_objectif(self):
        """Update objective from state_devices."""
        if "Objective" not in self.state_devices:
            return
        obj = self.state_devices["Objective"]
        lbl = obj.get("label", obj.get("Label", ""))
        if lbl in self._objectif_dict:
            self.current_objectiv = self._objectif_dict[lbl]

    def set_focal_plane(self, z):
        """No Z for flow cytometry."""
        pass

    def snap_frame(self, mask=None, exposure=50, intensity=100, **kwargs):
        """Collect events and return stream visualization.

        Each snap collects a batch of events from the sample tube.
        All three modes show the same cells at the same positions,
        allowing the agent to co-register measurements across channels.

        Returns a 512x512 image:
          - mode 0: brightfield stream (size=FSC, darkness=SSC)
          - mode 1: FL1-FITC fluorescence stream (CD3 marker)
          - mode 2: FL2-PE fluorescence stream (CD14 marker)
        """
        self._update_mode()
        self._update_objectif()

        # Determine events per snap based on objective
        obj_multiplier = {10: 1, 20: 2, 40: 5}
        n_events = self.events_per_snap * obj_multiplier.get(self.current_objectiv, 1)

        # Collect new events from the sample
        start = self._n_collected
        end = min(start + n_events, len(self._sample_fsc))
        if end > start:
            self._collected_fsc.extend(self._sample_fsc[start:end])
            self._collected_ssc.extend(self._sample_ssc[start:end])
            self._collected_fl1.extend(self._sample_fl1[start:end])
            self._collected_fl2.extend(self._sample_fl2[start:end])
            self._collected_pop.extend(self._sample_pop[start:end])
            self._n_collected = end

        # Generate deterministic cell positions for this batch
        self._batch_start = start
        self._batch_end = end
        self._batch_positions = self._generate_batch_positions(start, end)

        self._snap_count += 1

        if self.mode == 0:
            return self._render_stream_bf()
        elif self.mode == 1:
            return self._render_stream_fl1()
        elif self.mode == 2:
            return self._render_stream_fl2()
        return self._render_stream_bf()

    def _generate_batch_positions(self, start, end):
        """Generate deterministic cell positions for the current batch.

        Positions are seeded from event indices so the same cells
        appear at the same positions regardless of which channel
        is being rendered.
        """
        w, h = self.viewport_width, self.viewport_height
        ch_y0 = h // 4
        ch_y1 = 3 * h // 4
        y_center = (ch_y0 + ch_y1) / 2
        ch_h = ch_y1 - ch_y0

        positions = []
        for idx in range(start, end):
            # Deterministic RNG per cell (seeded from event index)
            cell_rng = np.random.default_rng(self._seed + 5000 + idx)
            x = cell_rng.uniform(20, w - 20)
            y = y_center + cell_rng.normal(0, ch_h * 0.15)
            y = np.clip(y, ch_y0 + 10, ch_y1 - 10)
            positions.append((float(x), float(y)))

        return positions

    def _draw_channel_background(self, img):
        """Draw the flow channel walls and laminar flow gradient."""
        w, h = self.viewport_width, self.viewport_height
        ch_y0 = h // 4
        ch_y1 = 3 * h // 4
        ch_h = ch_y1 - ch_y0

        for y in range(ch_y0, ch_y1):
            dist = abs(y - (ch_y0 + ch_h // 2)) / (ch_h // 2)
            shade = int(180 - 40 * (1 - dist ** 2))
            img[y, :] = shade

        cv2.line(img, (0, ch_y0), (w, ch_y0), 100, 2)
        cv2.line(img, (0, ch_y1), (w, ch_y1), 100, 2)

        # Laser line
        laser_x = w // 2
        cv2.line(img, (laser_x, ch_y0 + 5), (laser_x, ch_y1 - 5), 240, 1)

    def _render_stream_bf(self):
        """Render brightfield stream: cells sized by FSC, darkened by SSC."""
        w, h = self.viewport_width, self.viewport_height
        img = np.full((h, w), 200, dtype=np.uint8)

        self._draw_channel_background(img)
        img_f = img.astype(np.float32)

        for i, (x, y) in enumerate(self._batch_positions):
            idx = self._batch_start + i
            if idx >= len(self._sample_fsc):
                break

            fsc = self._sample_fsc[idx]
            ssc = self._sample_ssc[idx]

            # Size proportional to FSC (forward scatter = cell size)
            r = max(2, int(fsc / 60))
            # Darkness proportional to SSC (side scatter = granularity)
            gray = max(30, min(160, int(170 - ssc / 4)))

            # Phase-contrast halo
            cv2.circle(img_f, (int(x), int(y)), r + 2, 250, 2, cv2.LINE_AA)
            # Cell body
            cv2.circle(img_f, (int(x), int(y)), r, float(gray), -1, cv2.LINE_AA)

        img = np.clip(img_f, 0, 255).astype(np.uint8)
        return img

    def _render_stream_fl1(self):
        """Render FL1-FITC fluorescence stream.

        Dark background. Cells glow proportional to FL1 signal.
        FL1+ cells (CD3-FITC, T lymphocytes) are bright.
        FL1- cells are dim.
        """
        w, h = self.viewport_width, self.viewport_height
        img = np.zeros((h, w), dtype=np.float32)

        # Faint channel walls for orientation
        ch_y0, ch_y1 = h // 4, 3 * h // 4
        cv2.line(img, (0, ch_y0), (w, ch_y0), 30, 1)
        cv2.line(img, (0, ch_y1), (w, ch_y1), 30, 1)

        # Autofluorescence background in channel
        noise_rng = np.random.default_rng(self._seed + 3000 + self._snap_count)
        noise = noise_rng.normal(3, 1.5, (h, w)).astype(np.float32)
        mask = np.zeros((h, w), dtype=bool)
        mask[ch_y0:ch_y1, :] = True
        img[mask] = np.clip(noise[mask], 0, 10)

        for i, (x, y) in enumerate(self._batch_positions):
            idx = self._batch_start + i
            if idx >= len(self._sample_fl1):
                break

            fsc = self._sample_fsc[idx]
            fl1 = self._sample_fl1[idx]

            r = max(2, int(fsc / 60))
            # FL1 brightness: scale 0-700 → 0-255
            brightness = min(255, fl1 * 255 / 500)

            cv2.circle(img, (int(x), int(y)), r, float(brightness), -1, cv2.LINE_AA)
            # Slight halo for bright cells
            if brightness > 100:
                cv2.circle(img, (int(x), int(y)), r + 1,
                           float(brightness * 0.3), 1, cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_stream_fl2(self):
        """Render FL2-PE fluorescence stream.

        Dark background. Cells glow proportional to FL2 signal.
        FL2+ cells (CD14-PE, monocytes) are bright.
        FL2- cells are dim.
        """
        w, h = self.viewport_width, self.viewport_height
        img = np.zeros((h, w), dtype=np.float32)

        # Faint channel walls
        ch_y0, ch_y1 = h // 4, 3 * h // 4
        cv2.line(img, (0, ch_y0), (w, ch_y0), 30, 1)
        cv2.line(img, (0, ch_y1), (w, ch_y1), 30, 1)

        # Autofluorescence
        noise_rng = np.random.default_rng(self._seed + 4000 + self._snap_count)
        noise = noise_rng.normal(3, 1.5, (h, w)).astype(np.float32)
        mask = np.zeros((h, w), dtype=bool)
        mask[ch_y0:ch_y1, :] = True
        img[mask] = np.clip(noise[mask], 0, 10)

        for i, (x, y) in enumerate(self._batch_positions):
            idx = self._batch_start + i
            if idx >= len(self._sample_fl2):
                break

            fsc = self._sample_fsc[idx]
            fl2 = self._sample_fl2[idx]

            r = max(2, int(fsc / 60))
            # FL2 brightness: scale 0-700 → 0-255
            brightness = min(255, fl2 * 255 / 500)

            cv2.circle(img, (int(x), int(y)), r, float(brightness), -1, cv2.LINE_AA)
            if brightness > 100:
                cv2.circle(img, (int(x), int(y)), r + 1,
                           float(brightness * 0.3), 1, cv2.LINE_AA)

        return np.clip(img, 0, 255).astype(np.uint8)

    def reset_collection(self):
        """Reset collected events (start fresh acquisition)."""
        self._collected_fsc = []
        self._collected_ssc = []
        self._collected_fl1 = []
        self._collected_fl2 = []
        self._collected_pop = []
        self._n_collected = 0
        self._snap_count = 0
        self._batch_start = 0
        self._batch_end = 0
        self._batch_positions = []
        # Re-shuffle sample for fresh order
        rng = np.random.default_rng(self._seed + 200)
        perm = rng.permutation(len(self._sample_fsc))
        self._sample_fsc = self._sample_fsc[perm]
        self._sample_ssc = self._sample_ssc[perm]
        self._sample_fl1 = self._sample_fl1[perm]
        self._sample_fl2 = self._sample_fl2[perm]
        self._sample_pop = self._sample_pop[perm]

    # ── Ground truth ──

    def get_ground_truth(self):
        """Return ground truth for grading."""
        pop_counts = {}
        pop_fracs = {}
        for i, pop in enumerate(self._populations):
            name = pop["name"]
            count = int((self._sample_pop == i).sum())
            pop_counts[name] = count
            pop_fracs[name] = round(count / len(self._sample_pop), 3)

        # T-cell fraction: lymphocytes that are FL1+ (CD3+)
        lymph_mask = self._sample_pop == 0  # lymphocyte index
        n_lymph = lymph_mask.sum()
        # FL1+ threshold: above background (>100)
        fl1_pos = (self._sample_fl1 > 100) & lymph_mask
        t_cell_frac = float(fl1_pos.sum() / n_lymph) if n_lymph > 0 else 0.0

        return {
            "n_total": len(self._sample_fsc),
            "populations": pop_counts,
            "population_fractions": pop_fracs,
            "t_cell_fraction": round(t_cell_frac, 3),
            "n_populations": len([p for p in self._populations if p["fraction"] > 0.01]),
            "pop_fsc_means": {
                pop["name"]: round(float(self._sample_fsc[self._sample_pop == i].mean()), 1)
                for i, pop in enumerate(self._populations)
            },
            "pop_ssc_means": {
                pop["name"]: round(float(self._sample_ssc[self._sample_pop == i].mean()), 1)
                for i, pop in enumerate(self._populations)
            },
        }

    def get_collected_data(self):
        """Return currently collected event data (for advanced grading)."""
        return {
            "n_collected": self._n_collected,
            "n_snaps": self._snap_count,
            "fsc": list(self._collected_fsc),
            "ssc": list(self._collected_ssc),
            "fl1": list(self._collected_fl1),
            "fl2": list(self._collected_fl2),
            "pop": list(self._collected_pop),
        }
