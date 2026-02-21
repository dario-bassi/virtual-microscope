"""NewExperimentDevice — Allows resetting / recreating simulations.

A pymmcore ``GenericDevice`` with properties:
  - ``Action``: "Ready" (default), "Reset" (soft reset), "New" (full recreate)
  - ``Seed``: Random seed (0 = random)

When Action is set to "Reset" → calls bridge.reset_simulation(seed).
When Action is set to "New"   → calls bridge.recreate_simulation(seed).
Action auto-returns to "Ready" after each operation.
"""

import logging

from pymmcore_plus.experimental.unicore import GenericDevice
import virtual_microscope.engine.simulation_bridge as bridge_module

logger = logging.getLogger(__name__)


class NewExperimentDevice(GenericDevice):
    """Device for resetting or recreating simulations.

    Properties:
        Action: "Ready" | "Reset" | "New"
        Seed:   Integer seed (0 = random)
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_property("Action", default_value="Ready")
        self.register_property("Seed", default_value="0")

    def initialize(self) -> None:
        bridge_module.bridge_ready.wait(timeout=5.0)

    def set_property_value(self, prop_name: str, value: str) -> None:
        super().set_property_value(prop_name, value)

        if prop_name == "Action" and value in ("Reset", "New"):
            seed_str = self.get_property_value("Seed")
            seed = int(seed_str) if seed_str else 0
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)

            bridge = bridge_module.GLOBAL_BRIDGE
            if bridge is None:
                logger.warning("NewExperimentDevice: no bridge available")
                return

            if value == "Reset":
                bridge.reset_simulation(seed)
                logger.info("NewExperimentDevice: reset with seed=%d", seed)
            elif value == "New":
                bridge.recreate_simulation(seed)
                logger.info("NewExperimentDevice: recreated with seed=%d", seed)

            # Auto-return to Ready
            super().set_property_value("Action", "Ready")

    def shutdown(self) -> None:
        pass
