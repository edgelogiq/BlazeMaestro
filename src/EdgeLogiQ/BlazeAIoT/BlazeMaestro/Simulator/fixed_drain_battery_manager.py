"""Fixed-drain battery manager implementation (one class per file)."""
from __future__ import annotations
from typing import Callable, Dict, Optional
from src.EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.battery_manager import BatteryManager


class FixedDrainBatteryManager(BatteryManager):
    """Simple battery manager using a fixed drain model (P * t).

    - Tracks remaining Joules per host.
    - Optional power-off callback invoked on depletion.
    """
    def __init__(self) -> None:
        self._remaining_j: Dict[str, float] = {}
        self._poweroff_cb: Optional[Callable[[str], None]] = None

    def set_poweroff_fn(self, fn: Callable[[str], None]) -> None:
        """Set a callback to power off a host when depleted."""
        self._poweroff_cb = fn

    # BatteryManager API
    def register(self, host_id: str, capacity_joules: float) -> None:
        self._remaining_j[host_id] = float(capacity_joules)

    def has_host(self, host_id: str) -> bool:
        return host_id in self._remaining_j

    def on_task(self, host_id: str, duration_s: float, power_watts: float) -> None:
        if host_id not in self._remaining_j:
            return
        energy = float(power_watts) * float(duration_s)  # Joules
        self._remaining_j[host_id] -= energy
        if self._remaining_j[host_id] <= 0:
            self._remaining_j[host_id] = 0.0
            if self._poweroff_cb:
                try:
                    self._poweroff_cb(host_id)
                except Exception:
                    pass

    def on_idle(self, host_id: str, duration_s: float, power_watts_idle: float) -> None:
        if host_id not in self._remaining_j:
            return
        energy = float(power_watts_idle) * float(duration_s)
        self._remaining_j[host_id] = max(0.0, self._remaining_j[host_id] - energy)
        if self._remaining_j[host_id] <= 0 and self._poweroff_cb:
            try:
                self._poweroff_cb(host_id)
            except Exception:
                pass

    def remaining(self, host_id: str) -> float:
        return float(self._remaining_j.get(host_id, 0.0))

    def is_depleted(self, host_id: str) -> bool:
        return self._remaining_j.get(host_id, 0.0) <= 0.0
