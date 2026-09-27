"""Battery management abstractions for mobile hosts.

This file intentionally contains a single class (one-class-per-file policy).
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class BatteryManager(ABC):
    """Abstract interface for battery management.

    Implementations should track remaining energy per host and expose
    hooks to update charge on events (e.g., when a task is scheduled).
    """
    @abstractmethod
    def register(self, host_id: str, capacity_joules: float) -> None:
        """Register a host with initial battery capacity in Joules."""

    @abstractmethod
    def has_host(self, host_id: str) -> bool:
        """Return True if the host is managed by the battery system."""

    @abstractmethod
    def on_task(self, host_id: str, duration_s: float, power_watts: float) -> None:
        """Record the energy usage of a task on a host: E = P * t."""

    @abstractmethod
    def on_idle(self, host_id: str, duration_s: float, power_watts_idle: float) -> None:
        """Account for idle drain over a time window."""

    @abstractmethod
    def remaining(self, host_id: str) -> float:
        """Return remaining energy in Joules for a host (0 if unknown)."""

    @abstractmethod
    def is_depleted(self, host_id: str) -> bool:
        """Return True if the host's remaining energy is depleted (<= 0)."""
