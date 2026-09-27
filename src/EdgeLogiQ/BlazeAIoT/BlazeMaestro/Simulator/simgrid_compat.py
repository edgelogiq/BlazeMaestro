"""Compatibility layer for simgrid imports.

If simgrid is available, re-export its symbols. Otherwise, provide safe no-ops
so the simulator can run without SimGrid (latency=0, no energy exec).
"""
from __future__ import annotations

SIMGRID_AVAILABLE = True
__all__ = [
    "Engine",
    "Actor",
    "Host",
    "this_actor",
    "SIMGRID_AVAILABLE",
]

try:
    from simgrid import Engine, Actor, Host, this_actor  # type: ignore
except Exception:  # pragma: no cover - environment-dependent
    SIMGRID_AVAILABLE = False

    class _DummyHost:
        def __init__(self, name: str):
            self._name = name

        def route_to(self, other: "_DummyHost"):
            return ([], 0.0)

        def turn_off(self):
            # No-op in compat mode
            return None

        @property
        def name(self):
            return self._name

        # Provide speed-related APIs used by compute actors
        def speed(self):  # flop/s equivalent
            return 0.0

        def pstate(self):
            return 0

        def pstate_speed(self, _p):
            return 0.0

    class Host:  # type: ignore
        @staticmethod
        def by_name(name: str) -> _DummyHost:
            return _DummyHost(name)

    class Engine:  # type: ignore
        def __init__(self, argv=None):
            pass

        def load_platform(self, path: str):
            return None

        def run(self):
            # Nothing to run in compat mode
            return None

    class Actor:  # type: ignore
        @staticmethod
        def create(name: str, host: _DummyHost, func, *args, **kwargs):
            # No-op to avoid calling SimGrid-specific actor code
            return None

    class this_actor:  # type: ignore
        @staticmethod
        def get_host():
            return Host.by_name("compat-host")

        @staticmethod
        def execute(flops: float):
            # No-op in compat mode
            return None
