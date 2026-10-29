"""Compatibility layer for simgrid imports.

If simgrid is available, re-export its Engine. Otherwise, provide a no-op Engine
so the simulator can run without SimGrid.
"""
from __future__ import annotations

SIMGRID_AVAILABLE = True
__all__ = [
    "Engine",
    "SIMGRID_AVAILABLE",
]

try:
    from simgrid import Engine  # type: ignore
except Exception:  # pragma: no cover - environment-dependent
    SIMGRID_AVAILABLE = False

    class Engine:  # type: ignore
        def __init__(self, argv=None):
            pass

        def load_platform(self, path: str):
            return None

        def run(self):
            # Nothing to run in compat mode
            return None
