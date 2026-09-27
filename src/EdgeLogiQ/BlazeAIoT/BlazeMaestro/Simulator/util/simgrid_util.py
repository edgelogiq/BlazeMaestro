from __future__ import annotations

"""
Shared SimGrid Engine utilities.

Provides a single place to initialize and reuse a SimGrid Engine across scripts
to avoid reinitialization issues and potential segfaults.
"""

import os
import atexit
import traceback
from .. import simgrid_compat as sg


class SimgridUtil:
    _shared_engine = None

    @staticmethod
    def get_engine(platform_xml_path: str):
        """Get or create a shared SimGrid Engine.

        Returns the Engine instance if SimGrid is available and the platform
        file exists; otherwise returns None.
        """
        # Reuse if already created
        if SimgridUtil._shared_engine is not None:
            return SimgridUtil._shared_engine

        sim_ok = getattr(sg, "SIMGRID_AVAILABLE", False)
        print(f"[SIMGRID] Requested engine init. SIMGRID_AVAILABLE={sim_ok}")
        if not sim_ok:
            return None

        if not os.path.exists(platform_xml_path):
            print(f"[SIMGRID][ERROR] Platform XML not found: {platform_xml_path}")
            return None

        try:
            argv = [f"simgrid_{os.getpid()}", "--cfg=contexts/stack-size:131072"]
            print(f"[SIMGRID] Initializing Engine with argv={argv} platform='{platform_xml_path}'")
            eng = sg.Engine(argv)
            eng.load_platform(platform_xml_path)
            print("[SIMGRID] Engine created and platform loaded once.")
            SimgridUtil._shared_engine = eng
            atexit.register(SimgridUtil.cleanup_engine)
            return eng
        except Exception as e:  # pragma: no cover - environment-dependent
            print(f"[SIMGRID][EXCEPTION] Engine init/load failed: {e}")
            traceback.print_exc()
            SimgridUtil._shared_engine = None
            return None

    @staticmethod
    def _get_simgrid_engine(platform_xml_path: str):
        """Compatibility alias for older code paths."""
        return SimgridUtil.get_engine(platform_xml_path)

    @staticmethod
    def cleanup_engine():
        """Explicitly clean up the shared SimGrid engine."""
        if SimgridUtil._shared_engine is not None:
            # Deleting the object is the intended way to trigger cleanup in bindings
            del SimgridUtil._shared_engine
            SimgridUtil._shared_engine = None
