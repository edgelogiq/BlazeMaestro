"""Run src/train.py for one training seed without the SimGrid native engine.

train.py instantiates a SimGrid engine at start-up but never uses it afterwards:
latencies come from NetworkUtil.build_latency_matrix_with_jitter, which falls back
to shortest paths over the platform XML links. This wrapper lets the training run
on machines where the SimGrid Python bindings cannot be built (no CMake/Boost).

Usage (same arguments as train.py):
    PYTHONPATH=src python src/campaign/train_seed.py --seed 1 --export-dir out/campaign/train/seed1 ...
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util import simgrid_util

_orig_get_engine = simgrid_util.SimgridUtil.get_engine


def _get_engine(*args, **kwargs):
    try:
        engine = _orig_get_engine(*args, **kwargs)
    except Exception:
        engine = None
    return engine if engine is not None else object()


simgrid_util.SimgridUtil.get_engine = staticmethod(_get_engine)

sys.argv = [os.path.join(SRC, "train.py")] + sys.argv[1:]
runpy.run_path(os.path.join(SRC, "train.py"), run_name="__main__")
