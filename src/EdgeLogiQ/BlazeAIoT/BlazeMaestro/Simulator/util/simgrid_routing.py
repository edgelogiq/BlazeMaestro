"""Inter-node latency matrix computed by SimGrid's routing.

SimGrid resolves the route between every pair of hosts of the platform (zone routing,
e.g. Floyd, through routers) and returns the end-to-end latency of that route (sum of
the latencies of its links). A SimGrid Engine can only be created once per process, so
the computation runs in a short-lived child process; results are cached per platform.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CACHE: Dict[Tuple[str, float, Tuple[str, ...]], np.ndarray] = {}


def simgrid_available() -> bool:
    try:
        import simgrid  # noqa: F401
        return True
    except Exception:
        return False


def _worker(platform_xml: str, node_ids: List[str], loop_latency: Dict[str, float], conn) -> None:
    try:
        from simgrid import Engine
        engine = Engine(["blazemaestro-routing", "--log=root.thres:critical"])
        engine.load_platform(platform_xml)
        engine.netzone_root.seal()  # compute the zone routing tables (Floyd/Dijkstra/...)
        hosts = {h.name: h for h in engine.all_hosts}
        n = len(node_ids)
        mat = np.zeros((n, n), dtype=float)
        for i, a in enumerate(node_ids):
            for j, b in enumerate(node_ids):
                if i == j:
                    # Same convention as the XML parser: optional "<node>-loop" link.
                    mat[i, j] = loop_latency.get(a, 0.0)
                    continue
                _, latency = hosts[a].route_to(hosts[b])
                mat[i, j] = float(latency)
        conn.send(("ok", mat))
    except Exception as exc:  # pragma: no cover - environment-dependent
        conn.send(("error", repr(exc)))
    finally:
        conn.close()


def latency_matrix(platform_xml: str, node_ids: Sequence[str],
                   loop_latency: Optional[Dict[str, float]] = None,
                   timeout: float = 120.0) -> Optional[np.ndarray]:
    """Route latencies (seconds) between the given hosts, or None if SimGrid is unavailable
    or fails (callers then fall back to the XML shortest-path computation)."""
    if not simgrid_available():
        return None
    path = os.path.abspath(platform_xml)
    key = (path, os.path.getmtime(path), tuple(node_ids))
    if key in _CACHE:
        return _CACHE[key].copy()
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(path, list(node_ids), dict(loop_latency or {}), child))
    proc.start()
    child.close()
    result = None
    try:
        if parent.poll(timeout):
            status, payload = parent.recv()
            if status == "ok":
                result = payload
    except EOFError:  # SimGrid aborted (e.g. invalid platform): fall back
        result = None
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
    if result is not None:
        _CACHE[key] = result
        return result.copy()
    return None
