"""Compare SimGrid-routed latencies with the XML shortest-path (Dijkstra) matrix.

For each platform: build the host list from the platform XML, compute both matrices for
all ordered pairs and report the maximum absolute difference and the differing pairs.

Usage:
    python src/campaign/compare_simgrid_latency.py src/config/platform_sim.xml out/campaign/scal/inst/*/platform.xml
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util import network_util
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.simgrid_routing import latency_matrix


class _Node:
    def __init__(self, node_id):
        self.node_id = node_id


def main(paths):
    worst = 0.0
    rows = []
    for path in paths:
        ids = [h.get("id") for h in ET.parse(path).getroot().iter("host")]
        nodes = [_Node(i) for i in ids]
        os.environ["BLAZE_LATENCY_BACKEND"] = "dijkstra"
        dij = network_util.NetworkUtil.build_latency_from_platform_xml(path, nodes)
        root = ET.parse(path).getroot()
        link_latency, _ = network_util._parse_links_from_xml(root)
        loops = {i: float(link_latency[f"{i}-loop"]) for i in ids if f"{i}-loop" in link_latency}
        sg = latency_matrix(path, ids, loops)
        if sg is None:
            rows.append((path, len(ids), None, []))
            continue
        diff = np.abs(sg - dij)
        bad = [(ids[i], ids[j], dij[i, j], sg[i, j]) for i, j in zip(*np.where(diff > 1e-9))]
        worst = max(worst, float(diff.max()))
        rows.append((path, len(ids), float(diff.max()), bad))
    for path, n, d, bad in rows:
        status = "SimGrid indisponible" if d is None else f"écart max = {d:.3e} s, paires différentes = {len(bad)}"
        print(f"{path} ({n} hôtes, {n * (n - 1)} paires) : {status}")
        for a, b, x, y in bad[:10]:
            print(f"    {a} -> {b} : Dijkstra {x * 1000:.3f} ms, SimGrid {y * 1000:.3f} ms")
    print(f"\n{len(rows)} plateformes, écart maximal global = {worst:.3e} s")


if __name__ == "__main__":
    main(sys.argv[1:])
