from __future__ import annotations

"""Network utilities: latency matrix builders (SimGrid and XML fallback).

Centralized helpers to compute pairwise network latencies between nodes,
with caching to reduce repeated native calls.
"""

import xml.etree.ElementTree as ET
import os
import numpy as np
from typing import Dict, Tuple, Set, List, Any, Union

from ..network_jitter import NetworkJitterModel
from ..latency_matrix_with_jitter import LatencyMatrixWithJitter

# Module-level cache to avoid repeated SimGrid route queries
_latency_cache: Dict[Tuple[str, str], float] = {}
_latency_cache_stats_printed = False


def _parse_latency_value(lat: str) -> float:
    """Convert latency string to float seconds. Supports ms, s, us, or unitless."""
    try:
        if lat.endswith("ms"):
            return float(lat[:-2]) / 1000.0
        elif lat.endswith("us"):
            return float(lat[:-2]) / 1_000_000.0
        elif lat.endswith("s"):
            return float(lat[:-1])
        else:
            return float(lat)
    except Exception:
        return 0.0


def _parse_links_from_xml(root: ET.Element) -> Tuple[Dict[str, float], Set[str]]:
    """Parse <link> elements for latency and wifi links."""
    link_latency: Dict[str, float] = {}
    wifi_links: Set[str] = set()
    for link in root.iter("link"):
        lid = link.get("id")
        lat = link.get("latency", "0s")
        if lid and "wifi" in lid.lower():
            wifi_links.add(lid)
        link_latency[lid] = _parse_latency_value(lat) if lid else 0.0
    return link_latency, wifi_links


def _parse_routes_from_xml(
    root: ET.Element, link_latency: Dict[str, float]
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[Tuple[str, str], List[str]]]:
    """Parse <route> elements for graph and route link mapping."""
    from collections import defaultdict

    graph = defaultdict(list)
    route_links: Dict[Tuple[str, str], List[str]] = {}
    for route in root.iter("route"):
        src = route.get("src")
        dst = route.get("dst")
        if not src or not dst:
            continue
        hop = 0.0
        route_link_ids: List[str] = []
        for lctn in route.iter("link_ctn"):
            lid = lctn.get("id")
            if lid is None:
                continue
            hop += float(link_latency.get(lid, 0.0))
            route_link_ids.append(lid)
        graph[src].append((dst, hop))
        graph[dst].append((src, hop))
        route_links[(src, dst)] = route_link_ids
    return graph, route_links


def _dijkstra(graph: Dict[str, List[Tuple[str, float]]], src: str) -> Dict[str, float]:
    """Dijkstra's algorithm for shortest path latencies from src."""
    import heapq

    dist = {src: 0.0}
    pq = [(0.0, src)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            nd = d + float(w)
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _routed_by_simgrid(platform_xml_path: str, node_ids: List[str],
                       link_latency: Dict[str, float], fallback: np.ndarray) -> np.ndarray:
    """Base latency matrix from SimGrid's routing when the bindings are installed.

    Falls back to the XML shortest-path matrix when SimGrid is missing or fails, or when
    BLAZE_LATENCY_BACKEND=dijkstra is set.
    """
    if os.environ.get("BLAZE_LATENCY_BACKEND", "simgrid").lower() == "dijkstra":
        return fallback
    from .simgrid_routing import latency_matrix
    loops = {nid: float(link_latency[f"{nid}-loop"]) for nid in node_ids if f"{nid}-loop" in link_latency}
    routed = latency_matrix(platform_xml_path, node_ids, loops)
    return routed if routed is not None else fallback


class NetworkUtil:
    """Namespace for network-related utilities (static methods)."""

    @staticmethod
    def build_latency_from_platform_xml(platform_xml_path: str, nodes: List[Any]) -> np.ndarray:
        """Parse platform.xml routes/links to build a pairwise latency matrix.

        Supports units: ms, s, us; defaults to float seconds if unitless.
        Includes optional loopback links named "<node_id>-loop".
        """
        tree = ET.parse(platform_xml_path)
        root = tree.getroot()
        link_latency, _ = _parse_links_from_xml(root)
        graph, _ = _parse_routes_from_xml(root, link_latency)
        node_ids = [getattr(n, "node_id", "") for n in nodes]
        idx_by_id = {nid: i for i, nid in enumerate(node_ids)}
        n = len(nodes)
        mat = np.zeros((n, n), dtype=float)
        for nid in node_ids:
            if nid not in idx_by_id:
                continue
            src_i = idx_by_id[nid]
            dist = _dijkstra(graph, nid)
            for vid, dj in dist.items():
                j = idx_by_id.get(vid)
                if j is not None:
                    if nid == vid:
                        loopback_id = f"{nid}-loop"
                        mat[src_i, j] = float(link_latency.get(loopback_id, 0.0))
                    else:
                        mat[src_i, j] = float(dj)
        return _routed_by_simgrid(platform_xml_path, node_ids, link_latency, mat)

    @staticmethod
    def build_latency_matrix_with_jitter(
        platform_xml_path: str,
        nodes: List[Any],
        *,
        enable_jitter: bool = True,
        jitter_percent: float = 10.0,
        enable_packet_loss: bool = False,
        packet_loss_rate: float = 0.01,
        seed: int | None = None,
    ) -> Union[np.ndarray, LatencyMatrixWithJitter]:
        """Build latency matrix with optional jitter and packet loss modeling.

        When enable_jitter is True, returns a LatencyMatrixWithJitter using:
          - base latency matrix from platform.xml
          - link_mapping for each (src,dst) pair (primary link id)
          - wifi_links set for increased jitter via model
        Otherwise, returns the plain latency matrix (np.ndarray).
        """
        if not enable_jitter:
            return NetworkUtil.build_latency_from_platform_xml(platform_xml_path, nodes)
        tree = ET.parse(platform_xml_path)
        root = tree.getroot()
        link_latency, wifi_links = _parse_links_from_xml(root)
        graph, route_links = _parse_routes_from_xml(root, link_latency)
        node_ids = [getattr(n, "node_id", "") for n in nodes]
        idx_by_id = {nid: i for i, nid in enumerate(node_ids)}
        n = len(nodes)
        mat = np.zeros((n, n), dtype=float)
        link_mapping: Dict[Tuple[int, int], str] = {}
        for nid in node_ids:
            if nid not in idx_by_id:
                continue
            src_i = idx_by_id[nid]
            dist = _dijkstra(graph, nid)
            for vid, dj in dist.items():
                j = idx_by_id.get(vid)
                if j is None:
                    continue
                if nid == vid:
                    loopback_id = f"{nid}-loop"
                    mat[src_i, j] = float(link_latency.get(loopback_id, 0.0))
                    link_mapping[(src_i, j)] = loopback_id
                else:
                    mat[src_i, j] = float(dj)
                    links = route_links.get((nid, vid), route_links.get((vid, nid), []))
                    link_mapping[(src_i, j)] = links[0] if links else f"{nid}-{vid}"
        mat = _routed_by_simgrid(platform_xml_path, node_ids, link_latency, mat)
        jitter_model = NetworkJitterModel(
            base_jitter_percent=jitter_percent,
            wifi_jitter_multiplier=2.0,
            congestion_enabled=True,
            packet_loss_enabled=enable_packet_loss,
            packet_loss_rate=packet_loss_rate,
            seed=seed,
        )

        return LatencyMatrixWithJitter(mat, link_mapping, wifi_links, jitter_model)
