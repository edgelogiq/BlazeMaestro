"""
Snapshot metrics facade.

Provides read-only metrics from a precomputed snapshot so orchestrators
never perform live queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .base import BaseMetricsFacade


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable metrics snapshot for a single fetch cycle."""

    timestamp: float
    node_ids: List[str]
    node_capacity: Dict[str, Dict[str, float]]
    node_available: Dict[str, Dict[str, float]]
    node_properties: Dict[str, Dict[str, Any]]
    latency_matrix: Optional[List[List[float]]]


class SnapshotMetricsFacade(BaseMetricsFacade):
    """Facade that serves metrics from a precomputed snapshot."""

    def __init__(self, snapshot: MetricsSnapshot):
        self._snapshot = snapshot
        self._node_index = {node_id: idx for idx, node_id in enumerate(snapshot.node_ids)}

    def get_node_capacity(self, node_id: str) -> Dict[str, float]:
        return dict(self._snapshot.node_capacity.get(node_id, {}))

    def get_node_available(self, node_id: str) -> Dict[str, float]:
        return dict(self._snapshot.node_available.get(node_id, {}))

    def get_node_property(self, node_id: str, key: str) -> Optional[Any]:
        return self._snapshot.node_properties.get(node_id, {}).get(key)

    def get_latency_matrix(self, nodes: Optional[Sequence[Any]] = None) -> Any:
        if self._snapshot.latency_matrix is None:
            return None
        if not nodes:
            return self._snapshot.latency_matrix

        # Reorder matrix to match requested node order
        size = len(nodes)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for i, src in enumerate(nodes):
            src_id = str(getattr(src, "node_id", ""))
            src_idx = self._node_index.get(src_id)
            if src_idx is None:
                continue
            for j, dst in enumerate(nodes):
                dst_id = str(getattr(dst, "node_id", ""))
                dst_idx = self._node_index.get(dst_id)
                if dst_idx is None:
                    continue
                try:
                    matrix[i][j] = float(self._snapshot.latency_matrix[src_idx][dst_idx])
                except Exception:
                    matrix[i][j] = 0.0
        return matrix
