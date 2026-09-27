from __future__ import annotations
from BlazeAIoT.Core.Utils import BlazeLogger

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseMetricsFacade(ABC):
    """Abstract facade for node metrics.

    Orchestrators should depend on this interface instead of simulator objects directly.
    """

    @abstractmethod
    def get_node_capacity(self, node_id: str) -> Dict[str, float]:
        """Return capacity metrics for a node.

        Keys: cpu, ram_gb, gpu_gb
        """

    @abstractmethod
    def get_node_available(self, node_id: str) -> Dict[str, float]:
        """Return currently available resources for a node.

        Keys: cpu, ram_gb, gpu_gb
        """

    @abstractmethod
    def get_node_property(self, node_id: str, key: str) -> Optional[Any]:
        """Optional extra properties, e.g., gpu_shared, energy, labels, etc."""

    @abstractmethod
    def get_latency_matrix(self, nodes: list) -> Any:
        """Return network latency matrix between nodes.

        Args:
            nodes: List of Node objects

        Returns:
            2D numpy array where latency_matrix[i][j] is the latency from node i to node j
        """

