from typing import Dict, Optional, Any, TYPE_CHECKING
from BlazeAIoT.Core.Utils import BlazeLogger

if TYPE_CHECKING:
    from ..Metrics.base import BaseMetricsFacade
    from ..Topology.node import Node

class NodeMetricsWrapper:
    """Wrapper that provides node-specific metrics interface using a shared metrics facade."""

    def __init__(self, node: 'Node', facade: Optional['BaseMetricsFacade'] = None):
        self._node = node
        self._facade = facade

    def set_facade(self, facade: 'BaseMetricsFacade'):
        """Set or update the metrics facade."""
        self._facade = facade

    def get_capacity(self) -> Dict[str, float]:
        """Return capacity metrics for this node."""
        if self._facade:
            return self._facade.get_node_capacity(self._node.node_id)
        # Fallback to direct access if no facade is set
        # Support both base Node and SimPyNode
        cpu_container = getattr(self._node, "cpu_container", None)
        if cpu_container is not None:
            cpu = float(getattr(cpu_container, "capacity", 0.0))
        else:
            cpu = float(getattr(self._node, "cpu", 0.0))
        ram_container = getattr(self._node, "ram_container", None)
        if ram_container is not None:
            ram = float(getattr(ram_container, "capacity", 0.0))
        else:
            ram = float(getattr(self._node, "ram", 0.0))
        gpu = float(getattr(getattr(self._node, "gpu", None), "capacity", 0.0)) if getattr(self._node, "gpu",
                                                                                           None) else 0.0
        return {"cpu": cpu, "ram_gb": ram, "gpu_gb": gpu}

    def get_available(self) -> Dict[str, float]:
        """Return currently available resources for this node."""
        if self._facade:
            return self._facade.get_node_available(self._node.node_id)
        # Fallback to direct access if no facade is set
        cpu_container = getattr(self._node, "cpu_container", None)
        if cpu_container is not None:
            cpu_avail = float(getattr(cpu_container, "level", 0.0))
        else:
            cpu_avail = float(getattr(self._node, "cpu", 0.0))
        ram_container = getattr(self._node, "ram_container", None)
        if ram_container is not None:
            ram_avail = float(getattr(ram_container, "level", 0.0))
        else:
            ram_avail = float(getattr(self._node, "ram", 0.0))
        gpu_avail = float(getattr(getattr(self._node, "gpu", None), "level", 0.0)) if getattr(self._node, "gpu",
                                                                                              None) else 0.0
        return {"cpu": cpu_avail, "ram_gb": ram_avail, "gpu_gb": gpu_avail}

    def get_property(self, key: str) -> Optional[Any]:
        """Get optional extra properties, e.g., gpu_shared, energy, labels, etc."""
        if self._facade:
            return self._facade.get_node_property(self._node.node_id, key)
        # Fallback to direct access if no facade is set
        return getattr(self._node, key, None)