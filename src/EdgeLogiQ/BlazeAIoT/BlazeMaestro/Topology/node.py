"""
Node class representing a machine in a cluster.
"""
from typing import Optional, TYPE_CHECKING
from ..Metrics.node_metrics_wrapper import NodeMetricsWrapper
if TYPE_CHECKING:
    from ..Metrics.base import BaseMetricsFacade

class Node:
    """
    Represents a machine in a cluster (framework-agnostic, no SimPy logic).
    """
    def __init__(self, node_id, cpu, ram, gpu, gpu_shared: bool = False, energy_level=None, metrics_facade: Optional['BaseMetricsFacade'] = None, performance: float = 1.0, role: Optional[str] = None, tier: Optional[str] = None, zone: Optional[str] = None, accelerator: Optional[str] = None, workload_type: Optional[str] = None, affinity_group: Optional[str] = None, schedulable: bool = True):
        self.node_id = node_id
        self.cpu = cpu
        self.ram = ram
        self.gpu = gpu if gpu > 0 else None
        self.gpu_shared = gpu_shared
        self.energy_level = energy_level
        self.metrics = NodeMetricsWrapper(self, metrics_facade)
        self.performance = performance
        # Optional node affinity/label attributes for service selector matching
        self.role = role  # e.g., "master", "worker", "gpu-node"
        self.tier = tier  # e.g., "edge", "fog", "cloud"
        self.zone = zone  # e.g., "us-west-1a", "zone-1"
        self.accelerator = accelerator  # e.g., "gpu", "tpu", "fpga"
        self.workload_type = workload_type  # e.g., "compute", "memory", "io"
        self.affinity_group = affinity_group  # e.g., "group-a", "group-b"
        self.schedulable = schedulable  # Whether workloads can be scheduled on this node

    def __str__(self):
        gpu_capacity = self.gpu if self.gpu else 0
        mem_type = "shared" if getattr(self, "gpu_shared", False) else "dedicated"
        return (
            f"Node(id={self.node_id}, cpu={self.cpu}, ram={self.ram}, gpu={gpu_capacity}({mem_type}), energy={self.energy_level})"
        )
