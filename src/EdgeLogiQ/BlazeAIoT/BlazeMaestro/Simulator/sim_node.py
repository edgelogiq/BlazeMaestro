from .simpy_resource_mixin import SimPyResourceMixin# Base Node (no SimPy logic)
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..Metrics.base import BaseMetricsFacade

import simpy


# SimPyNode: simulation-specific node with SimPy resource logic
class SimPyNode(SimPyResourceMixin, Node):
    def __init__(self, env, node_id, cpu, ram, gpu, gpu_shared: bool = False, energy_level=None, metrics_facade: Optional['BaseMetricsFacade'] = None, performance: float = 1.0):
        Node.__init__(self, node_id, cpu, ram, gpu, gpu_shared, energy_level, metrics_facade, performance)
        self.env = env
        self._init_simpy_resources(env, cpu, ram)
        if gpu > 0:
            self.gpu = simpy.Container(env, init=gpu, capacity=gpu)
        else:
            self.gpu = None
        self.processes = []

    def __str__(self):
        gpu_level = self.gpu.level if self.gpu else 0
        gpu_capacity = self.gpu.capacity if self.gpu else 0
        mem_type = "shared" if getattr(self, "gpu_shared", False) else "dedicated"
        return (
            f"SimPyNode(id={self.node_id}, cpu_used={self.cpu_container.capacity - self.cpu_container.level:.2f}/{self.cpu_container.capacity:.2f}, "
            f"ram_used={self.ram_container.capacity - self.ram_container.level:.2f}/{self.ram_container.capacity:.2f}, "
            f"gpu_used={gpu_capacity - gpu_level:.2f}/{gpu_capacity}({mem_type}), "
            f"energy={self.energy_level})"
        )
