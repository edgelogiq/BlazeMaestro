"""
SimPy-only Node class representing a machine in a cluster.
"""
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.simpy_resource_mixin import SimPyResourceMixin
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node
import simpy
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..Metrics.base import BaseMetricsFacade


class NodeSimulator(Node, SimPyResourceMixin):
    """
    SimPy-specific Node implementation inherits from Node and adds SimPy resource logic.
    """
    def __init__(
        self,
        env,
        node_id,
        cpu,
        ram,
        gpu,
        gpu_shared: bool = False,
        energy_level=None,
        metrics_facade: Optional['BaseMetricsFacade'] = None,
        performance: float = 1.0,
        role: Optional[str] = None,
        tier: Optional[str] = None,
        zone: Optional[str] = None,
        accelerator: Optional[str] = None,
        workload_type: Optional[str] = None,
        affinity_group: Optional[str] = None,
        schedulable: bool = True,
    ):
        # Call Node.__init__ without env (Node doesn't take env)
        Node.__init__(
            self,
            node_id,
            cpu,
            ram,
            gpu,
            gpu_shared,
            energy_level,
            metrics_facade,
            performance,
            role=role,
            tier=tier,
            zone=zone,
            accelerator=accelerator,
            workload_type=workload_type,
            affinity_group=affinity_group,
            schedulable=schedulable,
        )
        self.env = env
        self._init_simpy_resources(env, cpu, ram)
        if gpu > 0:
            self.gpu = simpy.Container(env, init=gpu, capacity=gpu)
        else:
            self.gpu = None
        self.processes = []

    def __str__(self):
        # Optionally, override string representation for SimPy-specific details
        return super().__str__()
