"""
Service Allocation module for BlazeMaestro.

Provides Facade pattern implementation for service allocation across
Kubernetes clusters and simulation environments.

Key components:
- BaseAllocator: Abstract interface for allocation implementations
- KubernetesAllocator: Kubernetes implementation using kubectl
- SimulatorAllocator: In-memory simulator implementation
- ServiceAllocationFacade: Unified facade for both backends
- AllocationEvent: Event model for broker integration

Usage:
    # Initialize for Kubernetes
    facade = ServiceAllocationFacade.create_for_kubernetes(
        context='prod-edge'
    )
    
    # Or for Simulator
    facade = ServiceAllocationFacade.create_for_simulator()
    
    # Perform allocation
    success = facade.allocate(
        service_id='vision-service',
        target_node='edge-gpu1',
        target_layer='edge',
        source_node='fog-cpu1',
        source_layer='fog',
        namespace='workloads'
    )
"""

from .allocation_event import AllocationEvent, AllocationEventType
from .base_allocator import BaseAllocator
from .kubernetes_allocator import KubernetesAllocator
from .simulator_allocator import SimulatorAllocator
from .allocation_facade import ServiceAllocationFacade

__all__ = [
    'AllocationEvent',
    'AllocationEventType',
    'BaseAllocator',
    'KubernetesAllocator',
    'SimulatorAllocator',
    'ServiceAllocationFacade',
]
