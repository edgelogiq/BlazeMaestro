"""
Test Orchestrator - Single Service Reallocation

This orchestrator performs a one-time reallocation of a single service to a specified
node or layer. It's useful for testing orchestration logic and validating service
placement behavior.

Configuration in blaze_maestro.yaml:
  orchestrator:
    type: test
    service_id: <service_id>        # ID of service to reallocate
    target_node_id: <node_id>       # Target node ID (takes precedence over layer)
    target_layer_id: <layer_id>     # Target layer ID (used if node_id not specified)
"""

from typing import Dict, Sequence, Any, List, Optional
from .base_orchestrator import BaseOrchestrator


class TestOrchestrator(BaseOrchestrator):
    """
    Test orchestrator that reallocates a single service to a specified node or layer.
    
    This orchestrator is designed for testing and validation purposes. It performs
    one-time reallocation of a specified service to a target node or layer and
    maintains that assignment for subsequent calls.
    """
    
    def __init__(self, services: Sequence[Any], nodes: Sequence[Any], 
                 metrics: Any, service_id: Optional[str] = None,
                 target_node_id: Optional[str] = None,
                 target_layer_id: Optional[str] = None,
                 current_allocation: Optional[Dict[str, str]] = None):
        """
        Initialize Test Orchestrator.
        
        Args:
            services: List of services to orchestrate
            nodes: List of available nodes
            metrics: Metrics facade
            service_id: ID of the service to reallocate
            target_node_id: Target node ID for reallocation (takes precedence)
            target_layer_id: Target layer ID for reallocation (used if node_id not set)
            current_allocation: Current service-to-node allocation to preserve
        
        Raises:
            ValueError: If service_id is not provided or not found in services
            ValueError: If target_node_id/target_layer_id are both not provided
            ValueError: If target_node_id doesn't exist
            ValueError: If target_layer_id doesn't exist
        """
        super().__init__(services, nodes, metrics)
        
        self.service_id = service_id
        self.target_node_id = target_node_id
        self.target_layer_id = target_layer_id
        self._current_allocation = current_allocation or {}
        self._reallocation_done = False
        
        # Validate configuration
        if not self.service_id:
            raise ValueError("TestOrchestrator requires 'service_id' parameter")
        
        # Find the service
        self.target_service = None
        for svc in self.services:
            if str(getattr(svc, 'service_id', '')) == str(self.service_id):
                self.target_service = svc
                break
        
        if self.target_service is None:
            available_ids = [str(getattr(s, 'service_id', 'unknown')) for s in self.services]
            raise ValueError(f"Service '{self.service_id}' not found. Available services: {available_ids}")
        
        # Validate target specification
        if not target_node_id and not target_layer_id:
            raise ValueError("TestOrchestrator requires either 'target_node_id' or 'target_layer_id'")
        
        # Determine target node index
        self.target_node_idx = None
        
        if target_node_id:
            # Look for node with matching node_id
            for i, node in enumerate(self.nodes):
                if str(getattr(node, 'node_id', '')) == str(target_node_id):
                    self.target_node_idx = i
                    break
            
            if self.target_node_idx is None:
                available_nodes = [str(getattr(n, 'node_id', 'unknown')) for n in self.nodes]
                raise ValueError(f"Target node '{target_node_id}' not found. Available nodes: {available_nodes}")
        else:
            # Look for first node in target layer
            for i, node in enumerate(self.nodes):
                if getattr(node, 'layer_id', None) == target_layer_id:
                    self.target_node_idx = i
                    break
            
            if self.target_node_idx is None:
                available_layers = set(getattr(n, 'layer_id', None) for n in self.nodes)
                raise ValueError(f"Target layer '{target_layer_id}' not found. Available layers: {available_layers}")
        
        print(f"[TestOrchestrator] Configured to reallocate service '{self.service_id}' "
              f"to node index {self.target_node_idx}")
    
    def _propose_assignment(self, services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """
        Propose an assignment that reallocates the target service once.
        
        On first call: Returns current assignment with target service moved to target node.
        On subsequent calls: Returns assignment unchanged (maintains the reallocation).
        
        Args:
            services: Optional subset of services (used by caller for filtering)
        
        Returns:
            Dict mapping service_id (str) -> node_id (str)
        """
        # Start with a copy of the current allocation to preserve all placements
        assignment = dict(self._current_allocation)
        
        # Only modify the target service
        target_service_id = str(self.service_id)
        
        if not self._reallocation_done:
            # First call: reallocate target service to target node
            target_node = self.nodes[self.target_node_idx]
            target_nid = str(getattr(target_node, 'node_id', f'node_{self.target_node_idx}'))
            assignment[target_service_id] = target_nid
            print(f"[TestOrchestrator] Reallocating service '{self.service_id}' to node '{target_nid}'")
            self._reallocation_done = True
        else:
            # Subsequent calls: ensure target service stays on target node
            target_node = self.nodes[self.target_node_idx]
            target_nid = str(getattr(target_node, 'node_id', f'node_{self.target_node_idx}'))
            assignment[target_service_id] = target_nid
        
        return assignment
    
    def _can_fit_service(self, service: Any, node: Any) -> bool:
        """
        Check if a service can fit on a node.
        
        Args:
            service: Service object
            node: Node object
        
        Returns:
            True if service can fit on node
        """
        # Get service demands
        cpu, ram, gpu, _, _ = self._service_demands(service)
        
        # Check node capacity
        node_cpu_cap = getattr(getattr(node, 'cpu', None), 'capacity', float('inf'))
        node_ram_cap = getattr(getattr(node, 'memory', None), 'capacity', float('inf'))
        node_gpu_cap = getattr(getattr(node, 'gpu', None), 'capacity', 0.0)
        
        # Simple check: does service fit in node's capacity
        if node_cpu_cap < float('inf') and cpu > node_cpu_cap:
            return False
        if node_ram_cap < float('inf') and ram > node_ram_cap:
            return False
        if gpu > 0 and node_gpu_cap < gpu:
            return False
        
        return True
    
    def run_single_episode_training(self, all_nodes: Sequence[Any], 
                                   services: Sequence[Any],
                                   latency: List[List[float]]) -> int:
        """
        Run single episode (not used for test orchestrator).
        
        Test orchestrator doesn't do iterative training.
        This method is provided for interface compatibility.
        
        Returns:
            1 (indicating single episode completed)
        """
        print("[TestOrchestrator] Single episode training (not applicable - test orchestrator doesn't iterate)")
        return 1
