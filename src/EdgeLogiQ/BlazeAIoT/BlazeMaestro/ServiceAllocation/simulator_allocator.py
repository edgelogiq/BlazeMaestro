"""
Simulator service allocator implementation.

In-memory allocation for simulation mode.
Tracks allocations and deallocations without external dependencies.

No events are published. Status is tracked in-memory only.
"""

from typing import Optional, Dict, Tuple
from .base_allocator import BaseAllocator



class SimulatorAllocator(BaseAllocator):
    """
    Simulator implementation of service allocator.
    
    Tracks service allocations in-memory without actual cluster operations.
    Useful for testing and simulation scenarios.
    """
    
    def __init__(self, service_manager_broker=None, verbose: bool = False):
        """
        Initialize simulator allocator.
        
        Args:
            service_manager_broker: Deprecated (unused)
            verbose: Enable verbose logging
        """
        super().__init__(service_manager_broker=None, verbose=verbose)
        # Track allocations: service_id -> (node, layer, replica_count)
        self._allocations: Dict[str, tuple] = {}
    
    def allocate(
        self,
        service_id: str,
        target_node: str,
        target_layer: str,
        source_node: Optional[str] = None,
        source_layer: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Simulate service allocation (in-memory).
        
        Flow:
        1. Update allocation tracking (target_node/layer with replica=1)
        2. Remove source allocation (set replica=0)
        
        Args:
            service_id: Service ID
            target_node: Target node
            target_layer: Target layer
            source_node: Source node (for deallocation)
            source_layer: Source layer
            namespace: Namespace (unused in simulator)
            **kwargs: Additional arguments
        
        Returns:
            Tuple[bool, Optional[str]]: (success status, failure reason if failed)
        """
        try:
            if self.verbose:
                print(f"[SimulatorAllocator] Allocating {service_id} to {target_node} ({target_layer})")
            
            # Step 1: Update allocation tracking (target gets 1 replica)
            self._allocations[f"{service_id}@{target_node}"] = (target_node, target_layer, 1)
            
            if self.verbose:
                print(f"[SimulatorAllocator] Simulated target allocation")
            
            # Step 2: Allocation complete
            
            # Step 3: Deallocate source (set replica=0)
            if source_node and source_layer:
                source_key = f"{service_id}@{source_node}"
                if source_key in self._allocations:
                    del self._allocations[source_key]
                
                if self.verbose:
                    print(f"[SimulatorAllocator] Deallocated source {source_node}")
            
            if self.verbose:
                print(f"[SimulatorAllocator] Successfully allocated {service_id}")
            
            return True, None
        
        except Exception as e:
            error_msg = str(e)
            if self.verbose:
                print(f"[SimulatorAllocator] Allocation failed: {error_msg}")
            return False, error_msg
    
    def deallocate(
        self,
        service_id: str,
        node: str,
        layer: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Simulate service deallocation (in-memory).
        
        Args:
            service_id: Service ID
            node: Node to deallocate from
            layer: Layer
            namespace: Namespace (unused in simulator)
            **kwargs: Additional arguments
        
        Returns:
            bool: True if successful
        """
        try:
            # Remove allocation
            allocation_key = f"{service_id}@{node}"
            if allocation_key in self._allocations:
                del self._allocations[allocation_key]
            
            if self.verbose:
                print(f"[SimulatorAllocator] Deallocated {service_id} from {node}")
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[SimulatorAllocator] Deallocation failed: {e}")
            return False
    
    def get_replica_count(
        self,
        service_id: str,
        node: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Get simulated replica count.
        
        Args:
            service_id: Service ID
            node: Node to check
            namespace: Namespace (unused)
            **kwargs: Additional arguments
        
        Returns:
            int: 1 if allocated, 0 otherwise
        """
        allocation_key = f"{service_id}@{node}"
        if allocation_key in self._allocations:
            _, _, replicas = self._allocations[allocation_key]
            return replicas
        return 0
    
    def get_all_allocations(self) -> Dict[str, tuple]:
        """
        Get all current allocations (useful for testing/debugging).
        
        Returns:
            Dict mapping allocation_key to (node, layer, replicas)
        """
        return dict(self._allocations)
    
    def clear_allocations(self) -> None:
        """Clear all allocations (for testing)."""
        self._allocations.clear()


