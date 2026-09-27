"""
Allocation Executor Module

Implements Single Responsibility Principle (SRP) for service allocation execution.
Handles the actual execution of service moves between nodes.
"""

from typing import Dict, List, Optional
from BlazeAIoT.Core.Utils import BlazeLogger


class AllocationExecutor:
    """
    Executes service allocation decisions.
    
    Responsibilities:
    - Compare proposed vs current allocation
    - Execute service moves (allocate/deallocate)
    - Track allocation changes
    - Report move statistics
    """
    
    def __init__(
        self,
        allocation_facade,
        nodes: list,
        config: dict
    ):
        """
        Initialize Allocation Executor.
        
        Args:
            allocation_facade: ServiceAllocationFacade instance
            nodes: List of Node objects
            config: Configuration dictionary
        """
        self._logger = BlazeLogger.get_instance()
        self.allocation_facade = allocation_facade
        self.nodes = nodes
        self.config = config
        self.node_to_layer_map = self._build_node_to_layer_map()
    
    def _build_node_to_layer_map(self) -> Dict[str, str]:
        """Build mapping from node_id to layer_id."""
        node_to_layer = {}
        for n in self.nodes:
            nid = getattr(n, 'node_id', '')
            lid = getattr(n, 'layer_id', None)
            if not lid and '-' in nid:
                lid = nid.split('-')[0]
            node_to_layer[nid] = lid or 'unknown'
        return node_to_layer
    
    def execute_allocation_moves(
        self,
        original_allocation: Dict[str, str],
        proposed_allocation: Dict[str, str],
        orchestratable_services: list
    ) -> Dict[str, int]:
        """
        Execute service allocation moves when orchestrator decides to relocate services.
        
        Compares original vs proposed allocation and calls allocate_service() for services
        that have moved to a different node.
        
        Args:
            original_allocation: Dict {service_id: node_id} before orchestration
            proposed_allocation: Dict {service_id: node_id} after orchestration
            orchestratable_services: List of Service objects
            
        Returns:
            Dict with 'succeeded' and 'failed' counts
        """
        if not self.allocation_facade:
            self._logger.debug("Allocation facade not initialized; skipping move execution")
            return {'succeeded': 0, 'failed': 0}
        
        # Build service_id -> service object map
        svc_map = {}
        for svc in orchestratable_services:
            sid = str(getattr(svc, 'service_id', None) or getattr(svc, 'id', None))
            if sid:
                svc_map[sid] = svc
        
        # Identify and execute moves
        moves_succeeded = 0
        moves_failed = 0
        
        for service_id, target_node in proposed_allocation.items():
            source_node = original_allocation.get(service_id)
            
            # Skip if no change
            if source_node == target_node:
                continue
            
            # Get service object
            svc = svc_map.get(service_id)
            if not svc:
                self._logger.warning(f"Service {service_id} not found; skipping move")
                continue
            
            # Get layer info
            source_layer = self.node_to_layer_map.get(source_node, 'unknown') if source_node else None
            target_layer = self.node_to_layer_map.get(target_node, 'unknown')
            
            # Get namespace
            namespace = self.config.get('services', {}).get('namespace', 'default')
            
            # Execute the move
            success = self._execute_single_move(
                service_id=service_id,
                target_node=target_node,
                target_layer=target_layer,
                source_node=source_node,
                source_layer=source_layer,
                namespace=namespace
            )
            
            if success:
                moves_succeeded += 1
            else:
                moves_failed += 1
        
        if moves_succeeded > 0 or moves_failed > 0:
            self._logger.info(f"[MOVE] Summary: {moves_succeeded} succeeded, {moves_failed} failed")
        
        return {'succeeded': moves_succeeded, 'failed': moves_failed}
    
    def _execute_single_move(
        self,
        service_id: str,
        target_node: str,
        target_layer: str,
        source_node: Optional[str],
        source_layer: Optional[str],
        namespace: str
    ) -> bool:
        """
        Execute a single service move.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self._logger.info(
                f"[MOVE] Executing: {service_id} from {source_node}({source_layer}) "
                f"→ {target_node}({target_layer})"
            )
            
            result = self.allocation_facade.allocate(
                service_id=service_id,
                target_node=target_node,
                target_layer=target_layer,
                source_node=source_node,
                source_layer=source_layer,
                namespace=namespace
            )
            
            # Handle both old bool return and new tuple return for backward compatibility
            if isinstance(result, tuple):
                success, reason = result
            else:
                success, reason = result, None
            
            if success:
                self._logger.info(f"[MOVE] ✓ Successfully allocated {service_id} to {target_node}")
            else:
                failure_msg = f"[MOVE] ✗ Failed to allocate {service_id} to {target_node}"
                if reason:
                    failure_msg += f" - Reason: {reason}"
                self._logger.warning(failure_msg)
            
            return success
        
        except Exception as e:
            self._logger.error(
                f"[MOVE] Exception allocating {service_id} to {target_node}: {e}",
                exc_info=True
            )
            return False
    
    def allocate_service(
        self,
        service_id: str,
        target_node: str,
        target_layer: str,
        source_node: Optional[str] = None,
        source_layer: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Allocate (move) a service to a target node/layer.
        
        This is a high-level method that uses the service allocation facade
        to move a service from its current placement to a new one.
        
        Args:
            service_id: Unique service identifier
            target_node: Target node ID where service will run
            target_layer: Target layer ID (e.g., 'edge', 'fog', 'cloud')
            source_node: Current node where service is running (optional)
            source_layer: Current layer (optional)
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            True if allocation succeeded, False otherwise
        """
        if not self.allocation_facade:
            self._logger.error("Service allocation not available (facade not initialized)")
            return False
        
        try:
            self._logger.info(
                f"Allocating {service_id} from {source_node}({source_layer}) "
                f"to {target_node}({target_layer})"
            )
            
            success = self.allocation_facade.allocate(
                service_id=service_id,
                target_node=target_node,
                target_layer=target_layer,
                source_node=source_node,
                source_layer=source_layer,
                namespace=namespace or 'default',
                **kwargs
            )
            
            if success:
                self._logger.info(f"✓ Successfully allocated {service_id} to {target_node}")
            else:
                self._logger.error(f"✗ Failed to allocate {service_id}")
            
            return success
        
        except Exception as e:
            self._logger.error(f"Allocation error: {e}", exc_info=True)
            return False
    
    def deallocate_service(
        self,
        service_id: str,
        node: str,
        layer: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Deallocate (remove) a service from a node/layer.
        
        Args:
            service_id: Service identifier
            node: Node to deallocate from
            layer: Layer ID
            namespace: Kubernetes namespace
            **kwargs: Additional backend-specific arguments
        
        Returns:
            True if successful, False otherwise
        """
        if not self.allocation_facade:
            self._logger.error("Service allocation not available (facade not initialized)")
            return False
        
        try:
            self._logger.info(f"Deallocating {service_id} from {node}({layer})")
            
            success = self.allocation_facade.deallocate(
                service_id=service_id,
                node=node,
                layer=layer,
                namespace=namespace or 'default',
                **kwargs
            )
            
            if success:
                self._logger.info(f"✓ Successfully deallocated {service_id}")
            else:
                self._logger.error(f"✗ Failed to deallocate {service_id}")
            
            return success
        
        except Exception as e:
            self._logger.error(f"Deallocation error: {e}", exc_info=True)
            return False
