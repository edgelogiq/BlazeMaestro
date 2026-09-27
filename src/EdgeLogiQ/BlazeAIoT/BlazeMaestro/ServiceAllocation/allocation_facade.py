"""
Service allocation facade.

Provides a unified interface for service allocation across different environments
(Kubernetes, Simulator). Routes operations to appropriate backend implementation.
"""

from BlazeAIoT.Core.Utils import BlazeLogger
from typing import Optional, Union, Tuple
from .base_allocator import BaseAllocator
from .kubernetes_allocator import KubernetesAllocator
from .simulator_allocator import SimulatorAllocator



class ServiceAllocationFacade:
    """
    Facade for service allocation operations.
    
    Abstracts away the complexity of different allocation backends.
    Provides a single interface for both Kubernetes and Simulator modes.
    
    Usage:
        # In Kubernetes mode
        facade = ServiceAllocationFacade(
            mode='kubernetes',
            kubeconfig_path='/path/to/kubeconfig',
            context='prod-context'
        )
        
        # In Simulator mode
        facade = ServiceAllocationFacade(mode='simulator')
        
        # Allocate service
        success = facade.allocate(
            service_id='my-service',
            target_node='edge-cpu1',
            target_layer='edge',
            source_node='fog-gpu1',
            source_layer='fog',
            namespace='default'  # Required for Kubernetes
        )
    """
    
    def __init__(
        self,
        mode: str = 'kubernetes',
        service_manager_broker=None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the allocation facade.
        
        Args:
            mode: Execution mode ('kubernetes' or 'simulator')
            service_manager_broker: Optional broker for event publishing
            verbose: Enable verbose logging
            **kwargs: Backend-specific arguments:
                - For Kubernetes:
                    - kubeconfig_path: Path to kubeconfig file
                    - context: Kubernetes context
                    - replica_check_timeout: Timeout for replica checks
                - For Simulator:
                    (no additional arguments needed)
        
        Raises:
            ValueError: If mode is not recognized
        """
        self.mode = mode.lower()
        self.verbose = verbose
        
        if self.mode == 'kubernetes':
            self.allocator = KubernetesAllocator(
                cluster_configs=kwargs.get('cluster_configs'),
                nodes=kwargs.get('nodes'),
                verbose=verbose,
                replica_check_timeout=kwargs.get('replica_check_timeout', 60.0),
                confirmation_timeout=kwargs.get('confirmation_timeout', 300.0)
            )
        elif self.mode == 'simulator':
            self.allocator = SimulatorAllocator(service_manager_broker=None, verbose=verbose)
        else:
            raise ValueError(f"Unknown allocation mode: {mode}")
        
        if verbose:
            print(f"[ServiceAllocationFacade] Initialized in {self.mode} mode")
    
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
        Allocate (move) a service to a target node/layer.
        
        This orchestrates the entire allocation flow:
        1. Publishes ALLOCATION_INITIATED event
        2. Scales up (or activates) the service on the target node
        3. Publishes ALLOCATION_IN_PROGRESS event
        4. Waits for broker confirmation (if broker configured)
        5. Scales down (or deactivates) the service on the source node
        6. Publishes ALLOCATION_COMPLETED event
        
        Args:
            service_id: Unique service identifier
            target_node: Target node ID where service will run
            target_layer: Target layer ID (e.g., 'edge', 'fog', 'cloud')
            source_node: Current node where service is running (optional)
            source_layer: Current layer (optional)
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            Tuple[bool, Optional[str]]: (success status, failure reason if failed)
        """
        try:
            if self.verbose:
                print(f"[ServiceAllocationFacade] Allocating {service_id} "
                      f"from {source_node}({source_layer}) to {target_node}({target_layer})")
            
            return self.allocator.allocate(
                service_id=service_id,
                target_node=target_node,
                target_layer=target_layer,
                source_node=source_node,
                source_layer=source_layer,
                namespace=namespace,
                **kwargs
            )
        except Exception as e:
            error_msg = str(e)
            BlazeLogger.get_instance().error(f"Allocation failed: {error_msg}")
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
        Deallocate (remove) a service from a node/layer.
        
        Args:
            service_id: Service identifier
            node: Node to deallocate from
            layer: Layer ID
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.verbose:
                print(f"[ServiceAllocationFacade] Deallocating {service_id} from {node}")
            
            return self.allocator.deallocate(
                service_id=service_id,
                node=node,
                layer=layer,
                namespace=namespace,
                **kwargs
            )
        except Exception as e:
            BlazeLogger.get_instance().error(f"Deallocation failed: {e}")
            return False
    
    def get_replica_count(
        self,
        service_id: str,
        node: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Get current replica count for a service on a node.
        
        Args:
            service_id: Service identifier
            node: Node to check
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            int: Current replica count (0 means not allocated)
        """
        try:
            return self.allocator.get_replica_count(
                service_id=service_id,
                node=node,
                namespace=namespace,
                **kwargs
            )
        except Exception as e:
            BlazeLogger.get_instance().error(f"Failed to get replica count: {e}")
            return None
    
    def get_allocator(self) -> BaseAllocator:
        """
        Get the underlying allocator instance.
        
        Useful for accessing mode-specific functionality or for testing.
        
        Returns:
            BaseAllocator: The active allocator implementation
        """
        return self.allocator
    
    @staticmethod
    def create_for_kubernetes(
        cluster_configs: Optional[list] = None,
        nodes: Optional[list] = None,
        service_manager_broker=None,
        verbose: bool = False
    ) -> 'ServiceAllocationFacade':
        """
        Factory method to create a Kubernetes-backed facade with multi-cluster support.
        
        Args:
            cluster_configs: List of cluster config dicts (kubeconfig_path, context, layer_id, name)
            nodes: List of node objects for node->cluster mapping
            service_manager_broker: Optional broker
            verbose: Verbose logging
        
        Returns:
            ServiceAllocationFacade in Kubernetes mode
        """
        return ServiceAllocationFacade(
            mode='kubernetes',
            cluster_configs=cluster_configs,
            nodes=nodes,
            service_manager_broker=service_manager_broker,
            verbose=verbose
        )
    
    @staticmethod
    def create_for_simulator(
        service_manager_broker=None,
        verbose: bool = False
    ) -> 'ServiceAllocationFacade':
        """
        Factory method to create a Simulator-backed facade.
        
        Args:
            service_manager_broker: Optional broker
            verbose: Verbose logging
        
        Returns:
            ServiceAllocationFacade in Simulator mode
        """
        return ServiceAllocationFacade(
            mode='simulator',
            service_manager_broker=service_manager_broker,
            verbose=verbose
        )
