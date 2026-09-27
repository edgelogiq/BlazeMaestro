"""
Abstract base class for service allocators.

Defines the interface that all service allocators must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from .allocation_event import AllocationEvent, AllocationEventType
import time
from typing import Callable, Optional


class BaseAllocator(ABC):
    """
    Abstract base class for service allocation implementations.
    
    An allocator is responsible for moving a service from its current
    node/layer to a target node/layer.
    """
    
    def __init__(self, service_manager_broker=None, verbose: bool = False):
        """
        Initialize the allocator.
        
        Args:
            service_manager_broker: Optional broker for event publishing (placeholder for user implementation)
            verbose: Enable verbose logging
        """
        self.service_manager_broker = service_manager_broker
        self.verbose = verbose
    
    @abstractmethod
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
        
        This is the main orchestration method that:
        1. Performs the allocation (deployment replica increment, etc.)
        2. Monitors pod status until ready
        3. Deallocates the original instance
        
        Args:
            service_id: Unique service identifier
            target_node: Target node ID for the service
            target_layer: Target layer ID for the service
            source_node: Source node ID (current placement)
            source_layer: Source layer ID (current placement)
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            Tuple[bool, Optional[str]]: (success status, failure reason if failed)
        """
        pass
    
    @abstractmethod
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
            service_id: Unique service identifier
            node: Node ID to deallocate from
            layer: Layer ID to deallocate from
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            bool: True if deallocation succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def get_replica_count(
        self,
        service_id: str,
        node: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Get current replica count for a service on a specific node.
        
        Args:
            service_id: Unique service identifier
            node: Node ID to check
            namespace: Kubernetes namespace (required for Kubernetes mode)
            **kwargs: Additional backend-specific arguments
        
        Returns:
            int: Current replica count (0 or greater)
        """
        pass
    
    def _publish_event(self, event: AllocationEvent) -> None:
        """
        Publish allocation event to the ServiceManager Broker.
        
        This is called internally during allocation flow.
        User should implement broker integration in their setup.
        
        Args:
            event: AllocationEvent to publish
        """
        if self.verbose:
            print(f"[ServiceAllocation] Event: {event.event_type.value} for {event.service_id}")
        
        if self.service_manager_broker:
            try:
                # PLACEHOLDER: User will implement actual broker communication
                # Example integration:
                # self.service_manager_broker.publish("allocation.events", event.to_dict())
                self.service_manager_broker.publish("allocation.events", event.to_dict())
            except Exception as e:
                if self.verbose:
                    print(f"[ServiceAllocation] Failed to publish event: {e}")
    
    def _create_event(
        self,
        event_type: AllocationEventType,
        service_id: str,
        target_node: Optional[str] = None,
        target_layer: Optional[str] = None,
        source_node: Optional[str] = None,
        source_layer: Optional[str] = None,
        namespace: Optional[str] = None,
        deployment_name: Optional[str] = None,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AllocationEvent:
        """
        Create an allocation event.
        
        Args:
            event_type: Type of event
            service_id: Service ID
            target_node: Target node
            target_layer: Target layer
            source_node: Source node
            source_layer: Source layer
            namespace: Kubernetes namespace
            deployment_name: Kubernetes deployment name
            error_message: Error message if event_type is ALLOCATION_FAILED
            details: Additional details dictionary
        
        Returns:
            AllocationEvent instance
        """
        return AllocationEvent(
            event_type=event_type,
            service_id=service_id,
            source_node=source_node,
            target_node=target_node,
            source_layer=source_layer,
            target_layer=target_layer,
            timestamp=time.time(),
            namespace=namespace,
            deployment_name=deployment_name,
            error_message=error_message,
            details=details
        )


