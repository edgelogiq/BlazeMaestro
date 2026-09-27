"""
Allocation event model for broker communication.

This module defines events that are published to the ServiceManager Broker
when service allocation operations are initiated or completed.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class AllocationEventType(Enum):
    """Types of allocation events."""
    ALLOCATION_INITIATED = "allocation_initiated"
    ALLOCATION_IN_PROGRESS = "allocation_in_progress"
    ALLOCATION_COMPLETED = "allocation_completed"
    ALLOCATION_FAILED = "allocation_failed"
    DEALLOCATION_INITIATED = "deallocation_initiated"
    DEALLOCATION_COMPLETED = "deallocation_completed"


@dataclass
class AllocationEvent:
    """Event published to ServiceManager Broker during allocation."""
    
    event_type: AllocationEventType
    service_id: str
    source_node: Optional[str]
    target_node: Optional[str]
    source_layer: Optional[str]
    target_layer: Optional[str]
    timestamp: float
    namespace: Optional[str] = None
    deployment_name: Optional[str] = None
    error_message: Optional[str] = None
    details: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert event to dictionary for broker serialization."""
        return {
            'event_type': self.event_type.value,
            'service_id': self.service_id,
            'source_node': self.source_node,
            'target_node': self.target_node,
            'source_layer': self.source_layer,
            'target_layer': self.target_layer,
            'timestamp': self.timestamp,
            'namespace': self.namespace,
            'deployment_name': self.deployment_name,
            'error_message': self.error_message,
            'details': self.details or {}
        }
