"""
Service (Kubernetes-like) and Container models for the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class ContainerSpec:
    id: str
    cpu: float
    ram_gb: float
    gpu_gb: float = 0.0
    exec_time: float | None = None  # Execution time in seconds (None if not time-constrained)
    image_size_gb: float = 0.5  # approximate image size
    exec_time_available: bool = True  # Track if exec_time is reliable
    # Optional observed usage metrics from services_update.xml (aggregated across instances)
    cpu_usage_min: float | None = None
    cpu_usage_max: float | None = None
    cpu_usage_last: float | None = None
    ram_usage_min_gb: float | None = None
    ram_usage_max_gb: float | None = None
    ram_usage_last_gb: float | None = None

@dataclass
class DependencySpec:
    """Represents a dependency on another service's output.

    service_id: upstream service identifier this service depends on.
    size_mb: approximate message size in megabytes (optional, for transfer time modeling).
    constraint_level: 'hard' = timing-critical (exec+latency must fit in period),
                      'soft' = latency-sensitive (optimize but not strict),
                      'none' = best-effort (no constraint)
    """
    service_id: str
    size_mb: float = 0.0
    blocking: bool = True
    feedback: bool = False
    constraint_level: str = 'hard'  # 'hard', 'soft', or 'none'

class Service:
    def __init__(self, service_id: str, containers: list[ContainerSpec], dependencies: list[DependencySpec] | None = None, execution_rate_hz: float = 1.0, locked_node_id: str | None = None, locked_layer_id: str | None = None, is_daemonset: bool = False, daemonset_layer_id: str | None = None, multi_layer: bool = False, layers: list[str] | None = None, node_selectors: dict[str, str] | None = None, deployments: dict[str, dict] | None = None, control_loop: bool = False):
        """
        :param service_id: Unique identifier for the service
        :param containers: List of ContainerSpec objects
        :param dependencies: List of DependencySpec objects (optional)
        :param execution_rate_hz: Execution rate in Hz (default 1.0)
        :param locked_node_id: If set, this service can only be assigned to the node with this ID
        :param locked_layer_id: If set, this service can only be assigned to nodes in this layer
        :param is_daemonset: If True, this service runs on every node in the specified layer
        :param daemonset_layer_id: Layer ID for daemonset deployment (required if is_daemonset=True)
        :param multi_layer: If True, this service spans multiple layers with per-layer rules
        :param layers: List of layer IDs this multi-layer service spans
        :param node_selectors: Dict of label key-value pairs for node affinity (e.g., {"gpu": "true", "zone": "us-west"})
        :param deployments: Dict mapping deployment names to their config (e.g., {"wifi": {}, "ethernet": {}}) for multi-deployment support
        """
        self.service_id = service_id
        self.containers = containers
        self.dependencies: list[DependencySpec] = dependencies or []
        self.node = None
        self.execution_rate_hz = execution_rate_hz  # in Hz
        self.locked_node_id = locked_node_id  # Node ID this service is locked to, or None if not locked
        self.locked_layer_id = locked_layer_id  # Layer ID this service is locked to, or None if not locked
        self.is_daemonset = is_daemonset  # Whether this service is a daemonset
        self.daemonset_layer_id = daemonset_layer_id  # Layer ID for daemonset deployment
        self.multi_layer = multi_layer  # Whether this service spans multiple layers
        self.layers = layers or []  # List of layers this service spans
        self.node_selectors = node_selectors or {}  # Node affinity label selectors
        self.deployments: dict[str, dict] = deployments or {}  # Multi-deployment support (e.g., wifi, ethernet)
        self.control_loop = bool(control_loop)
        self.violation_history: list[dict] = []  # Track violations for reassignment decisions
        self.last_known_assignment: str | None = None  # Remember the last successful assignment

        # Validate configuration
        # Allow multi-layer daemonsets without a specific layer id
        if self.is_daemonset and self.daemonset_layer_id is None:
            if not self.multi_layer:
                raise ValueError(f"Service {service_id}: daemonset_layer_id must be specified when is_daemonset=True")

        if self.is_daemonset and self.locked_node_id is not None:
            raise ValueError(f"Service {service_id}: cannot specify both is_daemonset=True and locked_node_id")

        if self.locked_node_id is not None and self.locked_layer_id is not None:
            raise ValueError(f"Service {service_id}: cannot specify both locked_node_id and locked_layer_id")

        if self.is_daemonset and self.locked_layer_id is not None:
            raise ValueError(f"Service {service_id}: cannot specify both is_daemonset=True and locked_layer_id")

        if self.multi_layer and not self.layers:
            # Multi-layer service should have layers specified, but not critical
            pass

    @property
    def period(self) -> float:
        """Return the period (in seconds) for the service based on its execution rate."""
        if self.execution_rate_hz > 0:
            return 1.0 / self.execution_rate_hz
        return float('inf')

    def has_reliable_exec_time(self) -> bool:
        """Check if all containers have reliable execution time information."""
        return all(
            getattr(c, 'exec_time', None) is not None and 
            getattr(c, 'exec_time_available', True) 
            for c in self.containers
        )

    def get_execution_time(self, fallback_strategy: str = 'dependency_based') -> tuple[float | None, bool]:
        """
        Get execution time for the service.
        
        Returns:
            tuple: (execution_time, is_reliable)
                - execution_time: Calculated or estimated execution time (None if not time-constrained)
                - is_reliable: True if based on actual measurements, False if estimated
        """
        # Check if any container has exec_time
        exec_times = [c.exec_time for c in self.containers if c.exec_time is not None]
        
        if exec_times and self.has_reliable_exec_time():
            # All containers have reliable exec_time
            return max(exec_times), True
        elif exec_times:
            # Has exec_time but not reliable - return with unreliable flag
            return max(exec_times), False
        
        # No exec_time available - this is a non-time-constrained service
        # Return None to indicate no timing constraint
        return None, False

    def record_violation(self, violation_type: str, details: dict):
        """Record a constraint violation for this service."""
        violation_record = {
            'timestamp': details.get('timestamp', 0),
            'type': violation_type,
            'details': details,
            'node_id': details.get('node_id'),
            'exec_time': details.get('exec_time'),
            'period': details.get('period')
        }
        self.violation_history.append(violation_record)
        
        # Keep only recent violations (last 10)
        self.violation_history = self.violation_history[-10:]

    def is_daemonset_service(self) -> bool:
        """Check if this service is configured as a daemonset."""
        return self.is_daemonset

    def get_daemonset_layer(self) -> str | None:
        """Get the layer ID for this daemonset service, or None if not a daemonset."""
        return self.daemonset_layer_id if self.is_daemonset else None

    def is_locked_to_node(self) -> bool:
        """Check if this service is locked to a specific node."""
        return self.locked_node_id is not None

    def is_locked_to_layer(self) -> bool:
        """Check if this service is locked to a specific layer."""
        return self.locked_layer_id is not None

    def get_deployment_names(self) -> list[str]:
        """Get the names of all deployments associated with this service."""
        return list(self.deployments.keys()) if self.deployments else []

    def get_locked_layer(self) -> str | None:
        """Get the layer ID this service is locked to, or None if not locked to a layer."""
        return self.locked_layer_id

