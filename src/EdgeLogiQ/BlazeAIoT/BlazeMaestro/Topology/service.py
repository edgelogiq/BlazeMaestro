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

    def _estimate_exec_time_from_dependencies(self) -> float:
        """Estimate execution time based on dependency chain complexity."""
        if not self.dependencies:
            # No dependencies, assume light processing
            return 0.5
        
        # More dependencies = more complex processing
        base_time = 0.5
        dependency_factor = len(self.dependencies) * 0.3
        data_transfer_factor = sum(dep.size_mb for dep in self.dependencies) * 0.001  # 1ms per MB
        
        estimated_time = base_time + dependency_factor + data_transfer_factor
        return min(estimated_time, self.period * 0.8)  # Don't exceed 80% of period

    def _estimate_exec_time_from_rate(self) -> float:
        """Estimate execution time based on execution rate, assuming reasonable utilization."""
        if self.execution_rate_hz <= 0:
            return 1.0
        
        # Assume 60% utilization is reasonable for a sustainable operation
        period = self.period
        estimated_time = period * 0.6
        return estimated_time

    def _estimate_exec_time_from_resources(self) -> float:
        """Estimate execution time based on resource requirements."""
        total_cpu = sum(getattr(c, 'cpu', 1) for c in self.containers)
        total_ram = sum(getattr(c, 'ram_gb', 1.0) for c in self.containers)
        total_gpu = sum(getattr(c, 'gpu_gb', 0.0) for c in self.containers)
        
        # Simple heuristic: more resources = potentially more computation
        base_time = 0.5
        cpu_factor = total_cpu * 0.2
        ram_factor = total_ram * 0.1
        gpu_factor = total_gpu * 0.5  # GPU tasks often take longer
        
        estimated_time = base_time + cpu_factor + ram_factor + gpu_factor
        return min(estimated_time, self.period * 0.8)  # Don't exceed 80% of period

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

    def should_reassign(self, current_time: float = 0, violation_threshold: int = 3) -> bool:
        """
        Determine if service should be reassigned based on violation history.
        
        Args:
            current_time: Current simulation time
            violation_threshold: Number of recent violations that trigger reassignment
            
        Returns:
            bool: True if service should be reassigned
        """
        if len(self.violation_history) < violation_threshold:
            return False
        
        # Check recent violations (last 5 minutes of simulation time)
        recent_violations = [
            v for v in self.violation_history 
            if current_time - v.get('timestamp', 0) <= 300  # 5 minutes
        ]
        
        return len(recent_violations) >= violation_threshold

    def clear_violation_history(self):
        """Clear violation history after successful reassignment."""
        self.violation_history.clear()

    def is_daemonset_service(self) -> bool:
        """Check if this service is configured as a daemonset."""
        return self.is_daemonset

    def get_daemonset_layer(self) -> str | None:
        """Get the layer ID for this daemonset service, or None if not a daemonset."""
        return self.daemonset_layer_id if self.is_daemonset else None

    def create_daemonset_instance(self, node_id: str, instance_suffix: str | None = None) -> 'Service':
        """
        Create a per-node instance of this daemonset service.

        Args:
            node_id: The node ID where this instance will run
            instance_suffix: Optional suffix for the instance ID (defaults to node_id)

        Returns:
            A new Service instance configured for the specific node

        Raises:
            ValueError: If this service is not a daemonset
        """
        if not self.is_daemonset:
            raise ValueError(f"Service {self.service_id} is not a daemonset and cannot create instances")

        suffix = instance_suffix if instance_suffix else node_id
        instance_id = f"{self.service_id}-{suffix}"

        # Create a new service instance locked to the specific node
        instance = Service(
            service_id=instance_id,
            containers=self.containers.copy(),  # Share container specs
            dependencies=self.dependencies.copy() if self.dependencies else None,
            execution_rate_hz=self.execution_rate_hz,
            locked_node_id=node_id,  # Lock to specific node
            is_daemonset=False,  # Instances are not daemonsets themselves
            daemonset_layer_id=None
        )

        return instance

    def is_locked_to_node(self) -> bool:
        """Check if this service is locked to a specific node."""
        return self.locked_node_id is not None

    def is_locked_to_layer(self) -> bool:
        """Check if this service is locked to a specific layer."""
        return self.locked_layer_id is not None

    def has_multiple_deployments(self) -> bool:
        """Check if this service has multiple deployments (e.g., wifi and ethernet variants)."""
        return len(self.deployments) > 1

    def get_deployment_names(self) -> list[str]:
        """Get the names of all deployments associated with this service."""
        return list(self.deployments.keys()) if self.deployments else []

    def get_deployment_config(self, deployment_name: str) -> dict:
        """Get the configuration for a specific deployment variant."""
        return self.deployments.get(deployment_name, {}) if self.deployments else {}

    def add_deployment(self, deployment_name: str, config: dict | None = None):
        """Add or update a deployment variant for this service."""
        if self.deployments is None:
            self.deployments = {}
        self.deployments[deployment_name] = config or {}

    def get_all_deployment_names_from_k8s(self) -> list[str]:
        """Get the list of Kubernetes deployment names for this service."""
        names = []
        # Check for direct k8s_deployment attribute (single deployment)
        if hasattr(self, 'k8s_deployment') and self.k8s_deployment:
            names.append(self.k8s_deployment)
        # Check for k8s_deployments (multiple deployments)
        if hasattr(self, 'k8s_deployments') and self.k8s_deployments:
            names.extend(self.k8s_deployments)
        return names


    def get_locked_layer(self) -> str | None:
        """Get the layer ID this service is locked to, or None if not locked to a layer."""
        return self.locked_layer_id

    def can_run_on_node(self, node_id: str, node_to_layer_map: dict[str, str] | None = None) -> bool:
        """
        Check if this service can run on the specified node based on locking constraints.

        Args:
            node_id: The node ID to check
            node_to_layer_map: Optional mapping of node_id -> layer_id for layer lock validation

        Returns:
            bool: True if the service can run on this node, False otherwise
        """
        # If locked to a specific node, only that node is allowed
        if self.locked_node_id is not None:
            return node_id == self.locked_node_id

        # If locked to a layer, check if the node is in that layer
        if self.locked_layer_id is not None:
            if node_to_layer_map is None:
                # Cannot validate layer membership without the map
                return False
            node_layer = node_to_layer_map.get(node_id)
            return node_layer == self.locked_layer_id

        # No locking constraints, can run anywhere
        return True

    def get_eligible_nodes(self, all_node_ids: list[str], node_to_layer_map: dict[str, str] | None = None, node_labels: dict[str, str] | None = None, current_node_id: str | None = None) -> list[str]:
        """
        Get list of nodes where this service is eligible to run based on locking constraints.

        Args:
            all_node_ids: List of all available node IDs
            node_to_layer_map: Optional mapping of node_id -> layer_id for layer lock validation
            node_labels: Optional mapping of node_id -> labels (key-value pairs) for node affinity
            current_node_id: Current node where service is running (for multi-layer layer preservation)

        Returns:
            list[str]: List of eligible node IDs
        """
        eligible_nodes = set()

        # If locked to a specific node, return only that node (if it exists)
        if self.locked_node_id is not None:
            if self.locked_node_id in all_node_ids:
                eligible_nodes.add(self.locked_node_id)
        elif self.locked_layer_id is not None and node_to_layer_map is not None:
            # If locked to a layer, add only nodes in that layer
            eligible_nodes.update(
                node_id for node_id in all_node_ids
                if node_to_layer_map.get(node_id) == self.locked_layer_id
            )
        elif self.multi_layer and current_node_id and node_to_layer_map is not None:
            # Multi-layer service: preserve current layer, don't migrate across layers
            current_layer = node_to_layer_map.get(current_node_id)
            if current_layer and current_layer in self.layers:
                # Restrict to nodes in the same layer as current placement
                eligible_nodes.update(
                    node_id for node_id in all_node_ids
                    if node_to_layer_map.get(node_id) == current_layer
                )
            else:
                # No current layer or layer not in allowed list, allow all nodes in allowed layers
                eligible_nodes.update(
                    node_id for node_id in all_node_ids
                    if node_to_layer_map.get(node_id) in self.layers
                )
        else:
            # No locking constraints, all nodes are eligible
            eligible_nodes.update(all_node_ids)

        # Filter by node selectors (affinity rules)
        if self.node_selectors and node_labels is not None:
            eligible_nodes = {
                node_id for node_id in eligible_nodes
                if all(node_labels.get(node_id, {}).get(k) == v for k, v in self.node_selectors.items())
            }

        return list(eligible_nodes)

    def check_node_label_compatibility(self, node_id: str, node_labels: dict[str, str] | None = None) -> bool:
        """
        Check if the service's node selectors are compatible with the given node's labels.

        Args:
            node_id: The node ID to check
            node_labels: Mapping of node_id -> labels (key-value pairs) for node affinity

        Returns:
            bool: True if the node matches the service's selectors, False otherwise
        """
        if self.node_selectors and node_labels is not None:
            node_label_map = node_labels.get(node_id, {})
            return all(node_label_map.get(k) == v for k, v in self.node_selectors.items())
        return True  # No selectors, compatible by default
