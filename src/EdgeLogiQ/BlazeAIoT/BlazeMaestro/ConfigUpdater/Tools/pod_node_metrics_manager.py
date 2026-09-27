"""
Manage pod-node metrics associations to prevent cross-contamination.

This module maintains a clean mapping of (pod_name, node_name) → metrics
to ensure metrics are never mixed between different pod instances on different nodes.
"""
from typing import Dict, Tuple, Optional, Any
from BlazeAIoT.Core.Utils import BlazeLogger


class PodNodeMetricsManager:
    """
    Manages metrics association for pod instances across nodes.
    
    Key responsibilities:
    1. Maintain (pod_name, node_name) → metrics mapping
    2. Prevent metrics from being applied to wrong pod instances
    3. Track which pods have been updated from Prometheus
    4. Support node IP → name resolution without losing association
    """
    
    def __init__(self):
        """Initialize metrics manager."""
        self.logger = BlazeLogger.get_instance()
        # Primary key: (pod_name, normalized_node_name) → metrics dict
        self.pod_node_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Track IP → name mappings for resolution
        self.node_ip_to_name: Dict[str, str] = {}
        # Track which (pod_name, node) pairs have been updated
        self.updated_pairs: set = set()
        # Reverse lookups for debugging
        self.service_for_pod: Dict[str, str] = {}  # pod_name → service_id
    
    def register_node_resolution(self, node_ip: str, node_name: str):
        """Register an IP → node name mapping."""
        if node_ip != node_name and '.' in node_ip:
            self.node_ip_to_name[node_ip] = node_name
            self.logger.debug(f"[METRICS_MGR] Registered node resolution: {node_ip} → {node_name}")
    
    def resolve_node(self, node_identifier: str) -> str:
        """Resolve node IP to name, or return as-is if already a name."""
        if not node_identifier:
            return node_identifier
        # Try direct mapping
        if node_identifier in self.node_ip_to_name:
            return self.node_ip_to_name[node_identifier]
        # Try case-insensitive
        resolved = self.node_ip_to_name.get(node_identifier.lower())
        if resolved:
            return resolved
        # Return as-is
        return node_identifier
    
    def add_metrics(self, pod_name: str, node_name: str, metrics: Dict[str, Any],
                   service_id: str = None):
        """
        Add or update metrics for a pod on a specific node.
        
        Args:
            pod_name: Name of the pod (e.g., "robot-localization-xyz123")
            node_name: Normalized node name (e.g., "edge-gpu2")
            metrics: Dict with cpu_usage, ram_usage_gb, exec_time, etc.
            service_id: Service ID for tracking
        """
        key = (pod_name, node_name)
        self.pod_node_metrics[key] = metrics.copy()
        self.updated_pairs.add(key)
        if service_id:
            self.service_for_pod[pod_name] = service_id
        
        self.logger.debug(
            f"[METRICS_MGR] Stored metrics for {pod_name}@{node_name}: "
            f"cpu={metrics.get('cpu_usage', 0):.3f}, "
            f"ram={metrics.get('ram_usage_gb', 0):.3f}, "
            f"exec_time={metrics.get('exec_time', 0):.6f}"
        )
    
    def get_metrics(self, pod_name: str, node_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific pod on a specific node.
        
        Returns None if no metrics found - indicates instance should keep its current values.
        """
        key = (pod_name, node_name)
        metrics = self.pod_node_metrics.get(key)
        
        if metrics is None:
            self.logger.debug(f"[METRICS_MGR] No metrics found for {pod_name}@{node_name}")
        
        return metrics
    
    def has_metrics(self, pod_name: str, node_name: str) -> bool:
        """Check if metrics exist for a pod-node pair."""
        return (pod_name, node_name) in self.pod_node_metrics
    
    def get_all_metrics_for_pod(self, pod_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics for a pod across all nodes.
        Returns dict of node_name → metrics.
        """
        result = {}
        for (pname, node), metrics in self.pod_node_metrics.items():
            if pname == pod_name:
                result[node] = metrics
        return result
    
    def get_all_metrics_for_service(self, service_id: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Get all metrics for all pods in a service.
        Returns dict of (pod_name, node_name) → metrics.
        """
        result = {}
        for pod_node_key, metrics in self.pod_node_metrics.items():
            pod_name = pod_node_key[0]
            if self.service_for_pod.get(pod_name) == service_id:
                result[pod_node_key] = metrics
        return result
    
    def mark_updated(self, pod_name: str, node_name: str):
        """Mark a pod-node pair as having been updated."""
        self.updated_pairs.add((pod_name, node_name))
    
    def was_updated(self, pod_name: str, node_name: str) -> bool:
        """Check if a pod-node pair was updated in this cycle."""
        return (pod_name, node_name) in self.updated_pairs
    
    def reset_updates(self):
        """Clear the updated tracking for next cycle."""
        self.updated_pairs.clear()
        self.logger.debug("[METRICS_MGR] Reset update tracking for next cycle")
    
    def validate_no_mixing(self) -> bool:
        """
        Validate that metrics are properly segregated.
        Returns True if all metrics are clean (no mixing).
        """
        # Check that pod instances on different nodes have different metrics
        pods_by_node = {}
        for (pod_name, node_name), metrics in self.pod_node_metrics.items():
            if pod_name not in pods_by_node:
                pods_by_node[pod_name] = {}
            pods_by_node[pod_name][node_name] = metrics
        
        issues = 0
        for pod_name, nodes_metrics in pods_by_node.items():
            if len(nodes_metrics) > 1:
                # Pod on multiple nodes - check metrics are different
                metrics_values = [tuple(sorted(m.items())) for m in nodes_metrics.values()]
                if len(set(metrics_values)) == 1:
                    self.logger.warning(
                        f"[METRICS_VALIDATION] {pod_name} has identical metrics on different nodes: "
                        f"{list(nodes_metrics.keys())}"
                    )
                    issues += 1
        
        if issues == 0:
            self.logger.debug("[METRICS_VALIDATION] ✓ All metrics properly segregated")
        return issues == 0
    
    def debug_dump(self):
        """Dump current metrics state for debugging."""
        self.logger.debug("[METRICS_MGR_DUMP] Current pod-node metrics:")
        for (pod_name, node_name), metrics in self.pod_node_metrics.items():
            self.logger.debug(
                f"  {pod_name}@{node_name}: cpu={metrics.get('cpu_usage', 0):.3f}, "
                f"ram={metrics.get('ram_usage_gb', 0):.3f}"
            )
