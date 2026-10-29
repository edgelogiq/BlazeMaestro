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
    
    def mark_updated(self, pod_name: str, node_name: str):
        """Mark a pod-node pair as having been updated."""
        self.updated_pairs.add((pod_name, node_name))
    
    def reset_updates(self):
        """Clear the updated tracking for next cycle."""
        self.updated_pairs.clear()
        self.logger.debug("[METRICS_MGR] Reset update tracking for next cycle")
    
