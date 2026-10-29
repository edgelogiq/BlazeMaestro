from __future__ import annotations
from BlazeAIoT.Core.Utils import BlazeLogger

from typing import Dict, Optional, Any, Sequence
from .base import BaseMetricsFacade


class SimulatorMetricsFacade(BaseMetricsFacade):
    def __init__(self, nodes: Sequence[Any], latency_matrix: Optional[Any] = None, wifi_controller: Optional[Any] = None):
        # Map node_id -> node
        self._nodes = {getattr(n, "node_id", str(i)): n for i, n in enumerate(nodes)}
        # Cache pre-computed latency matrix if provided
        self._cached_latency_matrix = latency_matrix
        self._nodes_list = list(nodes)
        # Optional WiFi controller for dynamic latency adjustment
        self._wifi_controller = wifi_controller

    def _node(self, node_id: str):
        return self._nodes[node_id]

    def get_node_capacity(self, node_id: str) -> Dict[str, float]:
        n = self._node(node_id)
        # Support both base Node (cpu, ram attributes) and NodeSimulator (cpu_container, ram_container)
        cpu_container = getattr(n, "cpu_container", None)
        if cpu_container is not None:
            cpu = float(getattr(cpu_container, "capacity", 0.0))
        else:
            cpu = float(getattr(n, "cpu", 0.0))
        ram_container = getattr(n, "ram_container", None)
        if ram_container is not None:
            ram = float(getattr(ram_container, "capacity", 0.0))
        else:
            ram = float(getattr(n, "ram", 0.0))
        gpu = float(getattr(getattr(n, "gpu", None), "capacity", 0.0)) if getattr(n, "gpu", None) else 0.0
        return {"cpu": cpu, "ram_gb": ram, "gpu_gb": gpu}

    def get_node_available(self, node_id: str) -> Dict[str, float]:
        n = self._node(node_id)
        # Support both base Node and NodeSimulator
        cpu_container = getattr(n, "cpu_container", None)
        if cpu_container is not None:
            cpu_avail = float(getattr(cpu_container, "level", 0.0))
        else:
            cpu_avail = float(getattr(n, "cpu", 0.0))
        ram_container = getattr(n, "ram_container", None)
        if ram_container is not None:
            ram_avail = float(getattr(ram_container, "level", 0.0))
        else:
            ram_avail = float(getattr(n, "ram", 0.0))
        gpu_avail = float(getattr(getattr(n, "gpu", None), "level", 0.0)) if getattr(n, "gpu", None) else 0.0
        return {"cpu": cpu_avail, "ram_gb": ram_avail, "gpu_gb": gpu_avail}

    def get_node_property(self, node_id: str, key: str) -> Optional[Any]:
        n = self._node(node_id)
        return getattr(n, key, None)

    def get_latency_matrix(self, nodes: Optional[list] = None) -> Any:
        """Return simulated latency matrix between nodes.

        Uses cached matrix from platform initialization, or computes from node attributes.
        Applies WiFi degradation if controller is available.

        Args:
            nodes: List of Node objects (optional; defaults to the nodes used to initialize the facade)

        Returns:
            2D numpy array where latency_matrix[i][j] is latency from node i to node j in seconds
        """
        import numpy as np

        if nodes is None:
            nodes = self._nodes_list

        # Use cached latency matrix if available (from platform.xml)
        latency_matrix = None
        if self._cached_latency_matrix is not None:
            try:
                if isinstance(self._cached_latency_matrix, np.ndarray):
                    # Already a valid numpy array
                    if self._cached_latency_matrix.ndim == 2:
                        latency_matrix = self._cached_latency_matrix.copy()
                else:
                    # Validate structure before converting: should be list of lists with consistent lengths
                    if isinstance(self._cached_latency_matrix, (list, tuple)):
                        # Check if all rows are lists/tuples of equal length
                        if len(self._cached_latency_matrix) > 0:
                            first_row_len = None
                            is_valid = True
                            for row in self._cached_latency_matrix:
                                if not isinstance(row, (list, tuple)):
                                    is_valid = False
                                    break
                                row_len = len(row)
                                if first_row_len is None:
                                    first_row_len = row_len
                                elif row_len != first_row_len:
                                    is_valid = False
                                    break
                            
                            if is_valid and first_row_len is not None:
                                # Safe to convert
                                latency_matrix = np.array(self._cached_latency_matrix, dtype=float)
            except Exception:
                # Silently fall back to computed latency matrix on any error
                pass

        if latency_matrix is None:
            # Fall back: compute from node layer attributes
            n_nodes = len(nodes)
            latency_matrix = np.zeros((n_nodes, n_nodes))

            # Build node to layer mapping
            node_to_layer = {}
            for node in nodes:
                layer_id = getattr(node, "layer_id", None)
                if not layer_id and hasattr(node, "node_id") and "-" in node.node_id:
                    layer_id = node.node_id.split("-")[0]
                node_to_layer[node.node_id] = layer_id or "unknown"

            # Compute latencies based on layer hierarchy with jitter
            # Layer hierarchy: cloud (farthest) > fog > edge (closest) > device
            layer_order = {"cloud": 3, "fog": 2, "edge": 1, "device": 0}

            rng = np.random.RandomState()  # Use default seeding for variation each call
            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):
                    if i == j:
                        # Same node: 0ms with minimal noise (< 0.01ms)
                        latency_matrix[i][j] = max(0.0, rng.normal(0.0, 0.00001))
                    elif node_to_layer.get(node_i.node_id) == node_to_layer.get(node_j.node_id):
                        # Same layer: 1ms ± 10%
                        base = 0.001
                        jitter = rng.normal(base, base * 0.1)
                        latency_matrix[i][j] = max(0.0, jitter)
                    else:
                        # Different layers: latency depends on layer distance
                        layer_i = node_to_layer.get(node_i.node_id, "unknown")
                        layer_j = node_to_layer.get(node_j.node_id, "unknown")

                        layer_rank_i = layer_order.get(layer_i, 0)
                        layer_rank_j = layer_order.get(layer_j, 0)
                        layer_distance = abs(layer_rank_i - layer_rank_j)

                        # Base latency: 20ms per layer distance
                        # edge<->fog: 20ms, edge<->cloud: 40ms, fog<->cloud: 60ms
                        base = 0.020 * layer_distance
                        jitter = rng.normal(base, base * 0.1)
                        latency_matrix[i][j] = max(0.0, jitter)

        # Apply WiFi latency adjustments if controller available
        if self._wifi_controller:
            latency_matrix = self._apply_wifi_latency_adjustments(latency_matrix, nodes)

        return latency_matrix
    
    def _apply_wifi_latency_adjustments(self, latency_matrix, nodes):
        """Apply WiFi link quality adjustments to latency matrix.
        
        For WiFi-connected robots, all paths go through the connected AP:
        robot -> AP -> destination
        """
        
        try:
            # Get all WiFi link characteristics
            wifi_links = getattr(self._wifi_controller, 'wifi_links', {})
            robots = getattr(self._wifi_controller, 'robots', {})

            base_matrix = getattr(latency_matrix, "base_matrix", None)
            target_matrix = base_matrix if base_matrix is not None else latency_matrix
            
            # Track which nodes are robots with WiFi
            wifi_robots = {}  # robot_id -> (idx, connected_ap_id, wifi_latency)
            
            for link_id, link in wifi_links.items():
                robot_id = link.robot_id
                if robot_id not in robots:
                    continue
                
                robot = robots[robot_id]
                connected_ap = robot.connected_ap
                
                # Determine WiFi latency based on connection status
                if not connected_ap:
                    # No WiFi connection - use very high latency
                    wifi_latency = 5.0  # 5 seconds = no connectivity
                else:
                    # Get latency based on SNR level
                    snr_level = link.current_snr_level
                    link_latencies = link.latencies  # [good, medium, poor]
                    wifi_latency = link_latencies[snr_level]
                
                # Find node indices for robot and AP
                robot_idx = None
                ap_idx = None
                
                for i, node in enumerate(nodes):
                    node_id = getattr(node, 'node_id', None)
                    if node_id == robot_id:
                        robot_idx = i
                    elif node_id == connected_ap:
                        ap_idx = i
                
                if robot_idx is not None:
                    wifi_robots[robot_id] = (robot_idx, ap_idx, wifi_latency)
                    
                    # Direct WiFi link to AP
                    if ap_idx is not None:
                        target_matrix[robot_idx][ap_idx] = wifi_latency
                        target_matrix[ap_idx][robot_idx] = wifi_latency
            
            # Now compute transitive paths: robot -> AP -> any_other_node
            for robot_id, (robot_idx, ap_idx, wifi_latency) in wifi_robots.items():
                if ap_idx is None:
                    # No AP connection, all paths are very high latency
                    for j in range(len(nodes)):
                        if j != robot_idx:
                            target_matrix[robot_idx][j] = 5.0
                            target_matrix[j][robot_idx] = 5.0
                else:
                    # For all destinations, route through the AP
                    for j in range(len(nodes)):
                        if j != robot_idx:
                            # robot -> AP -> destination
                            via_ap = target_matrix[robot_idx][ap_idx] + target_matrix[ap_idx][j]
                            target_matrix[robot_idx][j] = via_ap
                            
                            # reverse: destination -> AP -> robot
                            target_matrix[j][robot_idx] = target_matrix[j][ap_idx] + target_matrix[ap_idx][robot_idx]
        
        except Exception as e:
            BlazeLogger.get_instance().debug(f"Failed to apply WiFi latency adjustments: {e}")
        
        return latency_matrix



    def export_latency_matrix_to_xml(self, nodes: list, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export current latency matrix to XML file for comparison analysis.

        Args:
            nodes: List of Node objects
            output_path: Path where XML file will be written
            metadata: Optional dict with additional metadata

        Returns:
            Full path to written XML file
        """
        from .latency_matrix_exporter import LatencyMatrixExporter

        latency_matrix = self.get_latency_matrix(nodes)
        meta = metadata or {}
        meta["environment"] = "simulator"

        return LatencyMatrixExporter.export_to_xml(
            latency_matrix, nodes, output_path, source="simulator", metadata=meta
        )
