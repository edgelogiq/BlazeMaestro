"""
WiFi Orchestration Integration - Adapter for WiFi Mobility Controller

Bridges the WiFi mobility controller with the orchestration system.
Handles registration, updates, and decision-making based on WiFi quality.

Architecture:
- WiFiOrchestrationAdapter: Main adapter integrating WiFi into orchestration
- WiFiQualityReactor: Reacts to WiFi quality changes for task scheduling
- WiFiStateProvider: Provides WiFi state to orchestrator

Follows SOLID principles:
- Adapter pattern: Adapts WiFi controller to orchestration interface
- Single Responsibility: Each class has one reason to change
- Dependency Injection: Dependencies passed in constructor
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import numpy as np
import math

# Import WiFi components
from .wifi_mobility_controller import WiFiMobilityController
from .wifi_scenario_manager import WiFiScenario, WiFiScenarioManager


@dataclass
class WiFiMetrics:
    """WiFi quality metrics at a time step."""
    robot_id: str
    ap_id: str
    signal_strength: float  # 0.0 to 1.0
    snr_level: int          # 0 (good), 1 (medium), 2 (poor)
    bandwidth_mbps: float
    latency_ms: float
    quality_changed: bool   # True if SNR level changed from last step


class WiFiQualityReactor:
    """Reacts to WiFi quality changes for orchestration decisions.
    
    Determines whether task migration is needed based on WiFi quality
    degradation or improvement.
    """
    
    def __init__(self, wifi_controller: WiFiMobilityController, logger: Optional[logging.Logger] = None):
        """Initialize WiFi quality reactor.
        
        Args:
            wifi_controller: WiFiMobilityController instance
            logger: Optional logger instance
        """
        self.wifi = wifi_controller
        self._logger = logger or logging.getLogger("WiFiQualityReactor")
        self.last_quality: Dict[str, int] = {}  # robot_id -> last SNR level
    
    def should_trigger_rebalancing(self, robot_id: str, current_snr_level: int, threshold: float = 0.5) -> bool:
        """Determine if WiFi quality change warrants task rebalancing.
        
        Uses hysteresis to avoid thrashing: rebalance only if SNR level
        changed significantly or consistently poor.
        
        Args:
            robot_id: Robot identifier
            current_snr_level: Current SNR level (0=good, 1=medium, 2=poor)
            threshold: Threshold for rebalancing decision
            
        Returns:
            True if rebalancing recommended
        """
        last_snr = self.last_quality.get(robot_id, 0)
        
        # Rebalance if: 
        # 1. SNR level degraded by 2 levels (good -> poor)
        # 2. Currently at poor level (2)
        # 3. Returning from poor to medium/good
        
        if current_snr_level == 2 and last_snr < 2:
            # Just degraded to poor
            self._logger.info(f"{robot_id}: WiFi degraded to POOR (SNR: {last_snr} -> {current_snr_level}), triggering rebalance")
            self.last_quality[robot_id] = current_snr_level
            return True
        
        if current_snr_level < 2 and last_snr == 2:
            # Improved from poor
            self._logger.info(f"{robot_id}: WiFi improved from POOR (SNR: {last_snr} -> {current_snr_level}), allowing local processing")
            self.last_quality[robot_id] = current_snr_level
            return True
        
        self.last_quality[robot_id] = current_snr_level
        return False
    
class WiFiStateProvider:
    """Provides WiFi state information to orchestrator.
    
    Aggregates WiFi metrics and makes them available for orchestration
    decisions through a simple interface.
    """
    
    def __init__(self, wifi_controller: WiFiMobilityController, logger: Optional[logging.Logger] = None):
        """Initialize WiFi state provider.
        
        Args:
            wifi_controller: WiFiMobilityController instance
            logger: Optional logger instance
        """
        self.wifi = wifi_controller
        self._logger = logger or logging.getLogger("WiFiStateProvider")
        self.metrics_history: Dict[str, List[WiFiMetrics]] = {}
    
    def update_state(self, current_time: float) -> Dict[str, WiFiMetrics]:
        """Update WiFi state and return current metrics.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dict mapping robot_id to WiFiMetrics
        """
        try:
            updates = self.wifi.update(current_time)
            metrics = {}
            
            for link_id, (snr_level, ap_id, signal_strength) in updates.items():
                # Extract robot_id from link (format: "frank-1-wifi6")
                robot_id = link_id.split("-wifi")[0] if "-wifi" in link_id else link_id
                
                link_chars = self.wifi.get_link_characteristics(link_id)
                
                metric = WiFiMetrics(
                    robot_id=robot_id,
                    ap_id=ap_id,
                    signal_strength=signal_strength,
                    snr_level=snr_level,
                    bandwidth_mbps=self._parse_bandwidth(link_chars.get("bandwidth", "600Mbps")),
                    latency_ms=self._parse_latency(link_chars.get("latency", "5ms")),
                    quality_changed=(snr_level != self.metrics_history.get(robot_id, [None])[-1].snr_level 
                                    if robot_id in self.metrics_history and self.metrics_history[robot_id]
                                    else True)
                )
                
                metrics[robot_id] = metric
                
                # Store in history
                if robot_id not in self.metrics_history:
                    self.metrics_history[robot_id] = []
                self.metrics_history[robot_id].append(metric)
                
                # Keep history to last 100 entries
                if len(self.metrics_history[robot_id]) > 100:
                    self.metrics_history[robot_id] = self.metrics_history[robot_id][-100:]
            
            return metrics
        except Exception as e:
            self._logger.error(f"Failed to update WiFi state: {e}")
            return {}
    
    @staticmethod
    def _parse_bandwidth(bandwidth_str: str) -> float:
        """Parse bandwidth string (e.g., '600Mbps') to float.
        
        Args:
            bandwidth_str: Bandwidth string
            
        Returns:
            Bandwidth in Mbps
        """
        try:
            return float(bandwidth_str.replace("Mbps", "").strip())
        except:
            return 600.0
    
    @staticmethod
    def _parse_latency(latency_str: str) -> float:
        """Parse latency string (e.g., '5ms') to float.
        
        Args:
            latency_str: Latency string
            
        Returns:
            Latency in milliseconds
        """
        try:
            return float(latency_str.replace("ms", "").strip())
        except:
            return 5.0


class WiFiOrchestrationAdapter:
    """Main adapter integrating WiFi mobility with orchestration.
    
    Coordinates WiFi controller, scenario manager, and orchestrator
    for holistic mobile robot edge computing simulation.
    """
    
    def __init__(self, wifi_controller: WiFiMobilityController, logger: Optional[logging.Logger] = None):
        """Initialize WiFi orchestration adapter.
        
        Args:
            wifi_controller: WiFiMobilityController instance
            logger: Optional logger instance
        """
        self.wifi = wifi_controller
        self._logger = logger or logging.getLogger("WiFiOrchestrationAdapter")
        self.scenario_manager = WiFiScenarioManager()
        self.reactor = WiFiQualityReactor(wifi_controller, self._logger)
        self.state_provider = WiFiStateProvider(wifi_controller, self._logger)
    
    def initialize_with_scenario(self, 
                                 scenario: WiFiScenario,
                                 time_step: float = 1.0) -> bool:
        """Initialize WiFi simulation with scenario.
        
        Args:
            scenario: WiFiScenario to run
            time_step: Time between updates in seconds
            
        Returns:
            True if successful
        """
        try:
            self.scenario_manager.set_current_scenario(scenario)
            
            # Register access points (from platform config)
            self.wifi.register_access_point("edge-gpu1", 0.0, 0.0, 50.0)
            self.wifi.register_access_point("edge-gpu2", 100.0, 0.0, 50.0)
            
            # Register WiFi links
            self.wifi.register_wifi_link("frank-1-wifi6", "frank-1", 
                                        ["600Mbps", "300Mbps", "100Mbps"],
                                        ["5ms", "10ms", "20ms"])
            self.wifi.register_wifi_link("frank-2-wifi6", "frank-2",
                                        ["600Mbps", "300Mbps", "100Mbps"],
                                        ["5ms", "10ms", "20ms"])
            
            # Register robots with initial positions
            trajectory_f1 = scenario.get_robot_trajectory("frank-1", 600.0, time_step)
            trajectory_f2 = scenario.get_robot_trajectory("frank-2", 600.0, time_step)
            
            if trajectory_f1:
                first_pos = trajectory_f1[0]
                self.wifi.register_robot("frank-1", first_pos.x, first_pos.y)
            
            if trajectory_f2:
                first_pos = trajectory_f2[0]
                self.wifi.register_robot("frank-2", first_pos.x, first_pos.y)
            
            self._logger.info(f"Initialized WiFi adapter with scenario: {scenario.name}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize WiFi adapter: {e}", exc_info=True)
            return False
    
    def update_robots_from_scenario(self, current_time: float) -> bool:
        """Update robot positions from current scenario.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if successful
        """
        try:
            scenario = self.scenario_manager.get_current_scenario()
            if not scenario:
                return False
            
            # Get robot trajectories
            time_step = 1.0
            trajectory_f1 = scenario.get_robot_trajectory("frank-1", current_time + time_step, time_step)
            trajectory_f2 = scenario.get_robot_trajectory("frank-2", current_time + time_step, time_step)
            
            # Update robot positions and velocities
            if trajectory_f1 and len(trajectory_f1) > 0:
                pos = trajectory_f1[-1]  # Last position
                self.wifi.set_robot_velocity("frank-1", pos.vx, pos.vy)
            
            if trajectory_f2 and len(trajectory_f2) > 0:
                pos = trajectory_f2[-1]  # Last position
                self.wifi.set_robot_velocity("frank-2", pos.vx, pos.vy)
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to update robots from scenario: {e}")
            return False
    
    def should_trigger_rebalancing(self, current_time: float) -> Tuple[bool, Dict]:
        """Determine if orchestrator should rebalance services based on WiFi.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            (should_rebalance: bool, rebalance_info: Dict with details)
        """
        try:
            # Update WiFi state
            metrics = self.state_provider.update_state(current_time)
            
            # Check each robot for degradation
            rebalance_info = {
                "triggered": False,
                "reason": "",
                "affected_robots": []
            }
            
            for robot_id, metric in metrics.items():
                if self.reactor.should_trigger_rebalancing(robot_id, metric.snr_level):
                    rebalance_info["triggered"] = True
                    rebalance_info["affected_robots"].append(robot_id)
                    rebalance_info["reason"] = f"WiFi quality change on {robot_id}"
            
            return rebalance_info["triggered"], rebalance_info
        except Exception as e:
            self._logger.error(f"Failed to check rebalancing: {e}")
            return False, {"triggered": False, "reason": str(e), "affected_robots": []}
    
    def get_wifi_metrics(self) -> Dict[str, WiFiMetrics]:
        """Get current WiFi metrics for all robots.
        
        Returns:
            Dict mapping robot_id to WiFiMetrics
        """
        metrics = {}
        for robot_id, metrics_list in self.state_provider.metrics_history.items():
            if metrics_list:
                metrics[robot_id] = metrics_list[-1]
        return metrics
    
    def update_latency_matrix_with_positions(self, 
                                             base_latency_matrix: Any,
                                             nodes: List[Any]) -> np.ndarray:
        """Update latency matrix with current robot positions.
        
        Uses distance-based latency for WiFi connections between robots and APs.
        For WiFi links, latency increases with distance from connected AP.
        
        Args:
            base_latency_matrix: Base latency matrix from platform.xml (np.ndarray or LatencyMatrixWithJitter)
            nodes: List of node objects
            
        Returns:
            Updated latency matrix with position-based WiFi latencies
        """
        # Handle LatencyMatrixWithJitter wrapper
        if hasattr(base_latency_matrix, 'base_matrix'):
            base_matrix = np.array(base_latency_matrix.base_matrix)
        else:
            base_matrix = np.array(base_latency_matrix)
        
        updated_matrix = base_matrix.copy()
        
        # Build node index map
        node_id_to_idx = {str(getattr(n, 'node_id', i)): i for i, n in enumerate(nodes)}
        
        # Get current robot positions and AP positions
        for robot_id, robot in self.wifi.robots.items():
            if robot.connected_ap is None:
                continue
                
            robot_node_id = robot_id
            ap_node_id = robot.connected_ap
            
            # Check if these nodes exist in the node list
            if robot_node_id not in node_id_to_idx or ap_node_id not in node_id_to_idx:
                continue
            
            robot_idx = node_id_to_idx[robot_node_id]
            ap_idx = node_id_to_idx[ap_node_id]
            
            # Get AP position
            ap = self.wifi.access_points.get(ap_node_id)
            if ap is None:
                continue
            
            # Calculate distance-based latency
            # Distance in meters, base latency + distance penalty
            dx = robot.position.x - ap.position.x
            dy = robot.position.y - ap.position.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # WiFi latency model: base (5ms) + distance penalty (0.1ms per meter)
            # Also affected by signal quality
            base_wifi_latency = 0.005  # 5ms in seconds
            distance_penalty = distance * 0.0001  # 0.1ms per meter
            signal_strength = ap.get_signal_strength(robot.position)
            
            # Poor signal adds extra latency
            quality_penalty = 0.0
            if signal_strength < 0.5:
                quality_penalty = 0.015  # +15ms for poor signal
            elif signal_strength < 0.75:
                quality_penalty = 0.005  # +5ms for medium signal
            
            wifi_latency = base_wifi_latency + distance_penalty + quality_penalty
            
            # Update bidirectional latencies between robot and AP
            updated_matrix[robot_idx, ap_idx] = wifi_latency
            updated_matrix[ap_idx, robot_idx] = wifi_latency
            
            # Also update latencies between robots going through APs
            for other_robot_id, other_robot in self.wifi.robots.items():
                if other_robot_id == robot_id or other_robot.connected_ap is None:
                    continue
                
                if other_robot_id not in node_id_to_idx:
                    continue
                    
                other_robot_idx = node_id_to_idx[other_robot_id]
                other_ap_idx = node_id_to_idx.get(other_robot.connected_ap)
                
                if other_ap_idx is None:
                    continue
                
                # Robot-to-robot latency: robot1->AP1 + AP1->AP2 + AP2->robot2
                other_dx = other_robot.position.x - self.wifi.access_points[other_robot.connected_ap].position.x
                other_dy = other_robot.position.y - self.wifi.access_points[other_robot.connected_ap].position.y
                other_distance = math.sqrt(other_dx*other_dx + other_dy*other_dy)
                other_signal = self.wifi.access_points[other_robot.connected_ap].get_signal_strength(other_robot.position)
                
                other_quality_penalty = 0.0
                if other_signal < 0.5:
                    other_quality_penalty = 0.015
                elif other_signal < 0.75:
                    other_quality_penalty = 0.005
                
                other_wifi_latency = base_wifi_latency + other_distance * 0.0001 + other_quality_penalty
                
                # AP-to-AP latency from base matrix
                ap_to_ap_latency = base_latency_matrix[ap_idx, other_ap_idx] if ap_idx != other_ap_idx else 0.0
                
                total_latency = wifi_latency + ap_to_ap_latency + other_wifi_latency
                updated_matrix[robot_idx, other_robot_idx] = total_latency
                updated_matrix[other_robot_idx, robot_idx] = total_latency
        
        return updated_matrix
