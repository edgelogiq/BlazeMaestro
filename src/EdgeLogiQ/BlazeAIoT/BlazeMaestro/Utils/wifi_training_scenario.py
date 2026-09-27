"""
WiFi Training Scenario - DynamicScenario extension for WiFi mobility

Extends DynamicScenario to support robot movement and WiFi quality
changes during training. Integrates with WiFi scenario manager for
realistic mobile robot simulation.

Architecture:
- WiFiDynamicScenario: Extends DynamicScenario with WiFi support
- Manages scenario timeline and applies WiFi-specific dynamics

Follows SOLID principles:
- Inheritance: Extends proven DynamicScenario pattern
- Single Responsibility: Only manages WiFi-specific changes
- Dependency Injection: Receives scenario manager and WiFi adapter
"""

from typing import List, Optional
import logging
import random
import numpy as np

from train_utils import DynamicScenario
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.service_simulator import Service


class WiFiDynamicScenario(DynamicScenario):
    """Extends DynamicScenario with WiFi mobility dynamics.
    
    Adds robot movement and WiFi quality changes to baseline loads.
    Suitable for training orchestrators to handle mobile edge computing.
    
    Timeline:
    - T=0-60s: Startup (robot at edge-gpu1, strong WiFi)
    - T=60-300s: Travel (degrading WiFi as distance increases)
    - T=300-360s: Arrival (robot at edge-gpu2, strong WiFi)
    - T=360s+: Return (degrading WiFi again)
    """
    
    def __init__(self, 
                 duration_minutes: int = 10,
                 time_step_minutes: int = 1,
                 wifi_adapter = None):
        """Initialize WiFi dynamic scenario.
        
        Args:
            duration_minutes: Scenario duration in minutes
            time_step_minutes: Time step duration in minutes
            wifi_adapter: Optional WiFiOrchestrationAdapter for WiFi updates
        """
        super().__init__()
        self.duration_minutes = duration_minutes
        self.time_step_minutes = time_step_minutes
        self.num_steps = duration_minutes // time_step_minutes
        self.current_step = 0
        self.wifi_adapter = wifi_adapter
        self._logger = logging.getLogger("WiFiDynamicScenario")
        
        # Track robot WiFi quality history for better decision-making
        self.robot_quality_history = {}
        self.num_rebalances_triggered = 0
    
    def get_time_label(self) -> str:
        """Get human-readable time label for current step."""
        return f"T={self.current_step * self.time_step_minutes}min"
    
    def apply_scenario_changes(self, services: List[Service], nodes: List, step: int):
        """Apply time-dependent changes including WiFi dynamics.
        
        Combines DynamicScenario baseline load changes with WiFi-specific
        dynamics that affect scheduling decisions.
        
        Args:
            services: List of services to modify
            nodes: List of nodes in platform
            step: Current time step
        """
        time_min = step * self.time_step_minutes
        time_sec = time_min * 60
        
        # Apply baseline load changes (inherited from DynamicScenario)
        self._apply_baseline_load_changes(services, nodes, step)
        
        # Apply WiFi-specific changes
        self._apply_wifi_changes(services, nodes, time_sec)
        
        # Log scenario state
        if step % max(1, self.num_steps // 5) == 0:
            self._log_scenario_state(time_sec, step)
    
    def _apply_baseline_load_changes(self, services: List[Service], nodes: List, step: int):
        """Apply baseline load changes from parent DynamicScenario."""
        # Update baseline loads for all nodes
        time_min = step * self.time_step_minutes
        time_sec = time_min * 60
        
        # Time-varying baseline (similar to compare_orchestrators.py pattern)
        if time_min < 2:
            # Normal operation
            scenario_desc = "Normal operation"
            for svc in services:
                svc.execution_rate_hz *= random.uniform(0.95, 1.05)
        elif time_min < 4:
            # Peak traffic
            scenario_desc = "Peak traffic (1.5x execution rates)"
            for svc in services:
                if not hasattr(svc, '_baseline_rate'):
                    setattr(svc, '_baseline_rate', svc.execution_rate_hz)
                baseline = getattr(svc, '_baseline_rate', svc.execution_rate_hz)
                svc.execution_rate_hz = baseline * 1.5 * random.uniform(0.95, 1.05)
        elif time_min < 6:
            # Increased workload
            scenario_desc = "Increased workload (1.2x execution times)"
            for svc in services:
                for container in svc.containers:
                    if not hasattr(container, '_baseline_exec_time'):
                        setattr(container, '_baseline_exec_time', container.exec_time)
                    baseline_time = getattr(container, '_baseline_exec_time', container.exec_time)
                    container.exec_time = baseline_time * 1.2 * random.uniform(0.95, 1.05)
        else:
            # Low traffic period
            scenario_desc = "Low traffic period"
            for svc in services:
                svc.execution_rate_hz = getattr(svc, '_baseline_rate', svc.execution_rate_hz) * 0.7
    
    def _apply_wifi_changes(self, services: List[Service], nodes: List, time_sec: float):
        """Apply WiFi-specific dynamics to services and nodes.
        
        Simulates how WiFi quality degradation affects service performance.
        """
        # Update WiFi state if adapter available
        if self.wifi_adapter:
            try:
                # Update robot positions from scenario
                self.wifi_adapter.update_robots_from_scenario(time_sec)
                
                # Get current WiFi metrics
                metrics = self.wifi_adapter.get_wifi_metrics()
                
                # Update motor utilization based on robot speed (if available)
                scenario = self.wifi_adapter.scenario_manager.get_current_scenario()
                max_speed = float(getattr(scenario, 'travel_speed', 0.33)) if scenario else 0.33
                for robot_id in metrics.keys():
                    status = self.wifi_adapter.wifi.get_robot_status(robot_id)
                    if status:
                        speed = float(status.get('speed', 0.0))
                        util = min(1.0, speed / max(max_speed, 1e-6))
                        for node in nodes:
                            if str(getattr(node, 'node_id', '')) == str(robot_id):
                                setattr(node, 'motor_speed_mps', speed)
                                setattr(node, 'motor_max_speed_mps', max_speed)
                                setattr(node, 'motor_utilization', util)

                # Adjust service behavior based on WiFi quality
                for robot_id, metric in metrics.items():
                    # For services running on this robot, adjust execution times
                    if metric.snr_level == 2:  # Poor WiFi
                        # Increase network-dependent service times
                        self._increase_service_times(services, factor=1.3)
                        
                        # Logging with hysteresis to avoid spam
                        if robot_id not in self.robot_quality_history or \
                           self.robot_quality_history[robot_id] != 2:
                            self._logger.info(f"[{self.get_time_label()}] {robot_id}: WiFi POOR (SNR={metric.snr_level}, "
                                            f"BW={metric.bandwidth_mbps}Mbps, Lat={metric.latency_ms}ms)")
                    
                    elif metric.snr_level == 1:  # Medium WiFi
                        self._increase_service_times(services, factor=1.15)
                        if robot_id not in self.robot_quality_history or \
                           self.robot_quality_history[robot_id] != 1:
                            self._logger.info(f"[{self.get_time_label()}] {robot_id}: WiFi MEDIUM (SNR={metric.snr_level})")
                    
                    else:  # Good WiFi
                        # Normal execution times
                        if robot_id not in self.robot_quality_history or \
                           self.robot_quality_history[robot_id] != 0:
                            self._logger.info(f"[{self.get_time_label()}] {robot_id}: WiFi GOOD (SNR={metric.snr_level})")
                    
                    self.robot_quality_history[robot_id] = metric.snr_level
                
                # Check if rebalancing should trigger
                should_rebalance, rebalance_info = self.wifi_adapter.should_trigger_rebalancing(time_sec)
                if should_rebalance:
                    self.num_rebalances_triggered += 1
                    self._logger.info(f"[{self.get_time_label()}] REBALANCE TRIGGERED: {rebalance_info['reason']}")
            
            except Exception as e:
                self._logger.debug(f"WiFi changes: {e}")
    
    def _increase_service_times(self, services: List[Service], factor: float = 1.2):
        """Increase execution times of services due to WiFi degradation.
        
        Args:
            services: List of services to adjust
            factor: Multiplier for execution times
        """
        for svc in services:
            # Increase execution times for network-dependent services
            if self._is_network_dependent(svc.service_id):
                for container in svc.containers:
                    if not hasattr(container, '_baseline_exec_time'):
                        setattr(container, '_baseline_exec_time', container.exec_time)
                    baseline = getattr(container, '_baseline_exec_time', container.exec_time)
                    # Apply factor but cap at reasonable maximum
                    container.exec_time = min(baseline * factor, baseline * 2.0)
    
    @staticmethod
    def _is_network_dependent(service_id: str) -> bool:
        """Determine if service is network-dependent.
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if service is network-dependent
        """
        # Services that heavily depend on network
        network_dependent = {
            "slam_toolbox",
            "robot_localization", 
            "pure_pursuit",
            "emergency_stop",
            "camera",  # Streams data
        }
        return service_id in network_dependent
    
    def _log_scenario_state(self, time_sec: float, step: int):
        """Log current scenario state.
        
        Args:
            time_sec: Current time in seconds
            step: Current time step
        """
        scenario = self.wifi_adapter.scenario_manager.get_current_scenario() if self.wifi_adapter else None
        scenario_desc = scenario.get_time_description(time_sec) if scenario else "No scenario"
        
        metrics = self.wifi_adapter.get_wifi_metrics() if self.wifi_adapter else {}
        metrics_str = " | ".join([f"{rid}: SNR={m.snr_level}" for rid, m in metrics.items()])
        
        self._logger.info(f"[Step {step}] {self.get_time_label()} - {scenario_desc} | {metrics_str}")
    
    def get_summary(self) -> dict:
        """Get scenario execution summary.
        
        Returns:
            Dictionary with scenario statistics
        """
        return {
            "duration_minutes": self.duration_minutes,
            "steps_completed": self.current_step,
            "num_steps": self.num_steps,
            "rebalances_triggered": self.num_rebalances_triggered,
            "scenario_name": self.wifi_adapter.scenario_manager.get_current_scenario().name 
                           if self.wifi_adapter else "Unknown"
        }
