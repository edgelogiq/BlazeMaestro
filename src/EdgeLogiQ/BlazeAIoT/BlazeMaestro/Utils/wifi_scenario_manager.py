"""
WiFi Scenario Manager - Realistic Robot Mobility Scenarios

Provides dynamic WiFi quality scenarios for training and simulation.
Integrates robot movement patterns with orchestration decision-making.

Architecture:
- WiFiScenario: Base class for WiFi scenarios
- RobotTravelScenario: Realistic robot travel between access points
- WiFiScenarioManager: Factory and manager for WiFi scenarios

Follows SOLID principles:
- Single Responsibility: Each class manages one scenario type
- Open/Closed: Easy to add new scenario types
- Liskov Substitution: All scenarios implement same interface
- Interface Segregation: Minimal public API
- Dependency Inversion: Depends on abstractions, not concrete classes
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import logging


@dataclass
class RobotPosition:
    """Represents a robot's position and velocity at a time step."""
    time: float
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


class WiFiScenario(ABC):
    """Abstract base class for WiFi scenarios.
    
    A scenario defines how robots move and WiFi quality changes over time.
    """
    
    def __init__(self, name: str):
        """Initialize scenario.
        
        Args:
            name: Human-readable scenario name
        """
        self.name = name
        self._logger = logging.getLogger(f"WiFiScenario.{name}")
    
    @abstractmethod
    def get_robot_trajectory(self, robot_id: str, duration: float, time_step: float) -> List[RobotPosition]:
        """Get robot movement trajectory over time.
        
        Args:
            robot_id: Identifier of robot (e.g., 'frank-1')
            duration: Total simulation duration in seconds
            time_step: Time between updates in seconds
            
        Returns:
            List of RobotPosition objects describing movement
        """
        pass
    
    @abstractmethod
    def get_time_description(self, current_time: float) -> str:
        """Get human-readable description of scenario at current time.
        
        Args:
            current_time: Current simulation time in seconds
            
        Returns:
            String describing what's happening in scenario
        """
        pass
    
    def should_react_to_wifi_quality(self, current_time: float) -> bool:
        """Determine if orchestrator should react to WiFi quality changes.
        
        Args:
            current_time: Current simulation time in seconds
            
        Returns:
            True if orchestrator should make decisions based on WiFi
        """
        return True


class RobotTravelScenario(WiFiScenario):
    """Realistic scenario: Robot travels between two WiFi access points.
    
    Simulates a mobile robot (frank-1) moving from edge-gpu1 area to
    edge-gpu2 area, with quality degradation as it moves away and
    improvement as it approaches.
    
    Timeline:
    - 0-60s: Robot starts at edge-gpu1 (0,0), initiates workload
    - 60-300s: Robot travels from edge-gpu1 to edge-gpu2
    - 300-360s: Robot at edge-gpu2, WiFi improves
    - 360s+: Robot returns to edge-gpu1
    
    The second robot (frank-2) remains stationary.
    """
    
    def __init__(self, duration_seconds: float = 600.0, travel_speed: float = 0.33):
        """Initialize robot travel scenario.
        
        Args:
            duration_seconds: Total scenario duration
            travel_speed: Robot speed in meters/second (realistic: 0.3 m/s for mobile robot)
        """
        super().__init__("RobotTravel")
        self.duration = duration_seconds
        self.travel_speed = travel_speed
        
        # Access point positions (from platform_sim.xml)
        self.ap1_pos = (0.0, 0.0)      # edge-gpu1
        self.ap2_pos = (100.0, 0.0)    # edge-gpu2
        self.distance_ap1_ap2 = 100.0
        
        # Travel phases
        self.phase_startup = 60.0      # Robot prepares, strong WiFi at AP1
        self.phase_travel = 240.0      # Robot travels between APs
        self.phase_arrival = 60.0      # Robot arrives at AP2, strong WiFi
        self.phase_return = 240.0      # Robot returns to AP1
    
    def get_robot_trajectory(self, robot_id: str, duration: float, time_step: float) -> List[RobotPosition]:
        """Generate robot trajectory for travel scenario.
        
        Args:
            robot_id: Robot identifier
            duration: Simulation duration in seconds
            time_step: Time between trajectory points
            
        Returns:
            List of RobotPosition objects
        """
        trajectory = []
        current_time = 0.0
        
        if robot_id == "frank-1":
            # frank-1 travels from AP1 to AP2
            while current_time <= duration:
                x, y, vx, vy = self._compute_frank1_position(current_time)
                trajectory.append(RobotPosition(
                    time=current_time,
                    x=x, y=y,
                    vx=vx, vy=vy
                ))
                current_time += time_step
        else:
            # Other robots stay at initial position
            while current_time <= duration:
                trajectory.append(RobotPosition(
                    time=current_time,
                    x=10.0, y=10.0,  # Static position
                    vx=0.0, vy=0.0
                ))
                current_time += time_step
        
        return trajectory
    
    def _compute_frank1_position(self, time: float) -> Tuple[float, float, float, float]:
        """Compute frank-1 position and velocity at given time.
        
        Returns:
            (x, y, vx, vy) position and velocity
        """
        if time < self.phase_startup:
            # Phase 1: Startup at AP1
            return 10.0, 10.0, 0.0, 0.0
        
        elif time < self.phase_startup + self.phase_travel:
            # Phase 2: Travel from AP1 (0,0) to AP2 (100,0)
            # Linear interpolation
            phase_time = time - self.phase_startup
            progress = phase_time / self.phase_travel  # 0 to 1
            
            # Path: start at (10,10) -> move toward (100,0)
            start_x, start_y = 10.0, 10.0
            end_x, end_y = 90.0, 0.0
            
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            
            # Velocity (constant during travel)
            vx = (end_x - start_x) / self.phase_travel
            vy = (end_y - start_y) / self.phase_travel
            
            return x, y, vx, vy
        
        elif time < self.phase_startup + self.phase_travel + self.phase_arrival:
            # Phase 3: Arrival at AP2
            return 90.0, 0.0, 0.0, 0.0
        
        else:
            # Phase 4: Return to AP1
            phase_time = time - (self.phase_startup + self.phase_travel + self.phase_arrival)
            progress = phase_time / self.phase_return  # 0 to 1
            
            start_x, start_y = 90.0, 0.0
            end_x, end_y = 10.0, 10.0
            
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            
            vx = (end_x - start_x) / self.phase_return
            vy = (end_y - start_y) / self.phase_return
            
            return x, y, vx, vy
    
    def get_time_description(self, current_time: float) -> str:
        """Describe what's happening at current time in scenario."""
        if current_time < self.phase_startup:
            return "Robot startup phase (strong WiFi at edge-gpu1)"
        
        elif current_time < self.phase_startup + self.phase_travel:
            phase_time = current_time - self.phase_startup
            progress_pct = 100.0 * phase_time / self.phase_travel
            return f"Robot traveling to edge-gpu2 ({progress_pct:.0f}% complete, WiFi degrading)"
        
        elif current_time < self.phase_startup + self.phase_travel + self.phase_arrival:
            return "Robot arrival phase (strong WiFi at edge-gpu2)"
        
        else:
            phase_time = current_time - (self.phase_startup + self.phase_travel + self.phase_arrival)
            progress_pct = 100.0 * phase_time / self.phase_return
            return f"Robot returning to edge-gpu1 ({progress_pct:.0f}% complete, WiFi degrading)"


class WiFiScenarioManager:
    """Factory and manager for WiFi scenarios.
    
    Provides centralized scenario creation and selection.
    Implements Factory pattern for extensibility.
    """
    
    # Registry of available scenarios
    _scenarios: Dict[str, type] = {
        "robot_travel": RobotTravelScenario,
    }
    
    def __init__(self):
        """Initialize scenario manager."""
        self._logger = logging.getLogger("WiFiScenarioManager")
        self._current_scenario: Optional[WiFiScenario] = None
    
    @classmethod
    def register_scenario(cls, name: str, scenario_class: type) -> None:
        """Register a new scenario type.
        
        Args:
            name: Scenario identifier
            scenario_class: Class implementing WiFiScenario
        """
        cls._scenarios[name] = scenario_class
        logging.getLogger("WiFiScenarioManager").info(f"Registered scenario: {name}")
    
    @classmethod
    def get_available_scenarios(cls) -> List[str]:
        """Get list of available scenario names.
        
        Returns:
            List of scenario identifiers
        """
        return list(cls._scenarios.keys())
    
    def create_scenario(self, scenario_name: str, **kwargs) -> Optional[WiFiScenario]:
        """Create a scenario instance.
        
        Args:
            scenario_name: Name of scenario to create
            **kwargs: Additional arguments for scenario constructor
            
        Returns:
            WiFiScenario instance or None if not found
        """
        if scenario_name not in self._scenarios:
            self._logger.error(f"Unknown scenario: {scenario_name}")
            return None
        
        try:
            scenario_class = self._scenarios[scenario_name]
            scenario = scenario_class(**kwargs)
            self._current_scenario = scenario
            self._logger.info(f"Created scenario: {scenario_name}")
            return scenario
        except Exception as e:
            self._logger.error(f"Failed to create scenario {scenario_name}: {e}")
            return None
    
    def get_current_scenario(self) -> Optional[WiFiScenario]:
        """Get currently active scenario.
        
        Returns:
            Current WiFiScenario or None
        """
        return self._current_scenario
    
    def set_current_scenario(self, scenario: WiFiScenario) -> None:
        """Set current scenario.
        
        Args:
            scenario: WiFiScenario to activate
        """
        self._current_scenario = scenario
        self._logger.info(f"Set current scenario: {scenario.name}")


# Singleton instance for convenient access
_scenario_manager_instance: Optional[WiFiScenarioManager] = None


def get_wifi_scenario_manager() -> WiFiScenarioManager:
    """Get or create the WiFi scenario manager singleton.
    
    Returns:
        WiFiScenarioManager instance
    """
    global _scenario_manager_instance
    if _scenario_manager_instance is None:
        _scenario_manager_instance = WiFiScenarioManager()
    return _scenario_manager_instance
