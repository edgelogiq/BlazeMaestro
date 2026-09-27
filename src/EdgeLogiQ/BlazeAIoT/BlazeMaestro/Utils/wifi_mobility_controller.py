#!/usr/bin/env python3
"""
WiFi Mobility Controller - Simulates mobile robots moving in the network
and dynamically adjusts WiFi link quality based on distance to access points.
"""

import math
import logging
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field


@dataclass
class Position:
    """2D position with coordinates"""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class WifiAccessPoint:
    """WiFi Access Point with position and coverage"""
    host_id: str
    position: Position
    coverage_radius: float
    s_max: float = -70.0   # Reference signal level at d_max (dBm)
    beta: float = 2.0      # Path-loss exponent

    def is_in_range(self, robot_pos: Position) -> bool:
        """Check if robot position is within coverage"""
        return self.position.distance_to(robot_pos) <= self.coverage_radius
    
    def get_signal_strength(self, robot_pos: Position) -> float:
        """
        Signal strength in dBm using the log-distance path-loss model
        (Rappaport, Wireless Communications):
            S(dBm) = S_max - 10*beta*log10(d / d_max)
        Returns -inf when out of coverage range.
        """
        distance = self.position.distance_to(robot_pos)
        if distance == 0:
            distance = 1e-9  # avoid log10(0)
        if distance > self.coverage_radius:
            return float('-inf')
        return self.s_max - 10 * self.beta * math.log10(distance / self.coverage_radius)


@dataclass
class MobileRobot:
    """Mobile robot with position and velocity"""
    host_id: str
    position: Position = field(default_factory=lambda: Position(0, 0))
    velocity: Position = field(default_factory=lambda: Position(0, 0))
    connected_ap: Optional[str] = None
    
    def update_position(self, dt: float) -> None:
        """Update position based on velocity and time delta"""
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
    
    def set_velocity(self, vx: float, vy: float) -> None:
        """Set robot velocity"""
        self.velocity.x = vx
        self.velocity.y = vy


@dataclass
class WiFiLink:
    """Represents a WiFi link with variable quality"""
    link_id: str
    robot_id: str
    # SNR levels: [good, medium, poor]
    bandwidths: List[str] = field(default_factory=lambda: ["600Mbps", "300Mbps", "100Mbps"])
    latencies: List[float] = field(default_factory=lambda: [5e-3, 10e-3, 20e-3])  # in seconds
    current_snr_level: int = 0  # 0=good, 1=medium, 2=poor


class WiFiMobilityController:
    """
    Controller for WiFi network mobility simulation.
    Manages robot positions, WiFi access points, and link quality adjustments.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.access_points: Dict[str, WifiAccessPoint] = {}
        self.robots: Dict[str, MobileRobot] = {}
        self.wifi_links: Dict[str, WiFiLink] = {}
        self.simulation_time: float = 0.0
        self.update_interval: float = 1.0  # Update every 1 second of simulation
        self.last_update_time: float = -float('inf')  # Force first update
    
    def register_access_point(self, host_id: str, x: float, y: float,
                             coverage_radius: float,
                             s_max: float = -70.0,
                             beta: float = 2.0) -> None:
        """Register a WiFi access point"""
        ap = WifiAccessPoint(
            host_id=host_id,
            position=Position(x, y),
            coverage_radius=coverage_radius,
            s_max=s_max,
            beta=beta
        )
        self.access_points[host_id] = ap
    
    def register_robot(self, host_id: str, x: float, y: float, 
                      vx: float = 0.0, vy: float = 0.0) -> None:
        """Register a mobile robot"""
        robot = MobileRobot(
            host_id=host_id,
            position=Position(x, y),
            velocity=Position(vx, vy)
        )
        self.robots[host_id] = robot
    
    def register_wifi_link(self, link_id: str, robot_id: str,
                          bandwidths: Optional[List[str]] = None,
                          latencies: Optional[List[float]] = None) -> None:
        """Register a WiFi link for a robot"""
        link = WiFiLink(
            link_id=link_id,
            robot_id=robot_id,
            bandwidths=bandwidths or ["600Mbps", "300Mbps", "100Mbps"],
            latencies=latencies or [5e-3, 10e-3, 20e-3]
        )
        self.wifi_links[link_id] = link
        self.logger.debug(f"Registered WiFi link '{link_id}' for robot '{robot_id}'")
    
    def set_robot_velocity(self, robot_id: str, vx: float, vy: float) -> None:
        """Set a robot's velocity"""
        if robot_id not in self.robots:
            self.logger.warning(f"Robot '{robot_id}' not found")
            return
        self.robots[robot_id].set_velocity(vx, vy)
        self.logger.info(f"Set velocity for robot '{robot_id}': ({vx}, {vy}) m/s")
    
    def update(self, current_time: float) -> Dict[str, Tuple[int, str, float]]:
        """
        Update robot positions and WiFi link qualities.
        
        Returns:
            Dictionary mapping link_id to (snr_level, best_ap, signal_strength)
        """
        # Check if enough time has passed for an update
        is_first_update = self.last_update_time == -float('inf')
        if not is_first_update and current_time - self.last_update_time < self.update_interval:
            return {}
        
        self.simulation_time = current_time
        # Don't move robots on first update, just report initial status
        dt = 0.0 if is_first_update else (current_time - self.last_update_time)
        self.last_update_time = current_time
        
        updates = {}
        
        # Update robot positions
        for robot_id, robot in self.robots.items():
            if dt > 0:
                robot.update_position(dt)
            self.logger.debug(f"Robot '{robot_id}' position: ({robot.position.x:.1f}, "
                            f"{robot.position.y:.1f})")
            
            # Find best access point based on signal strength
            best_ap = None
            best_ap_obj: Optional[WifiAccessPoint] = None
            best_signal = float('-inf')
            best_snr = 2  # worst case: poor

            for ap_id, ap in self.access_points.items():
                signal = ap.get_signal_strength(robot.position)

                if signal > best_signal:
                    best_signal = signal
                    best_ap = ap_id
                    best_ap_obj = ap

            # Map best signal to SNR level using AP parameters
            if best_ap_obj is not None and best_signal > float('-inf'):
                good_thresh = best_ap_obj.s_max + 10 * best_ap_obj.beta * 0.75
                med_thresh  = best_ap_obj.s_max + 10 * best_ap_obj.beta * 0.50
                if best_signal >= good_thresh:
                    best_snr = 0  # good
                elif best_signal >= med_thresh:
                    best_snr = 1  # medium
                else:
                    best_snr = 2  # poor

            # Update robot's connected AP
            if best_ap and best_signal > float('-inf'):
                robot.connected_ap = best_ap
                self.logger.info(f"Robot '{robot_id}' connected to AP '{best_ap}' "
                               f"(signal: {best_signal:.1f} dBm)")
            else:
                robot.connected_ap = None
                self.logger.warning(f"Robot '{robot_id}' lost WiFi connection")
            
            # Update associated WiFi link quality
            for link_id, link in self.wifi_links.items():
                if link.robot_id == robot_id:
                    old_snr = link.current_snr_level
                    link.current_snr_level = best_snr
                    
                    if old_snr != best_snr:
                        snr_names = ["Good", "Medium", "Poor"]
                        self.logger.info(f"WiFi link '{link_id}' changed from "
                                       f"{snr_names[old_snr]} to {snr_names[best_snr]}")
                    
                    updates[link_id] = (best_snr, best_ap or "none", best_signal)
        
        return updates
    
    def get_link_characteristics(self, link_id: str) -> Dict[str, any]:
        """Get current bandwidth and latency for a WiFi link"""
        if link_id not in self.wifi_links:
            return None
        
        link = self.wifi_links[link_id]
        snr = link.current_snr_level
        
        return {
            "link_id": link_id,
            "snr_level": snr,
            "bandwidth": link.bandwidths[snr],
            "latency": link.latencies[snr],
            "robot_id": link.robot_id,
            "connected_ap": self.robots[link.robot_id].connected_ap if link.robot_id in self.robots else None
        }
    
    def get_robot_status(self, robot_id: str) -> Optional[Dict]:
        """Get current status of a robot"""
        if robot_id not in self.robots:
            return None
        
        robot = self.robots[robot_id]
        return {
            "host_id": robot_id,
            "position": {"x": robot.position.x, "y": robot.position.y},
            "velocity": {"x": robot.velocity.x, "y": robot.velocity.y},
            "connected_ap": robot.connected_ap,
            "speed": math.sqrt(robot.velocity.x**2 + robot.velocity.y**2)
        }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all robots"""
        return {
            robot_id: self.get_robot_status(robot_id)
            for robot_id in self.robots.keys()
        }
    
    def set_update_interval(self, interval: float) -> None:
        """Set the update interval for position/quality updates"""
        self.update_interval = interval
        self.logger.info(f"WiFi update interval set to {interval} seconds")
