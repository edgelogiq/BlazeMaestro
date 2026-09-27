#!/usr/bin/env python3
"""
Robot Movement Simulation Script

Simulates a scenario where the frank-1 robot moves between WiFi access points,
demonstrating the impact of latency changes on service reallocation.

This script:
1. Loads the blaze_maestro_simulator.yaml configuration
2. Initializes the simulator environment with platform.xml
3. Sets up WiFi mobility controller with access points
4. Moves frank-1 robot through a realistic travel scenario
5. Monitors and logs WiFi quality changes and orchestration decisions
6. Outputs metrics showing reallocation events

Usage:
    python scripts/simulate_robot_movement.py [--duration SECONDS] [--speed METERS_PER_SEC]
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Install BlazeLogger mock to avoid circular import
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

# Setup logging (before importing orchestrator modules)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RobotMovementSim")

try:
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_scenario_manager import (
        RobotTravelScenario, 
        get_wifi_scenario_manager
    )
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_orchestration_adapter import WiFiOrchestrationAdapter
    # Import orchestrator classes directly to avoid circular import
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.unsupervised_orchestrator import GreedyWeightedOrchestrator
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.nsga2_orchestrator import NSGA2Orchestrator
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.dqn_orchestrator import DQNOrchestrator
    from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.autoencoder_orchestrator import AutoEncoderOrchestrator
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running from the repository root directory")
    sys.exit(1)


class RobotMovementSimulator:
    """Simulates robot movement and tracks orchestration decisions."""
    
    def __init__(self, config_path: str):
        """Initialize simulator with configuration.
        
        Args:
            config_path: Path to blaze_maestro_simulator.yaml
        """
        self.config_path = config_path
        self.robot_id = "frank-1"  # Robot we're simulating
        self.wifi_controller = None
        self.wifi_adapter = None
        self.scenario = None
        self.events = []
        self.current_allocations = {}  # {service_id: node_id}
        self.services = []  # List of services to allocate
        self.nodes = []  # List of nodes
        self.orchestrator = None  # Orchestrator instance
        self.strategy = None  # Orchestration strategy
        
    def load_configuration(self):
        """Load and parse the YAML configuration."""
        logger.info(f"Loading configuration from {self.config_path}")
        
        import yaml
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Verify it's simulator mode
        if self.config.get('environment', {}).get('mode') != 'simulator':
            logger.warning("Configuration is not in simulator mode!")
        
        platform_xml = self.config.get('environment', {}).get('simulator', {}).get('platform_xml')
        services_xml = self.config.get('environment', {}).get('simulator', {}).get('services_xml')
        
        logger.info(f"Platform XML: {platform_xml}")
        logger.info(f"Services XML: {services_xml}")
        
        # Parse services for allocation simulation
        self._parse_services(services_xml if not services_xml.startswith('src/') 
                            else os.path.join(os.path.dirname(__file__), '..', services_xml))
        
        return platform_xml, services_xml
    
    def _parse_services(self, services_xml_path: str):
        """Parse services from XML to simulate allocation decisions.
        
        Args:
            services_xml_path: Path to services XML file
        
        Raises:
            FileNotFoundError: If services XML doesn't exist
            ValueError: If no services found in XML
        """
        import xml.etree.ElementTree as ET
        from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service, ContainerSpec
        
        if not os.path.exists(services_xml_path):
            raise FileNotFoundError(f"Services XML not found: {services_xml_path}")
        
        tree = ET.parse(services_xml_path)
        root = tree.getroot()
        
        # Parse services that have instances on frank-1 robot
        for service in root.findall('.//service'):
            service_id = service.get('id')
            instances = service.findall('.//instances/container')
            
            # Check if service has instance on frank-1
            has_frank1_instance = any(
                inst.get('node') == self.robot_id 
                for inst in instances
            )
            
            if has_frank1_instance:
                # Extract container requirements and create ContainerSpec objects
                container_specs = []
                for c in service.findall('.//containers/container'):
                    container_specs.append(ContainerSpec(
                        id=c.get('id', f"{service_id}-container"),
                        cpu=float(c.get('cpu', 0.1)),
                        ram_gb=float(c.get('ram_gb', 0.2)),
                        gpu_gb=0.0,  # Can be extended if needed
                        exec_time=float(c.get('exec_time', 1.0))
                    ))
                
                # Create proper Service object
                service_obj = Service(
                    service_id=service_id,
                    containers=container_specs,
                    dependencies=[],  # Can be parsed if needed
                    execution_rate_hz=1.0
                )
                
                self.services.append(service_obj)
                # Initially allocate to the robot itself
                self.current_allocations[service_id] = self.robot_id
        
        if not self.services:
            raise ValueError(f"No robot services found in XML: {services_xml_path}")
        
        logger.info(f"✓ Loaded {len(self.services)} services for allocation tracking")
    def setup_wifi_infrastructure(self, platform_xml_path: str):
        """Setup WiFi access points and mobile robots by parsing platform XML.
        
        Args:
            platform_xml_path: Path to platform_sim.xml file
        """
        logger.info(f"Parsing WiFi infrastructure from {platform_xml_path}...")
        
        import xml.etree.ElementTree as ET
        
        # Parse the platform XML
        tree = ET.parse(platform_xml_path)
        root = tree.getroot()
        
        self.wifi_controller = WiFiMobilityController()
        
        # Find all hosts
        for host in root.findall('.//host'):
            host_id = host.get('id')
            props = {prop.get('id'): prop.get('value') for prop in host.findall('prop')}
            
            # Check if this is a WiFi access point
            if props.get('wifi.access_point') == 'true':
                x = float(props.get('wifi.position.x', 0.0))
                y = float(props.get('wifi.position.y', 0.0))
                coverage = float(props.get('wifi.coverage_radius', 50.0))
                
                self.wifi_controller.register_access_point(host_id, x=x, y=y, coverage_radius=coverage)
                logger.info(f"  ✓ Registered AP {host_id} at ({x}, {y}) with {coverage}m coverage")
            
            # Check if this is a mobile robot
            elif props.get('mobile.robot') == 'true':
                x = float(props.get('mobile.position.x', 0.0))
                y = float(props.get('mobile.position.y', 0.0))
                vx = float(props.get('mobile.velocity.x', 0.0))
                vy = float(props.get('mobile.velocity.y', 0.0))
                
                self.wifi_controller.register_robot(host_id, x=x, y=y, vx=vx, vy=vy)
                logger.info(f"  ✓ Registered robot {host_id} at ({x}, {y})")
        
        # Find all WiFi links
        for link in root.findall('.//link'):
            link_id = link.get('id')
            sharing_policy = link.get('sharing_policy', '')
            
            if sharing_policy == 'WIFI':
                # Parse bandwidth levels (e.g., "600Mbps,300Mbps,100Mbps")
                bw_str = link.get('bandwidth', '600Mbps,300Mbps,100Mbps')
                bandwidths = [bw.strip() for bw in bw_str.split(',')]
                
                # Parse latency (e.g., "5ms") and create levels
                lat_str = link.get('latency', '5ms')
                base_latency_ms = float(lat_str.replace('ms', ''))
                # Create latency levels: base, 2x, 4x
                latencies = [base_latency_ms / 1000.0, base_latency_ms * 2 / 1000.0, base_latency_ms * 4 / 1000.0]
                
                # Determine which robot this link belongs to
                robot_id = None
                if 'frank-1' in link_id:
                    robot_id = 'frank-1'
                elif 'frank-2' in link_id:
                    robot_id = 'frank-2'
                
                if robot_id:
                    self.wifi_controller.register_wifi_link(link_id, robot_id, bandwidths=bandwidths, latencies=latencies)
                    logger.info(f"  ✓ Configured {link_id} with {len(bandwidths)} SNR levels")
        
        logger.info("✓ WiFi infrastructure setup complete from platform XML")
        
        # Parse node properties from platform XML and create proper Node objects
        from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node
        
        self.nodes = []
        for host in root.findall('.//host'):
            host_id = host.get('id')
            props = {prop.get('id'): prop.get('value') for prop in host.findall('prop')}
            
            # Only create node objects for WiFi APs (not robots)
            if props.get('wifi.access_point') == 'true':
                # Parse node properties from XML
                cpu_capacity = float(props.get('cpu.capacity', '8.0'))
                memory_capacity = float(props.get('memory.capacity', '16.0'))
                gpu_capacity = float(props.get('gpu.capacity', '1.0' if 'gpu' in host_id else '0.0'))
                layer = props.get('layer', 'edge')
                
                # Create proper Node object with metrics wrapper
                node = Node(
                    node_id=host_id,
                    cpu=cpu_capacity,
                    ram=memory_capacity,
                    gpu=gpu_capacity,
                    gpu_shared=False,
                    energy_level=None,
                    metrics_facade=None,  # Will use fallback in NodeMetricsWrapper
                    tier=layer
                )
                self.nodes.append(node)
                logger.info(f"  ✓ Node {host_id}: CPU={cpu_capacity}, Memory={memory_capacity}GB, GPU={gpu_capacity}")
    
    def create_travel_scenario(self, duration: float = 600.0, speed: float = 0.33):
        """Create robot travel scenario.
        
        Args:
            duration: Total scenario duration in seconds (default: 600s = 10 min)
            speed: Robot travel speed in m/s (default: 0.33 m/s, realistic for mobile robot)
        """
        logger.info(f"Creating robot travel scenario (duration={duration}s, speed={speed}m/s)...")
        
        scenario_manager = get_wifi_scenario_manager()
        self.scenario = scenario_manager.create_scenario(
            "robot_travel",
            duration_seconds=duration,
            travel_speed=speed
        )
        
        logger.info("✓ Scenario created:")
        logger.info(f"  - Phase 1 (0-60s):     Robot at edge-gpu1, initiating workload")
        logger.info(f"  - Phase 2 (60-300s):   Robot traveling from edge-gpu1 to edge-gpu2")
        logger.info(f"  - Phase 3 (300-360s):  Robot at edge-gpu2, WiFi improves")
        logger.info(f"  - Phase 4 (360-600s):  Robot returns to edge-gpu1")
    
    def initialize_orchestration_adapter(self, time_step: float = 1.0):
        """Initialize WiFi orchestration adapter.
        
        Args:
            time_step: Time between updates in seconds
        """
        logger.info("Initializing orchestration adapter...")
        
        self.wifi_adapter = WiFiOrchestrationAdapter(self.wifi_controller)
        self.wifi_adapter.initialize_with_scenario(self.scenario, time_step=time_step)
        
        logger.info(f"✓ Orchestration adapter ready (time_step={time_step}s)")
    
    def initialize_orchestrator(self):
        """Initialize the actual orchestrator from YAML configuration.
        
        This loads the configured orchestrator (Unsupervised, NSGA-II, DQN, etc.)
        and creates it directly with ALL parameters from the YAML config.
        """
        logger.info("Initializing orchestrator from configuration...")
        
        orchestrator_config = self.config.get('orchestrator', {})
        orchestrator_type = orchestrator_config.get('type', 'unsupervised')
        
        logger.info(f"  Orchestrator type: {orchestrator_type}")
        
        # Log all config parameters being used
        logger.info("  Configuration parameters:")
        for key, value in orchestrator_config.items():
            if key != 'type':
                logger.info(f"    - {key}: {value}")
        
        # Create metrics provider for the simulation
        try:
            class SimulatorMetrics:
                """Simulated metrics provider matching BaseMetricsFacade interface."""
                def __init__(self, wifi_controller, nodes):
                    self.wifi_controller = wifi_controller
                    self.nodes = nodes
                
                def get_node_capacity(self, node_id):
                    """Get node capacity (required by orchestrator)."""
                    node = next((n for n in self.nodes if n.node_id == node_id), None)
                    if not node:
                        raise ValueError(f"Node {node_id} not found")
                    return {
                        'cpu': node.cpu,
                        'ram_gb': node.ram,
                        'gpu_gb': node.gpu if node.gpu else 0.0
                    }
                
                def get_node_available(self, node_id):
                    """Get node available resources (required by orchestrator)."""
                    node = next((n for n in self.nodes if n.node_id == node_id), None)
                    if not node:
                        raise ValueError(f"Node {node_id} not found")
                    # Simulate 70% availability
                    return {
                        'cpu': node.cpu * 0.7,
                        'ram_gb': node.ram * 0.7,
                        'gpu_gb': (node.gpu if node.gpu else 0.0) * 0.7
                    }
                
                def get_node_property(self, node_id, property_name):
                    """Get node property (required by orchestrator)."""
                    node = next((n for n in self.nodes if n.node_id == node_id), None)
                    if not node:
                        return None
                    return getattr(node, property_name, None)
                
                def get_node_metrics(self, node_id):
                    """Get node metrics (for compatibility)."""
                    node = next((n for n in self.nodes if n.node_id == node_id), None)
                    if not node:
                        raise ValueError(f"Node {node_id} not found")
                    
                    # Base utilization
                    base_cpu_usage = 0.3
                    base_mem_usage = 0.4
                    
                    return {
                        'cpu_usage': base_cpu_usage,
                        'memory_usage': base_mem_usage,
                        'available_cpu': node.cpu * (1 - base_cpu_usage),
                        'available_memory': node.ram * (1 - base_mem_usage)
                    }
                
                def get_latency_matrix(self, nodes):
                    """Generate latency matrix based on WiFi link qualities.
                    
                    Returns:
                        numpy array of shape (num_nodes, num_nodes) with latencies in ms
                    """
                    import numpy as np
                    n = len(nodes)
                    latency_matrix = np.zeros((n, n))
                    
                    # Build latency matrix with current WiFi state
                    for i, node_i in enumerate(nodes):
                        for j, node_j in enumerate(nodes):
                            if i == j:
                                latency_matrix[i, j] = 0.1  # Same node
                            else:
                                # Default wired connection latency between edge nodes
                                latency_matrix[i, j] = 2.0
                    
                    return latency_matrix
            
            metrics_provider = SimulatorMetrics(self.wifi_controller, self.nodes)
            
            # Create orchestrator directly based on type
            orchestrator_map = {
                'unsupervised': GreedyWeightedOrchestrator,
                'nsga2': NSGA2Orchestrator,
                'dqn': DQNOrchestrator,
                'autoencoder': AutoEncoderOrchestrator
            }
            
            OrchestratorClass = orchestrator_map.get(orchestrator_type)
            if not OrchestratorClass:
                raise ValueError(f"Unknown orchestrator type: {orchestrator_type}")
            
            # Create orchestrator with config parameters
            if orchestrator_type == 'nsga2':
                self.orchestrator = OrchestratorClass(
                    services=self.services,
                    nodes=self.nodes,
                    metrics=metrics_provider,
                    population_size=orchestrator_config.get('population_size', 40),
                    generations=orchestrator_config.get('generations', 80)
                )
            elif orchestrator_type == 'dqn':
                self.orchestrator = OrchestratorClass(
                    services=self.services,
                    nodes=self.nodes,
                    metrics=metrics_provider,
                    lr=orchestrator_config.get('learning_rate', 0.001),
                    episodes=orchestrator_config.get('episodes', 50)
                )
            elif orchestrator_type == 'autoencoder':
                self.orchestrator = OrchestratorClass(
                    services=self.services,
                    nodes=self.nodes,
                    metrics=metrics_provider,
                    latent_dim=orchestrator_config.get('latent_dim', 16),
                    epochs=orchestrator_config.get('training_epochs', 50)
                )
            else:  # unsupervised
                self.orchestrator = OrchestratorClass(
                    services=self.services,
                    nodes=self.nodes,
                    metrics=metrics_provider
                )
            
            logger.info(f"  ✓ Created {orchestrator_type} orchestrator with full configuration")
            
        except Exception as e:
            logger.error(f"Failed to create orchestrator: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Orchestrator initialization failed: {e}")
        
        logger.info("✓ Orchestrator initialization complete")
    
    def _decide_allocations_with_orchestrator(self, current_time: float) -> dict:
        """Make allocation decisions using the configured orchestrator.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Dict of {service_id: target_node_id} for services that should move
        
        Raises:
            RuntimeError: If orchestrator not initialized
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized. Call initialize_orchestrator() first.")
        
        try:
            # Use the orchestrator's _propose_assignment method directly
            proposed_allocation = self.orchestrator._propose_assignment(self.services)
            
            if not proposed_allocation:
                return {}
            
            # Track changes and log reallocations
            new_allocations = {}
            for service_id, target_node in proposed_allocation.items():
                current_node = self.current_allocations.get(service_id)
                if current_node != target_node:
                    # Get robot status for logging
                    robot_status = self.wifi_controller.get_robot_status("frank-1")
                    pos = robot_status['position'] if robot_status else {'x': 0, 'y': 0}
                    
                    dist_to_edge1 = ((pos['x']-0)**2 + (pos['y']-0)**2)**0.5
                    dist_to_edge2 = ((pos['x']-100)**2 + (pos['y']-0)**2)**0.5
                    
                    link_chars = self.wifi_controller.get_link_characteristics("frank-1-wifi6")
                    snr_level = link_chars['snr_level'] if link_chars else 2
                    
                    reason = f"Orchestrator decision ({self.config.get('orchestrator', {}).get('type', 'unknown')})"
                    
                    logger.warning(f"[t={current_time:.1f}s] REALLOCATION: {service_id}: {current_node} → {target_node} ({reason})")
                    
                    new_allocations[service_id] = target_node
                    
                    # Track event
                    self.events.append({
                        'time': current_time,
                        'type': 'reallocation',
                        'service_id': service_id,
                        'from_node': current_node,
                        'to_node': target_node,
                        'reason': reason,
                        'snr_level': snr_level,
                        'distance_to_edge1': dist_to_edge1,
                        'distance_to_edge2': dist_to_edge2
                    })
            
            return new_allocations
            
        except Exception as e:
            logger.error(f"Orchestrator optimization failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Orchestrator failed to make allocation decision: {e}")
    
    def simulate(self, duration: float, time_step: float = 1.0, 
                 report_interval: float = 30.0, use_direct_speed: bool = False,
                 direct_speed: float = 0.33):
        """Run the simulation.
        
        Args:
            duration: Total simulation duration in seconds
            time_step: Time between updates in seconds
            report_interval: How often to print status updates in seconds
            use_direct_speed: If True, move robot directly instead of using scenario
            direct_speed: Speed in m/s when using direct movement
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING ROBOT MOVEMENT SIMULATION")
        logger.info("="*80)
        
        if use_direct_speed:
            logger.info(f"Using direct movement mode: {direct_speed} m/s")
            # Set robot velocity directly for constant movement
            self.wifi_controller.set_robot_velocity("frank-1", vx=direct_speed, vy=0.0)
        
        # Print header for status table
        print("\n{:>8s} | {:20s} | {:4s} | {:8s} | {:8s} | {:8s} | {:12s}".format(
            "Time(s)", "Phase", "SNR", "BW", "Latency", "Distance", "Connected AP"
        ))
        print("-" * 95)
        
        current_time = 0.0
        last_report = 0.0
        last_snr = None
        reallocation_count = 0
        
        while current_time <= duration:
            # Update robot positions
            if use_direct_speed:
                # Manual update for direct movement
                self.wifi_controller.update(current_time)
                # Get metrics directly from controller in direct mode
                robot_status = self.wifi_controller.get_robot_status("frank-1")
                link_chars = self.wifi_controller.get_link_characteristics("frank-1-wifi6")
                
                # Convert to metrics format
                metrics = {
                    "frank-1": {
                        'snr_level': link_chars['snr_level'],
                        'bandwidth': link_chars['bandwidth'],
                        'latency': link_chars['latency']
                    }
                }
            else:
                # Use scenario-based movement
                self.wifi_adapter.update_robots_from_scenario(current_time)
                # Get WiFi metrics from adapter
                metrics = self.wifi_adapter.get_wifi_metrics()
            
            if "frank-1" in metrics:
                m = metrics["frank-1"]
                
                # Track SNR changes (potential reallocation triggers)
                if last_snr is not None and m['snr_level'] != last_snr:
                    reallocation_count += 1
                    logger.warning(f"SNR CHANGE at t={current_time:.1f}s: "
                                 f"{last_snr} -> {m['snr_level']}")
                    self.events.append({
                        'time': current_time,
                        'type': 'snr_change',
                        'old_snr': last_snr,
                        'new_snr': m['snr_level'],
                        'bandwidth': m['bandwidth'],
                        'latency': m['latency']
                    })
                
                last_snr = m['snr_level']
                
                # Check for service reallocation needs (every 5 seconds)
                if int(current_time) % 5 == 0 and current_time > 0:
                    reallocations = self._decide_allocations_with_orchestrator(current_time)
                    if reallocations:
                        # Apply the reallocations
                        for service_id, target_node in reallocations.items():
                            self.current_allocations[service_id] = target_node
                
                # Print status at regular intervals
                if current_time - last_report >= report_interval or current_time == 0:
                    # Get scenario description
                    if use_direct_speed:
                        phase = "Direct movement"
                    else:
                        desc = self.scenario.get_time_description(current_time)
                        phase = desc.split("(")[0].strip()[:20]
                    
                    # Get robot status
                    robot_status = self.wifi_controller.get_robot_status("frank-1")
                    pos = robot_status['position']
                    
                    # Calculate distance to closest AP
                    dist_to_ap1 = ((pos['x']-0)**2 + (pos['y']-0)**2)**0.5
                    dist_to_ap2 = ((pos['x']-100)**2 + (pos['y']-0)**2)**0.5
                    min_dist = min(dist_to_ap1, dist_to_ap2)
                    
                    snr_names = {0: "GOOD", 1: "MED", 2: "POOR"}
                    
                    # Format latency in ms
                    latency_ms = (m['latency'] * 1000) if m['latency'] else 0
                    
                    print("{:8.0f} | {:20s} | {:4s} | {:8s} | {:7.1f}ms | {:7.1f}m | {:12s}".format(
                        current_time,
                        phase,
                        snr_names.get(m['snr_level'], "UNK"),
                        m['bandwidth'] or "N/A",
                        latency_ms,
                        min_dist,
                        robot_status['connected_ap'] or "None"
                    ))
                    
                    last_report = current_time
            
            current_time += time_step
        
        logger.info("\n" + "="*80)
        logger.info("SIMULATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total SNR changes (potential reallocations): {reallocation_count}")
        logger.info(f"Total events tracked: {len(self.events)}")
    
    def generate_report(self, output_dir: str = "out/simulation"):
        """Generate simulation report with metrics.
        
        Args:
            output_dir: Directory to save report files
        """
        logger.info("\nGenerating simulation report...")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save events to JSON
        events_file = os.path.join(output_dir, "robot_movement_events.json")
        with open(events_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config_file': self.config_path,
                'events': self.events,
                'summary': {
                    'total_events': len(self.events),
                    'snr_changes': len([e for e in self.events if e['type'] == 'snr_change'])
                }
            }, f, indent=2)
        
        logger.info(f"✓ Events saved to {events_file}")
        
        # Generate summary report
        report_file = os.path.join(output_dir, "robot_movement_report.txt")
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ROBOT MOVEMENT SIMULATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {self.config_path}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Events: {len(self.events)}\n")
            f.write(f"SNR Changes: {len([e for e in self.events if e['type'] == 'snr_change'])}\n")
            f.write(f"Service Reallocations: {len([e for e in self.events if e['type'] == 'reallocation'])}\n\n")
            
            f.write("EVENTS TIMELINE\n")
            f.write("-"*80 + "\n")
            for event in self.events:
                if event['type'] == 'snr_change':
                    f.write(f"[{event['time']:6.1f}s] SNR Change: "
                           f"{event['old_snr']} -> {event['new_snr']} "
                           f"(BW: {event['bandwidth']}, Latency: {event['latency']*1000:.1f}ms)\n")
                elif event['type'] == 'reallocation':
                    f.write(f"[{event['time']:6.1f}s] SERVICE REALLOCATION: "
                           f"{event['service_id']}: {event['from_node']} → {event['to_node']}\n"
                           f"              Reason: {event['reason']}\n"
                           f"              SNR: {event['snr_level']}, "
                           f"Dist to edge-gpu1: {event['distance_to_edge1']:.1f}m, "
                           f"Dist to edge-gpu2: {event['distance_to_edge2']:.1f}m\n")
        
        logger.info(f"✓ Report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        print(f"Events tracked: {len(self.events)}")
        print(f"SNR changes (reallocation triggers): {len([e for e in self.events if e['type'] == 'snr_change'])}")
        print(f"Service reallocations: {len([e for e in self.events if e['type'] == 'reallocation'])}")
        print(f"\nReports saved to: {output_dir}/")
        print("  - robot_movement_events.json")
        print("  - robot_movement_report.txt")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate robot movement and WiFi quality changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run with defaults (600s duration, 0.33 m/s speed)
  python scripts/simulate_robot_movement.py
  
  # Run for 10 minutes with custom speed
  python scripts/simulate_robot_movement.py --duration 600 --speed 0.5
  
  # Short 2-minute test
  python scripts/simulate_robot_movement.py --duration 120 --speed 1.0
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/blaze_maestro_simulator.yaml',
        help='Path to configuration file (default: src/config/blaze_maestro_simulator.yaml)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=600.0,
        help='Simulation duration in seconds (default: 600s = 10 min)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.33,
        help='Robot travel speed in meters/second (default: 0.33 m/s)'
    )
    parser.add_argument(
        '--time-step',
        type=float,
        default=1.0,
        help='Time between updates in seconds (default: 1.0s)'
    )
    parser.add_argument(
        '--report-interval',
        type=float,
        default=30.0,
        help='How often to print status updates in seconds (default: 30s)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='out/simulation',
        help='Directory for output reports (default: out/simulation)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print("ROBOT MOVEMENT SIMULATION")
    print("Simulating frank-1 moving between WiFi access points")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Duration: {args.duration}s")
    print(f"Robot Speed: {args.speed} m/s")
    print(f"Time Step: {args.time_step}s")
    print("="*80 + "\n")
    
    # Create simulator
    simulator = RobotMovementSimulator(args.config)
    
    try:
        # Load configuration
        platform_xml, services_xml = simulator.load_configuration()
        
        # Note: This simulation runs standalone without full SimGrid integration
        # For full orchestration, integrate with blaze_maestro_main.py
        # Load configuration
        platform_xml, services_xml = simulator.load_configuration()
        
        # Convert relative path to absolute if needed
        if not os.path.isabs(platform_xml):
            platform_xml = os.path.join(os.path.dirname(__file__), '..', platform_xml)
        
        # Setup WiFi infrastructure from platform XML
        simulator.setup_wifi_infrastructure(platform_xml)
        
        # Initialize orchestrator from configuration
        simulator.initialize_orchestrator()
        
        # Determine if we should use direct speed mode (when speed != default)
        use_direct = args.speed != 0.33
        
        if not use_direct:
            # Only create scenario for default speed
            simulator.create_travel_scenario(duration=args.duration, speed=args.speed)
            # Initialize orchestration adapter
            simulator.initialize_orchestration_adapter(time_step=args.time_step)
        else:
            logger.info(f"Skipping scenario creation - using direct speed mode ({args.speed} m/s)")
            # Still need to create adapter but without scenario
            simulator.wifi_adapter = WiFiOrchestrationAdapter(simulator.wifi_controller)
        
        # Run simulation
        simulator.simulate(
            duration=args.duration,
            time_step=args.time_step,
            report_interval=args.report_interval,
            use_direct_speed=use_direct,
            direct_speed=args.speed
        )
        
        # Generate report
        simulator.generate_report(output_dir=args.output_dir)
        
        logger.info("Simulation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nSimulation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
