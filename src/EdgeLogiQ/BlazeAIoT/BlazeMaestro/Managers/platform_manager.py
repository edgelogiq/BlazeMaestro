"""
Platform Manager Module

Handles platform initialization (Kubernetes or Simulator) and topology discovery.
"""

from BlazeAIoT.Core.Utils import BlazeLogger
from typing import Tuple, List
import os

from ..Topology.platform import PlatformTopologyManager
from ..Topology.kubernetes_platform import MultiKubernetesPlatform


class PlatformManager:
    """Manages platform initialization and topology discovery."""
    
    def __init__(self, config: dict):
        """
        Initialize Platform Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.topology_manager = None
        self.layers = None
        self.nodes = None
        self.simpy_env = None
        self.platform_xml_path = None
        self._logger = BlazeLogger.get_instance()
    
    def initialize_platform(self) -> Tuple[List, List]:
        """
        Initialize platform (Kubernetes or Simulator) and discover topology.
        
        Returns:
            Tuple of (layers, schedulable_nodes)
        """
        env_config = self.config.get('environment', {})
        mode = env_config.get('mode', 'kubernetes')
        
        if mode == 'simulator':
            self._init_simulator_platform(env_config)
        else:
            self._init_kubernetes_platform(env_config)
        
        return self.layers, self.nodes
    
    def _init_simulator_platform(self, env_config: dict):
        """Initialize simulator platform."""
        print("\n[1/5] Initializing Simulator environment...")
        
        sim_config = env_config.get('simulator', {})
        platform_xml = sim_config.get('platform_xml', 'src/config/platform.xml')
        
        # Resolve relative paths
        if not os.path.isabs(platform_xml):
            project_root = self._get_project_root()
            platform_xml = os.path.join(project_root, platform_xml)
        
        # Always load from the configured file (not _update.xml)
        # Updates will be written to *_update.xml by the config updaters
        
        if not os.path.exists(platform_xml):
            raise FileNotFoundError(f"Platform XML file not found: {platform_xml}")
        
        print(f"  Loading topology from: {platform_xml}")
        self.platform_xml_path = platform_xml
        
        # Import simulator dependencies
        import simpy
        from ..Topology.simulator_platform import SimulatorPlatform
        
        # Create SimPy environment
        self.simpy_env = simpy.Environment()
        
        # Create simulator platform
        platform = SimulatorPlatform(self.simpy_env, platform_xml)
        self.topology_manager = PlatformTopologyManager(platform)
        
        # Discover topology
        self.layers, all_nodes = self.topology_manager.load_topology()
        
        # Filter schedulable nodes
        self.nodes = self._filter_schedulable_nodes(all_nodes)
        
        print(f"  ✓ Loaded {len(self.layers)} layers with {len(self.nodes)} schedulable nodes (total: {len(all_nodes)})")

        # Log mobile/battery-powered node power details
        mobile_nodes = [
            n for n in self.nodes
            if getattr(n, 'battery_powered', False)
            or str(getattr(n, 'layer_id', '')).lower() == 'mobile'
            or str(getattr(n, 'node_id', '')).startswith('frank-')
        ]
        if mobile_nodes:
            self._logger.info("Mobile/battery-powered node power profile:")
            for n in mobile_nodes:
                node_id = getattr(n, 'node_id', 'unknown')
                power_base = getattr(n, 'power_base', None)
                power_active = getattr(n, 'power_active', None)
                battery_voltage = getattr(n, 'battery_voltage', None)
                battery_capacity_ah = getattr(n, 'battery_capacity_ah', None)
                self._logger.info(
                    f"  {node_id}: power_base={power_base}W, power_active={power_active}W, "
                    f"battery_voltage={battery_voltage}V, battery_capacity_ah={battery_capacity_ah}Ah"
                )
    
    def _init_kubernetes_platform(self, env_config: dict):
        """Initialize Kubernetes platform."""
        print("\n[1/5] Connecting to Kubernetes clusters...")
        print("[DEBUG] KUBERNETES MODE ACTIVE")
        
        # Support both new and legacy cluster config locations
        cluster_configs = env_config.get('kubernetes', {}).get('clusters', [])
        if not cluster_configs:
            cluster_configs = self.config.get('clusters', [])
        
        if not cluster_configs:
            print("  ⚠ No 'clusters' configured in YAML. Skipping Kubernetes platform initialization.")
            print("    Tip: Create a config with clusters via: python blaze_maestro_main.py --create-config")
            self.topology_manager = None
            self.layers, self.nodes = [], []
            return
        
        platform = MultiKubernetesPlatform(cluster_configs)
        self.topology_manager = PlatformTopologyManager(platform)
        
        # Discover topology
        self.layers, all_nodes = self.topology_manager.load_topology()
        print(f"  [DEBUG] Kubernetes topology loaded: {len(all_nodes)} total nodes from discovery")
        
        # Filter schedulable nodes
        self.nodes = self._filter_schedulable_nodes(all_nodes)
        
        # Enrich Kubernetes nodes with battery properties from platform.xml if available
        self._enrich_nodes_with_platform_properties()
        
        print(f"  ✓ Discovered {len(self.layers)} layers with {len(self.nodes)} schedulable nodes (total: {len(all_nodes)})")
        
        # Log mobile/battery-powered node power details
        mobile_nodes = [n for n in self.nodes if getattr(n, 'battery_powered', False)]
        if mobile_nodes:
            self._logger.info("Mobile/battery-powered node power profile:")
            for n in mobile_nodes:
                node_id = getattr(n, 'node_id', 'unknown')
                battery_voltage = getattr(n, 'battery_voltage', None)
                battery_capacity_ah = getattr(n, 'battery_capacity_ah', None)
                self._logger.info(f"  {node_id}: battery_voltage={battery_voltage}V, battery_capacity_ah={battery_capacity_ah}Ah")
    
    def _filter_schedulable_nodes(self, all_nodes: List) -> List:
        """
        Filter nodes to only include schedulable ones.
        
        Args:
            all_nodes: List of all discovered nodes
            
        Returns:
            List of schedulable nodes
        """
        # Filter non-schedulable nodes
        nodes = [n for n in all_nodes if getattr(n, 'schedulable', True)]
        num_filtered = len(all_nodes) - len(nodes)
        
        if num_filtered > 0:
            filtered_ids = [str(getattr(n, 'node_id', '')) for n in all_nodes if n not in nodes]
            print(f"  Pre-filtered {num_filtered} non-schedulable node(s): {', '.join(filtered_ids)}")
        
        # Additional explicit filters from config
        explicit_unsched = set(self.config.get('unschedulable_nodes', []))
        if explicit_unsched:
            before = len(nodes)
            nodes = [n for n in nodes if str(getattr(n, 'node_id', '')) not in explicit_unsched]
            removed = before - len(nodes)
            if removed > 0:
                print(f"  Pre-filtered {removed} node(s) via config.unschedulable_nodes: {', '.join(sorted(explicit_unsched))}")
        
        return nodes
    
    def _enrich_nodes_with_platform_properties(self):
        """Load battery and power properties from platform.xml and apply to discovered Kubernetes nodes."""
        try:
            import xml.etree.ElementTree as ET
            
            sim_config = self.config.get('simulator', {})
            platform_xml = sim_config.get('platform_xml', 'src/config/platform.xml')
            
            # Resolve relative paths
            if not os.path.isabs(platform_xml):
                project_root = self._get_project_root()
                platform_xml = os.path.join(project_root, platform_xml)
            
            if not os.path.exists(platform_xml):
                return  # platform.xml not available, skip enrichment
            
            # Parse platform.xml to extract host properties
            tree = ET.parse(platform_xml)
            root = tree.getroot()
            
            host_props = {}
            for host_elem in root.findall('.//host'):
                host_id = host_elem.get('id')
                if not host_id:
                    continue
                
                props = {}
                for prop_elem in host_elem.findall('prop'):
                    prop_id = prop_elem.get('id')
                    prop_value = prop_elem.get('value')
                    
                    if prop_id and prop_value:
                        # Store schedulable property
                        if prop_id == 'schedulable':
                            props['schedulable'] = str(prop_value).lower() == 'true'
                        # Store battery properties
                        elif prop_id == 'battery.powered':
                            props['battery_powered'] = str(prop_value).lower() == 'true'
                        elif prop_id == 'battery.voltage':
                            props['battery_voltage'] = float(prop_value)
                        elif prop_id == 'battery.capacity.ah':
                            props['battery_capacity_ah'] = float(prop_value)
                        elif prop_id == 'power.jetson.base':
                            props['power_base'] = float(prop_value)
                        elif prop_id == 'power.jetson.active':
                            props['power_active'] = float(prop_value)
                
                if props:
                    host_props[host_id] = props
            
            # Apply loaded properties to discovered nodes
            for node in self.nodes:
                node_id = getattr(node, 'node_id', None)
                if node_id and node_id in host_props:
                    for prop_key, prop_value in host_props[node_id].items():
                        setattr(node, prop_key, prop_value)
                        self._logger.debug(f"Set {prop_key}={prop_value} on node {node_id}")
            
        except Exception as e:
            self._logger.warning(f"Failed to enrich nodes with platform properties: {e}")
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        # This file is at: maestro/src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/Managers/platform_manager.py
        # Project root is: maestro/
        # Need to go up 6 levels: Managers -> BlazeMaestro -> BlazeAIoT -> EdgeLogiQ -> src -> maestro
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
