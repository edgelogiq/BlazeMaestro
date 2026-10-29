"""
Service Manager Module

Handles service discovery, loading from XML, and building service configuration.
"""

from BlazeAIoT.Core.Utils import BlazeLogger
from typing import Dict, List
import xml.etree.ElementTree as ET
import os

from ..Topology.service_provider import ServiceManager
from ..Topology.xml_service_provider import XMLServiceProvider


class ServiceDiscoveryManager:
    """Manages service discovery and configuration."""
    
    def __init__(self, config: dict, nodes: List = None, layers: List = None):
        """
        Initialize Service Discovery Manager.
        
        Args:
            config: Configuration dictionary
            nodes: List of available nodes
            layers: List of available layers
        """
        self.config = config
        self.nodes = nodes or []
        self.layers = layers or []
        self.service_manager = None
        self.services = []
        self.locked_services = set()
        self.current_allocation = {}
        self.current_pod_refs = {}
        self._logger = BlazeLogger.get_instance()
    
    def set_topology(self, nodes: List, layers: List) -> None:
        """
        Set the topology (nodes and layers).
        
        Args:
            nodes: List of available nodes
            layers: List of available layers
        """
        self.nodes = nodes or []
        self.layers = layers or []
    
    def discover_services(self, simpy_env=None) -> List:
        """
        Discover and load services.
        
        Args:
            simpy_env: SimPy environment (required for simulator mode)
            
        Returns:
            List of discovered services
        """
        print("\n[2/5] Discovering services...")
        
        xml_path = None
        xml_is_empty = True
        
        try:
            xml_path = self.get_services_xml_path()
            tree = ET.parse(xml_path)
            root = tree.getroot()
            services_in_xml = root.findall('.//service')
            xml_is_empty = len(services_in_xml) == 0
            if xml_is_empty:
                print(f"  ⚠ Services XML exists but is empty: {xml_path}")
                # If update file is empty, try falling back to base file
                if xml_path.endswith('_update.xml'):
                    base_path = xml_path.replace('_update.xml', '.xml')
                    if os.path.exists(base_path):
                        print(f"  → Falling back to base file: {base_path}")
                        xml_path = base_path
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        services_in_xml = root.findall('.//service')
                        xml_is_empty = len(services_in_xml) == 0
        except FileNotFoundError:
            print(f"  ⚠ Services XML not found, will create from Kubernetes")
            xml_path = self.get_services_xml_path_without_validation()
            xml_is_empty = True
        except ET.ParseError as e:
            print(f"  ⚠ Services XML parse error: {e}")
            # If update file has parse error, try falling back to base file
            if xml_path and xml_path.endswith('_update.xml'):
                base_path = xml_path.replace('_update.xml', '.xml')
                if os.path.exists(base_path):
                    print(f"  → Falling back to base file: {base_path}")
                    try:
                        xml_path = base_path
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        services_in_xml = root.findall('.//service')
                        xml_is_empty = len(services_in_xml) == 0
                    except Exception as fallback_err:
                        print(f"  ⚠ Base file also has errors: {fallback_err}")
                        xml_is_empty = True
                else:
                    xml_is_empty = True
            else:
                xml_is_empty = True
        except Exception as e:
            print(f"  ⚠ Error reading services XML: {e}, will rebuild from Kubernetes")
            xml_is_empty = True
        
        # Build from Kubernetes if empty
        if xml_is_empty:
            print("  ⚠ Services XML is empty or missing.")
            print("  Note: Automatic Kubernetes service discovery is not yet implemented.")
            print("  Please provide a valid services.xml file.")
            raise FileNotFoundError(
                f"Services XML is required but missing or empty: {xml_path}\n"
                "Please provide a valid services.xml file in the config directory."
            )
        
        # Load services from XML
        try:
            env_mode = self.config.get('environment', {}).get('mode', 'kubernetes')
            service_provider = XMLServiceProvider(xml_path, simpy_env=simpy_env if env_mode == 'simulator' else None)
            self.service_manager = ServiceManager(service_provider)
            self.services = self.service_manager.load_services()
            print(f"  ✓ Loaded {len(self.services)} services from {os.path.basename(xml_path)}")
            BlazeLogger.get_instance().debug(f"XMLServiceProvider loaded services from: {xml_path}")
        except Exception as e:
            print(f"  ⚠ Could not load services from XML: {e}")
            self.services = []
            self.service_manager = None
            return self.services
        
        # Derive initial allocation from XML
        self._derive_initial_allocation_from_xml(xml_path)

        # Overlay current allocation from services_update.xml (flexible/layer_lock only)
        self._override_allocation_from_update_xml(xml_path)
        
        return self.services
    
    def get_services_xml_path(self) -> str:
        """
        Return absolute path to services.xml based on config.
        Validates that file exists.
        Always returns the configured source file, not *_update.xml
        """
        resolved = self.get_services_xml_path_without_validation()
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Services XML file not found: {resolved}")
        return resolved
    
    def get_services_xml_path_without_validation(self) -> str:
        """Return absolute path to services.xml without checking if it exists."""
        service_config = self.config.get('services', {})
        xml_path = service_config.get('xml_path', 'src/config/services.xml')
        
        if os.path.isabs(xml_path):
            return xml_path
        else:
            project_root = self._get_project_root()
            return os.path.join(project_root, xml_path)
    
    def _derive_initial_allocation_from_xml(self, xml_path: str):
        """Derive initial service allocation from XML placements."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            placements_found = 0
            
            for svc in root.findall('.//service'):
                sid = svc.get('id') or svc.get('service_id')
                if not sid:
                    continue
                
                placement = svc.find('placement')
                if placement is not None:
                    node = placement.get('node')
                    ptype = placement.get('type')
                    
                    # Track locks for orchestration constraints
                    if ptype in ('locked', 'node_lock', 'layer_lock'):
                        self.locked_services.add(sid)
                    
                    if node:
                        self.current_allocation[sid] = node
                        placements_found += 1
                
                # Check <instances> section
                if sid not in self.current_allocation:
                    instances = svc.find('instances')
                    if instances is not None:
                        # Check for multi-layer service with layer-rules
                        layer_rules = placement.find('layer-rules') if placement is not None else None
                        locked_layer_id = None
                        daemonset_layer_ids = set()
                        
                        if layer_rules is not None:
                            for layer_rule in layer_rules.findall('layer'):
                                layer_id = layer_rule.get('id')
                                rule_type = layer_rule.get('type', 'flexible')
                                if rule_type == 'layer-lock' and layer_id:
                                    locked_layer_id = layer_id
                                elif rule_type == 'daemonset' and layer_id:
                                    daemonset_layer_ids.add(layer_id)
                        
                        # Find instance in locked layer first
                        selected_container = None
                        if locked_layer_id:
                            for container in instances.findall('container[@node]'):
                                if container.get('layer') == locked_layer_id:
                                    selected_container = container
                                    break
                        
                        # Fallback: find first non-daemonset instance
                        if selected_container is None:
                            for container in instances.findall('container[@node]'):
                                container_layer = container.get('layer', '')
                                if container_layer not in daemonset_layer_ids:
                                    selected_container = container
                                    break
                        
                        # Last resort: use first instance with node attribute
                        if selected_container is None:
                            selected_container = instances.find('container[@node]')
                        
                        if selected_container is not None:
                            node = selected_container.get('node')
                            if node:
                                self.current_allocation[sid] = node
                                placements_found += 1
                
                # Pick up podRef if present
                pod_ref = svc.get('podRef')
                if pod_ref:
                    self.current_pod_refs[sid] = pod_ref
            
            if placements_found:
                print(f"  ✓ Initial allocation derived from XML placements ({placements_found} services)")
        except Exception:
            pass

    def _override_allocation_from_update_xml(self, xml_path: str):
        """Override allocation using placement node from services_update.xml for flexible/layer_lock."""
        try:
            update_path = xml_path.replace('.xml', '_update.xml')
            if not os.path.exists(update_path):
                return

            tree = ET.parse(update_path)
            root = tree.getroot()

            for svc in root.findall('.//service'):
                sid = svc.get('id') or svc.get('service_id')
                if not sid:
                    continue
                placement = svc.find('placement')
                if placement is None:
                    continue

                ptype = placement.get('type', '').lower()
                if ptype not in ('flexible', 'layer_lock', 'layer-lock'):
                    continue

                node = placement.get('node')
                if node:
                    self.current_allocation[sid] = node
        except Exception:
            pass
    
    def discover_current_allocation(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Discover current service allocation from Kubernetes.
        
        Returns a mapping: service_id -> [{'node': node_name, 'pod': pod_name, 'layer': layer_id}, ...]
        
        For simulator mode, returns an empty dict (allocation is managed internally).
        For Kubernetes mode, queries the cluster to find pod placements.
        
        Returns:
            Dict mapping service_id to list of pod placements
        """
        env_mode = self.config.get('environment', {}).get('mode', 'kubernetes')
        
        # Simulator mode: no external allocation discovery needed
        if env_mode == 'simulator':
            return {}
        
        # Kubernetes mode: try to discover from cluster
        try:
            from ..Executors.kubernetes_service_discovery import KubernetesServiceDiscovery
            
            # Build cluster config from environment config
            # Clusters are under environment.kubernetes.clusters, not environment.clusters
            clusters_config = self.config.get('environment', {}).get('kubernetes', {}).get('clusters', [])
            services_config = self.config.get('services', {})
            
            self._logger.info(f"[ServiceDiscoveryManager] Loading {len(clusters_config)} clusters from config")
            
            # Create service discoverer
            k8s_discovery = KubernetesServiceDiscovery(clusters_config, services_config)
            
            # Get set of known service IDs
            service_ids = {svc.service_id for svc in self.services if hasattr(svc, 'service_id')}
            self._logger.info(f"[ServiceDiscoveryManager] Known service IDs: {service_ids}")
            
            # Discover allocation
            allocation = k8s_discovery.discover_current_allocation(service_ids)
            self._logger.info(f"[ServiceDiscoveryManager] Discovery result: {len(allocation)} services found")
            return allocation
            
        except ImportError:
            self._logger.warning("KubernetesServiceDiscovery not available")
            return {}
        except Exception as e:
            self._logger.warning(f"Could not discover allocation from Kubernetes: {e}")
            import traceback
            self._logger.debug(traceback.format_exc())
            return {}
    
    def _get_project_root(self) -> str:
        """Get the project root directory."""
        # This file is at: maestro/src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/Managers/service_discovery_manager.py
        # Project root is: maestro/
        # Need to go up 6 levels: Managers -> BlazeMaestro -> BlazeAIoT -> EdgeLogiQ -> src -> maestro
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
