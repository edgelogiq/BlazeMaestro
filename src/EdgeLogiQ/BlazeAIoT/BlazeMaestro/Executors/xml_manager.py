"""
XML Manager Module

Implements Single Responsibility Principle (SRP) for XML operations.
Handles all XML file operations for services and platform configuration.
"""

import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from BlazeAIoT.Core.Utils import BlazeLogger


class XMLManager:
    """
    Manages XML configuration files for services and platform.
    
    Responsibilities:
    - Load and parse XML files
    - Update XML with current allocation and metrics
    - Create XML from Kubernetes discovery
    - Manage XML file versions (_update.xml)
    """
    
    def __init__(self, base_path: str):
        """
        Initialize XML Manager.
        
        Args:
            base_path: Base directory for XML files
        """
        self._logger = BlazeLogger.get_instance()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_services_xml_path(self, filename: str = 'services.xml') -> Path:
        """
        Get path to services XML file.
        
        Args:
            filename: Name of XML file
            
        Returns:
            Path to XML file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        xml_path = self.base_path / filename
        if not xml_path.exists():
            raise FileNotFoundError(f"Services XML not found: {xml_path}")
        return xml_path
    
    def get_services_update_path(self, base_filename: str = 'services.xml') -> Path:
        """Get path to services_update.xml (working copy)."""
        if base_filename.endswith('_update.xml'):
            return self.base_path / base_filename
        return self.base_path / base_filename.replace('.xml', '_update.xml')
    
    def load_xml(self, filepath: Path) -> ET.Element:
        """
        Load and parse XML file.
        
        Args:
            filepath: Path to XML file
            
        Returns:
            Root element of XML tree
        """
        try:
            tree = ET.parse(filepath)
            return tree.getroot()
        except ET.ParseError as e:
            self._logger.error(f"XML parse error in {filepath}: {e}")
            raise
    
    def save_xml(self, root: ET.Element, filepath: Path, pretty_print: bool = True) -> None:
        """
        Save XML to file.
        
        Args:
            root: Root element to save
            filepath: Destination file path
            pretty_print: Whether to pretty-print the XML
        """
        if pretty_print:
            rough = ET.tostring(root, encoding='utf-8')
            reparsed = minidom.parseString(rough)
            pretty_xml = reparsed.toprettyxml(indent='  ', encoding='utf-8').decode('utf-8')
            
            # Clean up excessive blank lines
            lines = pretty_xml.split('\n')
            cleaned_lines = [line for line in lines if line.strip() or not lines]
            pretty_xml = '\n'.join(cleaned_lines)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
        else:
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
    
    def extract_initial_allocation(self, xml_path: Path) -> Dict[str, str]:
        """
        Extract initial service allocation from XML placements.
        
        Args:
            xml_path: Path to services XML file
            
        Returns:
            Dict mapping service_id to node_id
        """
        allocation = {}
        locked_services = set()
        
        try:
            root = self.load_xml(xml_path)
            
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
                        locked_services.add(sid)
                    
                    if node:
                        allocation[sid] = node
                
                # Also check <instances> section
                if sid not in allocation:
                    node = self._extract_node_from_instances(svc, placement)
                    if node:
                        allocation[sid] = node
            
            if allocation:
                self._logger.debug(f"✓ Initial allocation from XML: {len(allocation)} services")
            
        except Exception as e:
            self._logger.warning(f"Could not extract allocation from XML: {e}")
        
        return allocation
    
    def _extract_node_from_instances(
        self,
        svc_element: ET.Element,
        placement: Optional[ET.Element]
    ) -> Optional[str]:
        """Extract node from instances section, preferring locked layer instances."""
        instances = svc_element.find('instances')
        if instances is None:
            return None
        
        # Check if this is a multi-layer service with layer-rules
        locked_layer_id = None
        daemonset_layer_ids = set()
        
        if placement is not None:
            layer_rules = placement.find('layer-rules')
            if layer_rules is not None:
                for layer_rule in layer_rules.findall('layer'):
                    layer_id = layer_rule.get('id')
                    rule_type = layer_rule.get('type', 'flexible')
                    if rule_type == 'layer-lock' and layer_id:
                        locked_layer_id = layer_id
                    elif rule_type == 'daemonset' and layer_id:
                        daemonset_layer_ids.add(layer_id)
        
        # Try to find an instance in the locked layer first
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
            return selected_container.get('node')
        
        return None
    
    def create_update_copy(self, source_path: Path, dest_path: Path) -> None:
        """
        Create an update copy of XML file if it doesn't exist.
        
        Args:
            source_path: Source XML file
            dest_path: Destination path for update copy
        """
        if not dest_path.exists():
            shutil.copy2(source_path, dest_path)
            self._logger.info(f"Created {dest_path.name} from {source_path.name}")
        else:
            self._logger.debug(f"Using existing {dest_path.name}")
    
    def update_with_allocation(
        self,
        xml_path: Path,
        allocation: Dict[str, List[Dict[str, str]]],
        preserve_placement_config: bool = True
    ) -> bool:
        """
        Update services XML with current allocation.
        
        Args:
            xml_path: Path to services XML
            allocation: Dict mapping service_id to list of pod info
            preserve_placement_config: If True, preserve placement config from base XML
            
        Returns:
            True if updated, False otherwise
        """
        if not allocation:
            self._logger.info("No allocation to update XML with")
            return False
        
        try:
            # Handle both string and Path objects
            if isinstance(xml_path, str):
                from pathlib import Path as PathlibPath
                xml_path = PathlibPath(xml_path)
            
            root = self.load_xml(xml_path)
            
            for sid, pod_list in allocation.items():
                if not pod_list:
                    continue
                
                # Find or create service element
                svc_el = self._find_or_create_service(root, sid)
                
                # Handle placement
                if preserve_placement_config:
                    # Only add placement if it doesn't exist
                    if svc_el.find('placement') is None:
                        self._add_placement_from_pods(svc_el, pod_list)
                else:
                    # Update placement based on current state
                    self._update_placement_from_pods(svc_el, pod_list)
            
            # Save updated XML
            update_path = self.get_services_update_path(xml_path.name)
            self.save_xml(root, update_path)
            
            self._logger.info(f"✓ Updated {update_path.name} with allocation ({len(allocation)} services)")
            return True
        
        except Exception as e:
            self._logger.error(f"Failed to update XML with allocation: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
            return False
    
    def _find_or_create_service(self, root: ET.Element, service_id: str) -> ET.Element:
        """Find existing service element or create new one."""
        for svc in root.findall('.//service'):
            sid = svc.get('id') or svc.get('service_id')
            if sid == service_id:
                return svc
        
        # Create new service
        svc = ET.SubElement(root, 'service')
        svc.set('id', service_id)
        svc.set('execution_rate_hz', '1.0')
        return svc
    
    def _add_placement_from_pods(self, svc_el: ET.Element, pod_list: List[Dict[str, str]]) -> None:
        """Add placement element based on pod distribution."""
        placement = ET.SubElement(svc_el, 'placement')
        
        if len(pod_list) > 1:
            # Multiple pods - daemonset or multi-layer
            unique_layers = set(p.get('layer', 'unknown') for p in pod_list)
            unique_layers.discard('unknown')
            
            if len(unique_layers) > 1:
                placement.set('type', 'multi-layer')
                placement.set('layers', ','.join(sorted(unique_layers)))
            else:
                placement.set('type', 'daemonset')
                if unique_layers:
                    placement.set('layer', list(unique_layers)[0])
        else:
            # Single pod
            node = pod_list[0].get('node', '')
            if node:
                placement.set('type', 'node_lock')
                placement.set('node', node)
            else:
                placement.set('type', 'flexible')
    
    def _update_placement_from_pods(self, svc_el: ET.Element, pod_list: List[Dict[str, str]]) -> None:
        """Update existing placement element based on pod distribution."""
        placement = svc_el.find('placement')
        if placement is None:
            placement = ET.SubElement(svc_el, 'placement')
        
        # Clear existing attributes
        placement.clear()
        
        self._add_placement_from_pods(svc_el, pod_list)
    
    def xml_exists_and_valid(self, filename: str) -> bool:
        """
        Check if XML file exists and is valid (non-empty with services).
        
        Args:
            filename: Name of XML file
            
        Returns:
            True if file exists and has services, False otherwise
        """
        try:
            xml_path = self.base_path / filename
            if not xml_path.exists():
                return False
            
            root = self.load_xml(xml_path)
            services = root.findall('.//service')
            return len(services) > 0
        
        except Exception as e:
            self._logger.debug(f"XML validation failed for {filename}: {e}")
            return False
