"""Services XML updater - handles services.xml updates with Prometheus metrics."""
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Optional, Set, Tuple, cast

from BlazeAIoT.Core.Utils import BlazeLogger
from .Tools import ServiceCreator, InstanceManager


class ServicesXMLUpdater:
    """Handles updating services.xml with service metrics from Prometheus."""

    def __init__(self, base_updater):
        """Initialize with reference to base updater."""
        self.base_updater = base_updater
        self._xml_tree = None
        self._xml_root = None
        self._xml_path = None
        self._original_xml_path = None
        self._intentionally_removed_instances = set()
        self.instance_manager = InstanceManager(base_updater)
        self.service_creator = ServiceCreator(base_updater)

    def update(self, xml_path: str, service_metrics: Dict[str, Dict], namespace: str = 'default', discovery_snapshot: Optional[Dict] = None) -> bool:
        """Update services.xml with real service metrics."""
        try:
            logger = BlazeLogger.get_instance()
            
            # Load or create XML
            # INITIALIZATION: Read from base services.xml (source of truth with historical data)
            # Then keep in memory and update throughout lifecycle
            if self._xml_tree is None:
                base_path = xml_path.replace('_update.xml', '.xml')
                
                if os.path.exists(base_path) and os.path.getsize(base_path) > 0:
                    tree = ET.parse(base_path)
                    root = tree.getroot()
                else:
                    root = ET.Element('services')
                    tree = ET.ElementTree(root)
                
                self._xml_tree = tree
                self._xml_root = root
                self._xml_path = base_path
                base, ext = os.path.splitext(xml_path)
                self._original_xml_path = base[:-7] + ext if base.endswith('_update') else xml_path
            else:
                tree = self._xml_tree
                root = self._xml_root
            
            root = cast(ET.Element, root)
            
            import copy
            original_root = copy.deepcopy(root)
            self._intentionally_removed_instances.clear()
            
            # Update existing services
            pod_node_metrics = self._build_pod_node_metrics_lookup(service_metrics)
            updated_count, matched_metrics = self.instance_manager.update_existing_services(
                root, service_metrics, pod_node_metrics)
            has_changes = updated_count > 0
            
            # Remove duplicates
            self._remove_duplicate_services(root)
            
            # Add new services
            new_count = self._add_new_services(root, service_metrics, matched_metrics)
            if new_count > 0:
                has_changes = True
            
            # Verify preservation
            self._verify_instance_preservation(original_root, root, self._intentionally_removed_instances)

            # Normalize instances blocks to prevent duplicates
            self._normalize_instances_elements(root)
            
            # Write update file
            update_path = self._compute_update_path(self._original_xml_path)
            self._write_update_file(tree, root, update_path, updated_count, new_count, has_changes)
            
            return True
        except Exception as e:
            BlazeLogger.get_instance().error(f"Failed to update services.xml: {e}")
            import traceback
            BlazeLogger.get_instance().error(traceback.format_exc())
            return False

    def _compute_update_path(self, xml_path: str) -> str:
        """Return the update-path for a given XML."""
        base, ext = os.path.splitext(xml_path)
        return f"{base}_update{ext}" if ext else f"{xml_path}_update"

    def _build_pod_node_metrics_lookup(self, service_metrics: Dict[str, Dict]) -> Dict:
        """Build a (pod_name, node_name) → metrics lookup."""
        logger = BlazeLogger.get_instance()
        lookup = {}
        
        for svc_name, svc_data in service_metrics.items():
            pods = svc_data.get('pods', []) or []
            by_node = svc_data.get('by_node', {}) or {}
            workload_exec_time = svc_data.get('exec_time')
            workload_exec_rate = svc_data.get('execution_rate_hz')
            
            for pod in pods:
                pod_name = pod.get('name')
                node_name = pod.get('node')
                if pod_name and node_name:
                    node_metrics = by_node.get(node_name, {})
                    pod_exec_time = pod.get('exec_time')
                    pod_exec_rate = pod.get('execution_rate_hz')
                    
                    final_exec_time = pod_exec_time if pod_exec_time is not None else (
                        node_metrics.get('exec_time') if node_metrics.get('exec_time') is not None else workload_exec_time
                    )
                    final_exec_rate = pod_exec_rate if pod_exec_rate is not None else (
                        node_metrics.get('execution_rate_hz') if node_metrics.get('execution_rate_hz') is not None else workload_exec_rate
                    )
                    
                    lookup[(pod_name, node_name)] = {
                        'cpu_usage': float(node_metrics.get('cpu_usage', 0.0)),
                        'ram_usage_gb': float(node_metrics.get('ram_usage_gb', 0.0)),
                        'exec_time': final_exec_time,
                        'execution_rate_hz': final_exec_rate,
                        'workload_level_exec_time': workload_exec_time,
                        'workload_level_exec_rate': workload_exec_rate
                    }
        
        return lookup

    def _remove_duplicate_services(self, root: ET.Element):
        """Remove duplicate service elements."""
        existing = set()
        to_remove = []
        
        for service in root.findall('service'):
            sid = service.get('id')
            if sid in existing:
                to_remove.append(service)
                BlazeLogger.get_instance().warning(f"Removing duplicate service: {sid}")
            else:
                existing.add(sid)
        
        for service in to_remove:
            root.remove(service)

    def _add_new_services(self, root: ET.Element, service_metrics: Dict[str, Dict], matched_metrics: set) -> int:
        """Add new services for unmatched workloads."""
        logger = BlazeLogger.get_instance()
        existing = {s.get('id') for s in root.findall('service')}
        count = 0
        
        for workload_name, metrics in service_metrics.items():
            if workload_name not in matched_metrics and workload_name not in existing:
                pods = metrics.get('pods', []) or []
                if not pods:
                    continue
                
                if '-daemonset' in workload_name.lower():
                    base = workload_name.lower().split('-daemonset')[0]
                    if base in existing:
                        continue
                
                logger.info(f"[ADD_NEW_SERVICES] Creating service: {workload_name}")
                if self.service_creator.create_service(root, workload_name, metrics):
                    count += 1
        
        return count

    def _verify_instance_preservation(self, original_root: ET.Element, updated_root: ET.Element, 
                                      intentionally_removed: Optional[Set[Tuple[str, str, str]]] = None):
        """Verify NO instances were removed during update."""
        logger = BlazeLogger.get_instance()
        intentionally_removed = intentionally_removed or set()
        
        if intentionally_removed:
            logger.info(f"[VERIFY_INSTANCES] Will not restore {len(intentionally_removed)} intentionally removed")
        
        # Build original instances map
        original_instances = {}
        for service in original_root.findall('service'):
            sid = service.get('id')
            instances = service.find('instances')
            if instances is not None:
                # Use pod_name as primary identifier (now that id is removed from instances)
                original_instances[sid] = {i.get('pod_name') 
                                          for i in instances.findall('container') 
                                          if i.get('pod_name')}
        
        # Check and restore if needed
        restored = 0
        for service in updated_root.findall('service'):
            sid = service.get('id')
            if sid not in original_instances:
                continue
            
            instances = service.find('instances')
            if instances is None:
                logger.warning(f"🚨 Service '{sid}' lost instances element - RESTORING")
                instances = ET.SubElement(service, 'instances')
                orig_svc = original_root.find(f".//service[@id='{sid}']")
                if orig_svc is not None:
                    for orig_inst in orig_svc.find('instances').findall('container'):
                        new_inst = ET.SubElement(instances, 'container')
                        for attr, val in orig_inst.attrib.items():
                            new_inst.set(attr, val)
                        restored += 1
            else:
                # Use pod_name as primary identifier
                updated_set = {i.get('pod_name') 
                              for i in instances.findall('container') if i.get('pod_name')}
                missing = original_instances[sid] - updated_set
                
                # Filter out intentionally removed instances
                filtered = {p for p in missing 
                           if (sid, p) not in intentionally_removed}
                
                if filtered:
                    logger.warning(f"🚨 Service '{sid}': restoring {len(filtered)} missing instances")
                    orig_svc = original_root.find(f".//service[@id='{sid}']")
                    for orig_inst in orig_svc.find('instances').findall('container'):
                        if orig_inst.get('pod_name') in filtered:
                            new_inst = ET.SubElement(instances, 'container')
                            for attr, val in orig_inst.attrib.items():
                                new_inst.set(attr, val)
                            restored += 1

    def _write_update_file(self, tree, root, update_path: str, updated: int, new: int, has_changes: bool):
        """Write the update file."""
        comment = ET.Comment(f" Updated from Prometheus on {datetime.now().isoformat()} ")
        root.insert(0, comment)
        self.base_updater._indent_xml(root)
        tree.write(update_path, encoding='utf-8', xml_declaration=True)

    def _normalize_instances_elements(self, root: ET.Element) -> None:
        """Ensure each service has exactly one <instances> element."""
        for service in root.findall('service'):
            instances_blocks = service.findall('instances')
            if len(instances_blocks) <= 1:
                continue

            primary = instances_blocks[0]
            for extra in instances_blocks[1:]:
                for inst in list(extra.findall('container')):
                    primary.append(inst)
                service.remove(extra)
