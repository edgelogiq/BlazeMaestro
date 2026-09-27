"""Seeds and manages instance elements in services.xml."""
import xml.etree.ElementTree as ET
from typing import Dict

from BlazeAIoT.Core.Utils import BlazeLogger


class InstanceSeeder:
    """Handles seeding and managing instances from discovery."""
    
    def __init__(self, base_updater):
        self.base_updater = base_updater
    
    def seed_instances_from_discovery(self, root: ET.Element, service_metrics: Dict, namespace: str):
        """Seed instances from discovery to ensure one per node."""
        logger = BlazeLogger.get_instance()
        
        host_layer_map = self._load_host_layer_mapping(root)
        
        # Seed from containers that don't have instances
        self._seed_from_containers(root, host_layer_map, logger)
        
        # Seed from discovered pods
        self._seed_from_pods(root, service_metrics, logger)
    
    def _seed_from_containers(self, root: ET.Element, host_layer_map: Dict[str, str], logger) -> int:
        """Seed instances from container definitions."""
        total_created = 0
        
        for service in root.findall('service'):
            instances = service.find('instances')
            if instances is None or len(list(instances)) == 0:
                containers = service.find('containers')
                if containers is None:
                    continue
                
                created = 0
                instances = ET.SubElement(service, 'instances') if instances is None else instances
                # Track existing instances by pod_name (primary) or node (fallback for newly seeded instances)
                existing_pods = {c.get('pod_name') for c in instances.findall('container') if c.get('pod_name')}
                existing_nodes = {c.get('node') for c in instances.findall('container')}
                
                for c in containers.findall('container'):
                    node = c.get('node')
                    if not node:
                        continue
                    cid = c.get('id')
                    # Only skip if we have a pod_name that already exists
                    if node in existing_nodes:
                        continue
                    
                    inst = ET.SubElement(instances, 'container')
                    # Don't set id on instances - use pod_name or node+container for identification
                    inst.set('node', node)
                    inferred_layer = self.base_updater._infer_layer_prefix(node)
                    if inferred_layer:
                        inst.set('layer', inferred_layer)
                    # Seed metrics
                    inst.set('cpu_usage_min', f"{0.000:.3f}")
                    inst.set('cpu_usage_max', f"{0.000:.3f}")
                    inst.set('cpu_usage_last', f"{0.000:.3f}")
                    inst.set('ram_usage_min_gb', f"{0.000:.3f}")
                    inst.set('ram_usage_max_gb', f"{0.000:.3f}")
                    inst.set('ram_usage_last_gb', f"{0.000:.3f}")
                    created += 1
                
                if created:
                    logger.debug(f"Seeded {created} instances for service '{service.get('id')}'")
                    total_created += created
        
        return total_created
    
    def _seed_from_pods(self, root: ET.Element, service_metrics: Dict, logger):
        """
        Seed instances from discovered pods.
        
        CRITICAL: Preserve existing instances as baseline.
        Only ADD instances for new pods discovered on different nodes.
        Never DELETE existing instances.
        """
        for service_name, svc_data in service_metrics.items():
            pods = svc_data.get('pods', []) or []
            by_node = svc_data.get('by_node', {}) or {}
            service_elem = next((s for s in root.findall('service') if s.get('id') == service_name), None)
            if service_elem is None:
                continue
            
            insts = service_elem.find('instances') or ET.SubElement(service_elem, 'instances')
            placement_node = None
            placement = service_elem.find('placement')
            if placement is not None:
                placement_node = placement.get('node')
            
            # If no pods discovered, but we have node-level metrics (e.g., exec metrics),
            # create synthetic instances to capture placement changes.
            if not pods:
                if by_node:
                    created = 0
                    existing_by_node = {c.get('node'): c for c in insts.findall('container')}
                    for node_name in by_node.keys():
                        if not node_name:
                            continue
                        if node_name in existing_by_node:
                            continue
                        inst = ET.SubElement(insts, 'container')
                        synthetic_name = f"{service_name}-synthetic-{node_name}"
                        inst.set('pod_name', synthetic_name)
                        inst.set('id', synthetic_name)
                        inst.set('node', node_name)
                        inferred_layer = self.base_updater._infer_layer_prefix(node_name)
                        if inferred_layer:
                            inst.set('layer', inferred_layer)
                        # Seed metrics
                        inst.set('cpu_usage_min', f"{0.000:.3f}")
                        inst.set('cpu_usage_max', f"{0.000:.3f}")
                        inst.set('cpu_usage_last', f"{0.000:.3f}")
                        inst.set('ram_usage_min_gb', f"{0.000:.3f}")
                        inst.set('ram_usage_max_gb', f"{0.000:.3f}")
                        inst.set('ram_usage_last_gb', f"{0.000:.3f}")
                        created += 1
                continue
            
            # Build set of active pods from discovery
            active_pods = {pod.get('name') for pod in pods if pod.get('name')}
            
            # Group existing containers by (pod_name, node) to find duplicates
            from collections import defaultdict
            containers_by_key = defaultdict(list)
            
            for c in insts.findall('container'):
                node_name = self.base_updater._normalize_node_id(c.get('node')) if c.get('node') else None
                pod_name = c.get('pod_name')
                if node_name and pod_name:
                    key = (pod_name, node_name)
                    containers_by_key[key].append(c)
            
            # Clean up: remove terminated pods but preserve their historical min/max metrics
            existing_pairs = {}  # (pod_name, node) → container_element
            containers_to_remove = []
            
            # First pass: identify active pods and collect historical metrics from terminated ones
            active_containers = {}  # (service, node) → active container element
            terminated_with_history = {}  # (service, node) → list of terminated containers WITH meaningful metrics
            terminated_without_history = []  # list of terminated containers with all zeros
            
            for key, container_list in containers_by_key.items():
                pod_name, node_name = key
                is_active = pod_name in active_pods
                
                if is_active:
                    # Found an active pod
                    if len(container_list) == 1:
                        existing_pairs[key] = container_list[0]
                        active_containers[node_name] = container_list[0]
                    else:
                        # Multiple entries - keep first, remove rest
                        existing_pairs[key] = container_list[0]
                        active_containers[node_name] = container_list[0]
                        for c in container_list[1:]:
                            containers_to_remove.append(c)
                else:
                    # Terminated pod - check if it has meaningful historical data
                    for c in container_list:
                        has_history = (
                            float(c.get('cpu_usage_max', 0) or 0) > 0 or
                            float(c.get('ram_usage_max_gb', 0) or 0) > 0
                        )
                        if has_history:
                            # Keep this terminated pod - it has historical data
                            if node_name not in terminated_with_history:
                                terminated_with_history[node_name] = []
                            terminated_with_history[node_name].append(c)
                        else:
                            # No meaningful data - can be removed
                            terminated_without_history.append(c)
                            containers_to_remove.append(c)
            
            # Second pass: merge historical metrics from terminated pods on SAME NODE into active ones
            for node_name, active_container in active_containers.items():
                if node_name in terminated_with_history:
                    # Merge historical min/max from terminated pods ON THIS NODE
                    for terminated in terminated_with_history[node_name]:
                        self._merge_historical_metrics(active_container, terminated, logger, service_name, node_name)
                        # After merging, we can remove the terminated pod
                        if terminated not in containers_to_remove:
                            containers_to_remove.append(terminated)
            
            # Remove duplicate containers from XML
            for c in containers_to_remove:
                insts.remove(c)
            
            created = 0
            updated = 0
            preserved = 0
            removed = len(containers_to_remove)
            
            for pod in pods:
                pod_name = pod.get('name')
                node_raw = pod.get('node')
                
                # If node is an IP and we have a placement node, prefer placement to avoid IPs in XML
                if node_raw and '.' in node_raw and placement_node:
                    node_raw = placement_node
                
                node = self.base_updater._normalize_node_id(node_raw) if node_raw else None
                if not pod_name or not node:
                    continue
                
                key = (pod_name, node)
                
                # Check if this exact (pod_name, node) pair already exists
                if key in existing_pairs:
                    # PRESERVE: Keep existing instance with its current metrics
                    existing_container = existing_pairs[key]
                    # Only update layer if needed
                    layer = self.base_updater._infer_layer_prefix(node_raw or node)
                    if layer and existing_container.get('layer') != layer:
                        existing_container.set('layer', layer)
                        updated += 1
                    else:
                        preserved += 1
                    continue
                
                # NEW: Create instance for this pod on this node (NOT SEEN BEFORE)
                inst = ET.SubElement(insts, 'container')
                inst.set('pod_name', pod_name)
                inst.set('id', pod_name)
                inst.set('node', node)
                inferred_layer = self.base_updater._infer_layer_prefix(node)
                if inferred_layer:
                    inst.set('layer', inferred_layer)
                # Initialize with zero metrics (will be populated from Prometheus later)
                inst.set('cpu_usage_min', f"{0.000:.3f}")
                inst.set('cpu_usage_max', f"{0.000:.3f}")
                inst.set('cpu_usage_last', f"{0.000:.3f}")
                inst.set('ram_usage_min_gb', f"{0.000:.3f}")
                inst.set('ram_usage_max_gb', f"{0.000:.3f}")
                inst.set('ram_usage_last_gb', f"{0.000:.3f}")
                created += 1
            
    def _load_host_layer_mapping(self, root: ET.Element) -> Dict[str, str]:
        """Load host->layer mapping from platform XML."""
        mapping = {}
        # Try to find platform.xml in same directory
        # (Would need xml_path to do this properly - for now return empty dict)
        return mapping    
    def _merge_historical_metrics(self, active_container, terminated_container, logger, service_name, node_name):
        """
        Merge historical min/max metrics from terminated pod into active pod.
        Preserves the lowest min and highest max values.
        """
        # Merge CPU metrics
        for metric_type in ['cpu_usage', 'ram_usage_gb']:
            # Min values - take the lowest
            active_min = float(active_container.get(f'{metric_type}_min', 0) or 0)
            term_min = float(terminated_container.get(f'{metric_type}_min', 0) or 0)
            if term_min > 0 and (active_min == 0 or term_min < active_min):
                active_container.set(f'{metric_type}_min', f"{term_min:.3f}")
            
            # Max values - take the highest
            active_max = float(active_container.get(f'{metric_type}_max', 0) or 0)
            term_max = float(terminated_container.get(f'{metric_type}_max', 0) or 0)
            if term_max > active_max:
                active_container.set(f'{metric_type}_max', f"{term_max:.3f}")
        
        # Merge exec_time and execution_rate_hz if present
        for metric in ['exec_time', 'execution_rate_hz']:
            term_attr = terminated_container.get(metric)
            if term_attr:
                try:
                    term_val = float(term_attr)
                    active_val = float(active_container.get(metric, 0) or 0)
                    # For exec metrics, preserve if active doesn't have them
                    if term_val > 0 and active_val == 0:
                        active_container.set(metric, f"{term_val:.6f}" if metric == 'exec_time' else f"{term_val:.2f}")
                except (ValueError, TypeError):
                    pass