"""Manages service instances and pod lifecycle."""
import xml.etree.ElementTree as ET

from BlazeAIoT.Core.Utils import BlazeLogger
from .metrics_applier import MetricsApplier
from .service_matcher import ServiceMatcher


class InstanceManager:
    """Manages service instances and pod lifecycle."""
    
    def __init__(self, base_updater):
        self.base_updater = base_updater
        self.metrics_applier = MetricsApplier()
    
    def update_existing_services(self, root, service_metrics, pod_node_metrics):
        """Update existing services with metrics. Returns (count, matched_set)."""
        matcher = ServiceMatcher()
        updated = 0
        matched = set()
        
        for service in root.findall('service'):
            sid = service.get('id') or ''
            match = matcher.find_match(sid, service_metrics, service)
            
            if match:
                matched.add(match)
                if self._update_service(service, sid, service_metrics[match], pod_node_metrics):
                    updated += 1
        
        return updated, matched
    
    def _update_service(self, service, sid, metrics, pod_node_metrics):
        """Update single service element."""
        logger = BlazeLogger.get_instance()
        changed = False
        
        # Update containers
        for cont in service.findall('.//container'):
            if self.metrics_applier.apply_to_container(cont, metrics):
                changed = True
        
        # Update instances
        pods = metrics.get('pods', []) or []
        by_node = metrics.get('by_node', {}) or {}
        
        if not pods:
            # No pod list (scaled to 0 or discovery gap) but we may have node-level metrics
            if by_node:
                instances = service.find('instances')
                if instances is None:
                    instances = ET.SubElement(service, 'instances')
                    changed = True
                
                existing_by_node = {cont.get('node'): cont for cont in instances.findall('container')}
                
                for node_name, node_metrics in by_node.items():
                    if not node_name:
                        continue
                    existing = existing_by_node.get(node_name)
                    if not existing:
                        inst = ET.SubElement(instances, 'container')
                        synthetic_name = f"{sid}-synthetic-{node_name}"
                        inst.set('pod_name', synthetic_name)
                        inst.set('id', synthetic_name)
                        inst.set('node', node_name)
                        inst.set('layer', self.base_updater._infer_layer_prefix(node_name) or 'default')
                        for k in ['cpu_usage_min', 'cpu_usage_max', 'cpu_usage_last',
                                 'ram_usage_min_gb', 'ram_usage_max_gb', 'ram_usage_last_gb']:
                            inst.set(k, '0.000')
                        existing = inst
                        existing_by_node[node_name] = inst
                        changed = True
                    
                    if self.metrics_applier.apply_to_instance(
                        existing,
                        pod_node_metrics,
                        sid,
                        existing.get('pod_name') or '',
                        node_name,
                        fallback=node_metrics
                    ):
                        changed = True
            return changed
        
        instances = service.find('instances')
        if instances is None:
            instances = ET.SubElement(service, 'instances')
            changed = True
        
        # Enforce one instance per node
        
        # Remove duplicates on same node
        for node, insts in self._group_by_node(instances):
            if len(insts) > 1:
                for inst in insts[1:]:
                    instances.remove(inst)
                    changed = True
        
        # Update/create instances for current pods
        existing_by_node = {cont.get('node'): cont for cont in instances.findall('container')}
        
        for pod in pods:
            pname = pod.get('name')
            pnode = pod.get('node')
            if not pname or not pnode:
                continue
            
            existing = existing_by_node.get(pnode)
            if existing:
                if existing.get('pod_name') != pname:
                    existing.set('pod_name', pname)
                    existing.set('id', pname)
                    changed = True
            else:
                inst = ET.SubElement(instances, 'container')
                inst.set('pod_name', pname)
                inst.set('id', pname)
                inst.set('node', pnode)
                inst.set('layer', self.base_updater._infer_layer_prefix(pnode) or 'default')
                for k in ['cpu_usage_min', 'cpu_usage_max', 'cpu_usage_last',
                         'ram_usage_min_gb', 'ram_usage_max_gb', 'ram_usage_last_gb']:
                    inst.set(k, '0.000')
                changed = True
            
            # Apply metrics
            if self.metrics_applier.apply_to_instance(
                existing or existing_by_node.get(pnode),
                pod_node_metrics,
                sid,
                pname,
                pnode,
                fallback=by_node.get(pnode)
            ):
                changed = True

        # Also create/update instances for nodes that only appear in by_node (e.g., exec metrics)
        for node_name, node_metrics in by_node.items():
            if not node_name:
                continue
            if node_name in existing_by_node:
                # Apply fallback metrics if pod-specific data is missing
                if self.metrics_applier.apply_to_instance(
                    existing_by_node[node_name],
                    pod_node_metrics,
                    sid,
                    existing_by_node[node_name].get('pod_name') or '',
                    node_name,
                    fallback=node_metrics
                ):
                    changed = True
                continue

            # Create synthetic instance for node-level metrics
            inst = ET.SubElement(instances, 'container')
            synthetic_name = f"{sid}-synthetic-{node_name}"
            inst.set('pod_name', synthetic_name)
            inst.set('id', synthetic_name)
            inst.set('node', node_name)
            inst.set('layer', self.base_updater._infer_layer_prefix(node_name) or 'default')
            for k in ['cpu_usage_min', 'cpu_usage_max', 'cpu_usage_last',
                     'ram_usage_min_gb', 'ram_usage_max_gb', 'ram_usage_last_gb']:
                inst.set(k, '0.000')
            existing_by_node[node_name] = inst
            changed = True

            if self.metrics_applier.apply_to_instance(
                inst,
                pod_node_metrics,
                sid,
                synthetic_name,
                node_name,
                fallback=node_metrics
            ):
                changed = True
        
        return changed
    
    def _group_by_node(self, instances):
        """Group instances by node."""
        by_node = {}
        for inst in instances.findall('container'):
            node = inst.get('node')
            if node:
                by_node.setdefault(node, []).append(inst)
        return by_node.items()
