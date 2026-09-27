"""Creates new service elements with proper structure."""
import xml.etree.ElementTree as ET
from typing import Dict

from .metrics_applier import MetricsApplier


class ServiceCreator:
    """Creates new service elements with proper structure."""
    
    def __init__(self, base_updater):
        self.base_updater = base_updater
        self.metrics_applier = MetricsApplier()
    
    def create_service(self, root: ET.Element, workload_name: str, metrics: Dict) -> bool:
        """Create DaemonSet or single service based on metrics."""
        by_node = metrics.get('by_node', {})
        workload_type = metrics.get('workload_type', 'Deployment')
        pods = metrics.get('pods', []) or []
        
        if not pods:
            return False  # Skip empty services
        
        if workload_type == 'DaemonSet' and len(by_node) > 1:
            return self._create_daemonset(root, workload_name, metrics, by_node)
        else:
            return self._create_single(root, workload_name, metrics, by_node)
    
    def _create_daemonset(self, root, name, metrics, by_node):
        """Create DaemonSet service with per-node containers."""
        svc = ET.Element('service')
        svc.set('id', name)
        placement = ET.SubElement(svc, 'placement')
        placement.set('type', 'daemonset')
        
        containers = ET.SubElement(svc, 'containers')
        for node, data in sorted(by_node.items()):
            if node == 'unknown':
                continue
            cont = ET.SubElement(containers, 'container')
            cont.set('id', f"{name}-{node}")
            cont.set('cpu', f"{data.get('cpu_usage', 0.5) * 1.2:.2f}")
            cont.set('ram_gb', f"{data.get('ram_usage_gb', 1.0) * 1.2:.2f}")
            cont.set('node', node)
            if 'exec_time' in data:
                cont.set('exec_time', f"{data['exec_time']:.4f}")
            elif 'exec_time' in metrics:
                cont.set('exec_time', f"{metrics['exec_time']:.4f}")
        
        self._add_instances(svc, name, metrics)
        root.append(svc)
        return True
    
    def _create_single(self, root, name, metrics, by_node):
        """Create single service (Deployment/StatefulSet)."""
        svc = ET.Element('service')
        svc.set('id', name)
        placement = ET.SubElement(svc, 'placement')
        
        if len(by_node) == 1:
            node = list(by_node.keys())[0]
            if node != 'unknown':
                placement.set('type', 'node_lock')
                placement.set('node', node)
            else:
                placement.set('type', 'flexible')
        else:
            placement.set('type', 'flexible')
        
        containers = ET.SubElement(svc, 'containers')
        if len(by_node) > 1:
            for node, data in sorted(by_node.items()):
                if node == 'unknown':
                    continue
                cont = ET.SubElement(containers, 'container')
                cont.set('id', f"{name}-{node}")
                cont.set('cpu', f"{(data.get('cpu_usage', 0.5) or 0.5) * 1.2:.2f}")
                cont.set('ram_gb', f"{(data.get('ram_usage_gb', 1.0) or 1.0) * 1.2:.2f}")
                cont.set('node', node)
        else:
            cont = ET.SubElement(containers, 'container')
            cont.set('id', f"{name}-container")
            cont.set('cpu', f"{metrics.get('cpu_usage', 0.5) * 1.2:.2f}")
            cont.set('ram_gb', f"{metrics.get('ram_usage_gb', 1.0) * 1.2:.2f}")
        
        self._add_instances(svc, name, metrics)
        root.append(svc)
        return True
    
    def _add_instances(self, service, name, metrics):
        """Add instances section with running pods."""
        pods = metrics.get('pods', []) or []
        by_node = metrics.get('by_node', {})
        instances = ET.SubElement(service, 'instances')
        
        seen = set()
        for pod_info in pods:
            pod_name = pod_info.get('name', '')
            if pod_name in seen:
                continue
            seen.add(pod_name)
            
            cont = ET.SubElement(instances, 'container')
            cont.set('pod_name', pod_name)
            cont.set('id', pod_name)
            cont.set('pods', '1')
            
            node_name = pod_info.get('node', 'unknown')
            if node_name != 'unknown':
                cont.set('node', node_name)
            
            if node_name in by_node:
                node_metrics = by_node[node_name]
                for attr, key in [
                    ('cpu_usage_min', 'cpu_usage'),
                    ('ram_usage_min_gb', 'ram_usage_gb')
                ]:
                    if key in node_metrics:
                        val = node_metrics[key]
                        cont.set(attr, f"{val:.3f}")
                        cont.set(attr.replace('_min', '_max'), f"{val:.3f}")
                        cont.set(attr.replace('_min', '_last'), f"{val:.3f}")
