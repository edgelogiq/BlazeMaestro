"""Configuration update orchestration and utility helpers."""
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Set

from BlazeAIoT.Core.Utils import BlazeLogger


class ConfigUpdateHelper:
    """Helper for configuration update operations."""
    
    def __init__(self, base_updater):
        self.base_updater = base_updater
    
    def get_allowed_layers(self, node_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> set:
        """Return a lowercase set of configured layers with fallback to inferred layers."""
        allowed = set()

        # Add layers from successfully accessed cluster_urls
        if getattr(self.base_updater, 'cluster_urls', None):
            allowed.update(k.lower() for k in self.base_updater.cluster_urls.keys() if k)

        if self.base_updater.metrics_facade and getattr(self.base_updater.metrics_facade, 'cluster_urls', None):
            allowed.update(k.lower() for k in self.base_updater.metrics_facade.cluster_urls.keys() if k)

        # Also include configured cluster layers even if Prometheus access failed
        if self.base_updater.metrics_facade and getattr(self.base_updater.metrics_facade, 'cluster_configs', None):
            for cluster_config in self.base_updater.metrics_facade.cluster_configs:
                layer_id = cluster_config.get('layer_id')
                if layer_id:
                    allowed.add(layer_id.lower())

        # Fallback: infer from observed node metrics if we have none configured
        if not allowed and node_metrics:
            for node_name in node_metrics.keys():
                prefix = self.base_updater._infer_layer_prefix(node_name)
                if prefix and prefix in self.base_updater.DEFAULT_LAYER_PREFIXES:
                    allowed.add(prefix)

        return allowed
    
    def is_services_xml_empty(self, services_xml: str) -> bool:
        """Return True if services.xml is empty or contains no <service> entries."""
        try:
            if not os.path.exists(services_xml) or os.path.getsize(services_xml) == 0:
                return True
            tree = ET.parse(services_xml)
            root = tree.getroot()
            return len(root.findall('service')) == 0
        except Exception:
            return True
    
    def get_prometheus_urls(self) -> list:
        """Build list of (layer_id, prom_url) tuples to query."""
        prometheus_urls = []
        if self.base_updater.cluster_urls:
            for layer_id, prom_url in self.base_updater.cluster_urls.items():
                prometheus_urls.append((layer_id, prom_url))
        else:
            prometheus_urls.append(('default', getattr(self.base_updater, 'prometheus_url', None)))
        return prometheus_urls
    
    def query_node_usage_metrics(self) -> tuple:
        """Query Prometheus for current node CPU and RAM usage. Returns (cpu_dict, ram_dict)."""
        usage_cpu_by_node = {}
        usage_ram_by_node = {}
        
        try:
            for _, prom_url in self.get_prometheus_urls():
                if not prom_url:
                    continue
                
                cpu_query = 'sum(rate(container_cpu_usage_seconds_total{container!=""}[5m])) by (node)'
                ram_query = 'sum(container_memory_working_set_bytes{container!=""}) by (node)'
                
                cpu_results = self.base_updater.query_prometheus_at_url(cpu_query, prom_url)
                ram_results = self.base_updater.query_prometheus_at_url(ram_query, prom_url)

                if cpu_results:
                    for r in cpu_results:
                        node = r.get('metric', {}).get('node')
                        if node:
                            try:
                                usage_cpu_by_node[node] = float(r['value'][1])
                            except Exception:
                                pass

                if ram_results:
                    for r in ram_results:
                        node = r.get('metric', {}).get('node')
                        if node:
                            try:
                                usage_ram_by_node[node] = float(r['value'][1]) / (1024 ** 3)
                            except Exception:
                                pass
        except Exception:
            pass
        
        return usage_cpu_by_node, usage_ram_by_node
    
    def merge_usage_into_metrics(self, node_metrics: Dict, usage_cpu: Dict, usage_ram: Dict) -> Dict:
        """Merge current usage metrics into node_metrics dict."""
        for node, cpu_used in usage_cpu.items():
            if node not in node_metrics:
                node_metrics[node] = {}
            node_metrics[node]['cpu_used'] = max(0.0, cpu_used)

        for node, ram_used in usage_ram.items():
            if node not in node_metrics:
                node_metrics[node] = {}
            node_metrics[node]['ram_used_gb'] = max(0.0, ram_used)
        
        return node_metrics
