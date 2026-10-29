"""
Node IP to Name Resolution Module.

Handles resolving node IP addresses to node names using:
1. Prometheus kubelet_node_name metrics
2. Kubernetes API (from multiple clusters via kubeconfig files)

Caches the mapping to avoid rebuilding multiple times per update cycle.
"""

import xml.etree.ElementTree as ET
from typing import Dict
from BlazeAIoT.Core.Utils import BlazeLogger


class NodeResolver:
    """Resolves node IP addresses to node names."""
    
    def __init__(self, prometheus_urls=None, cluster_configs=None, verify_ssl=True):
        """
        Initialize the node resolver.
        
        Args:
            prometheus_urls: List of Prometheus URLs to query
            cluster_configs: List of cluster configurations with kubeconfig paths
            verify_ssl: Whether to verify SSL certificates
        """
        self.prometheus_urls = prometheus_urls or []
        self.cluster_configs = cluster_configs or []
        self.verify_ssl = verify_ssl
        self._cache = None
        self.logger = BlazeLogger.get_instance()
    
    def get_mapping(self, root: ET.Element = None) -> Dict[str, str]:
        """
        Get IP-to-name mapping. Returns cached mapping if available.
        
        Args:
            root: XML root element (used to seed initial mapping from existing nodes)
            
        Returns:
            Dictionary mapping IP addresses (and names) to node names
        """
        if self._cache is not None:
            return self._cache
        
        mapping = {}
        
        # Step 1: Seed from XML (node names map to themselves)
        if root is not None:
            for svc in root.findall('service'):
                insts = svc.find('instances')
                if insts is None:
                    continue
                for c in insts.findall('container'):
                    node_name = c.get('node')
                    if node_name:
                        mapping[node_name.lower()] = node_name
        
        # Step 2: Query Prometheus for kubelet_node_name
        self._query_prometheus(mapping)
        
        # Step 3: Query Kubernetes API for node addresses
        self._query_kubernetes_api(mapping)
        
        # Cache the result
        self._cache = mapping
        
        # SAFETY CHECK: Remove any None values before returning
        none_entries = [(k, v) for k, v in mapping.items() if v is None]
        if none_entries:
            # Remove them
            for k in [k for k, v in none_entries]:
                del mapping[k]
        
        
        return mapping
    
    def _query_prometheus(self, mapping: Dict[str, str]):
        """Query Prometheus instances for node IP to name mappings."""
        from .prometheus_query_client import PrometheusQueryClient
        
        query = 'kubelet_node_name'
        
        for prom_url in self.prometheus_urls:
            try:
                client = PrometheusQueryClient(prometheus_url=prom_url, verify_ssl=self.verify_ssl)
                results = client.query(query)
                
                if results:
                    for result in results:
                        node_name = result.get('metric', {}).get('node')
                        instance = result.get('metric', {}).get('instance', '')
                        # Extract IP from instance (format: "172.16.200.25:10250")
                        node_ip = instance.split(':')[0] if ':' in instance else instance
                        
                        if node_name and node_ip:
                            mapping[node_ip] = node_name
                            mapping[node_ip.lower()] = node_name
            except Exception as e:
                self.logger.warning(f"[NodeResolver] Failed to query Prometheus {prom_url}: {e}")
    
    def _query_kubernetes_api(self, mapping: Dict[str, str]):
        """Query Kubernetes API from all configured clusters."""
        try:
            from kubernetes import client, config
            
            # Try in-cluster config (for primary cluster where pod is running)
            try:
                config.load_incluster_config()
                v1 = client.CoreV1Api()
                nodes = v1.list_node()
                for node in nodes.items:
                    node_name = node.metadata.name
                    for addr in node.status.addresses:
                        if addr.type == 'InternalIP':
                            node_ip = addr.address
                            if node_ip and node_name:
                                mapping[node_ip] = node_name
                                mapping[node_ip.lower()] = node_name
            except Exception as e:
                self.logger.debug(f"[NodeResolver] In-cluster K8s not available: {e}")
            
            # Query additional clusters via kubeconfig files
            for cluster_cfg in self.cluster_configs:
                kubeconfig_path = cluster_cfg.get('kubeconfig_path')
                context = cluster_cfg.get('context')
                
                if not kubeconfig_path or not context:
                    continue
                
                try:
                    import os
                    kubeconfig_expanded = os.path.expanduser(kubeconfig_path)
                    
                    config.load_kube_config(config_file=kubeconfig_expanded, context=context)
                    v1 = client.CoreV1Api()
                    nodes = v1.list_node()
                    
                    for node in nodes.items:
                        node_name = node.metadata.name
                        for addr in node.status.addresses:
                            if addr.type == 'InternalIP':
                                node_ip = addr.address
                                if node_ip and node_name:
                                    mapping[node_ip] = node_name
                                    mapping[node_ip.lower()] = node_name
                except Exception as e:
                    self.logger.warning(f"[NodeResolver] Failed to query K8s {context}: {e}")
        
        except ImportError:
            self.logger.warning("[NodeResolver] kubernetes library not available")
        except Exception as e:
            self.logger.warning(f"[NodeResolver] K8s API unavailable: {e}")
    
    def resolve_all_in_xml(self, root: ET.Element) -> int:
        """
        Resolve all node IPs to names in the XML tree.
        
        Args:
            root: XML root element
            
        Returns:
            Number of nodes resolved
        """
        mapping = self.get_mapping(root)
        resolved_count = 0
        
        for service in root.findall('service'):
            instances = service.find('instances')
            if instances is None:
                continue
            
            for container in instances.findall('container'):
                node_raw = container.get('node', '')
                if not node_raw:
                    continue
                
                # Try exact match, then case-insensitive
                resolved = mapping.get(node_raw) or mapping.get(node_raw.lower())
                
                if resolved and resolved != node_raw:
                    container.set('node', resolved)
                    resolved_count += 1
        
        return resolved_count
