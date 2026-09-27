"""Queries Prometheus with intelligent fallbacks."""
from typing import Dict, List, Optional, Set, Tuple

from BlazeAIoT.Core.Utils import BlazeLogger


class PrometheusQuerier:
    """Helper class to query Prometheus with intelligent fallbacks."""
    
    def __init__(self, query_client, base_updater):
        self.query_client = query_client
        self.base_updater = base_updater
    
    def query_metrics_for_layer(self, layer_id: str, prom_url: str, namespace: str) -> Tuple[Dict, Dict, Set]:
        """Query CPU and RAM metrics for a layer with fallbacks. Returns (cpu_dict, ram_dict, layers_queried)."""
        logger = BlazeLogger.get_instance()
        
        per_layer_cpu = {}
        per_layer_ram = {}
        layers_queried = set()
        
        prom_url = prom_url.rstrip('/')
        logger.debug(f"Querying Prometheus at {prom_url} for layer '{layer_id}', namespace={namespace}")
        
        # Try CPU queries with fallbacks
        cpu_found = self._query_cpu_metrics(layer_id, prom_url, namespace, per_layer_cpu)
        if cpu_found:
            layers_queried.add(layer_id)
        
        # Try RAM queries with fallbacks
        ram_found = self._query_ram_metrics(layer_id, prom_url, namespace, per_layer_ram)
        
        return per_layer_cpu, per_layer_ram, layers_queried
    
    def _query_cpu_metrics(self, layer_id: str, prom_url: str, namespace: str, result_dict: Dict) -> bool:
        """Query CPU by node with fallbacks. Returns True if metrics found."""
        logger = BlazeLogger.get_instance()
        
        # Primary: CPU by node
        cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!=""}}[5m])) by (node)'
        cpu_results = self.query_client.query_at_url(cpu_query, prom_url)
        
        if cpu_results:
            return self._parse_cpu_results(layer_id, cpu_results, result_dict)
        
        # Fallback 1: CPU by pod
        cpu_query_fallback = f'sum by (pod)(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod!="",container!=""}}[5m]))'
        logger.debug(f"  [{layer_id}] Retrying CPU query with 'pod' label")
        cpu_results = self.query_client.query_at_url(cpu_query_fallback, prom_url)
        
        if cpu_results:
            return self._parse_cpu_pod_results(layer_id, cpu_results, result_dict)
        
        # Fallback 2: Join with kube_pod_info
        return self._query_cpu_with_pod_info(layer_id, prom_url, namespace, result_dict)
    
    def _parse_cpu_results(self, layer_id: str, results: List, result_dict: Dict) -> bool:
        """Parse CPU by node results."""
        logger = BlazeLogger.get_instance()
        
        found = 0
        for r in results:
            try:
                node_raw = r['metric'].get('node', '')
                node = self.base_updater._normalize_node_id(node_raw)
                if node:
                    result_dict[(layer_id, node)] = float(r['value'][1])
                    found += 1
                    logger.debug(f"  [{layer_id}] CPU: {node} = {result_dict[(layer_id, node)]:.3f}")
            except Exception as e:
                logger.debug(f"  Error parsing CPU: {e}")
        return found > 0
    
    def _parse_cpu_pod_results(self, layer_id: str, results: List, result_dict: Dict) -> bool:
        """Parse CPU by pod results."""
        logger = BlazeLogger.get_instance()
        
        for r in results:
            try:
                pod_name = r['metric'].get('pod', '')
                if pod_name:
                    result_dict[(layer_id, pod_name)] = float(r['value'][1])
                    logger.debug(f"  [{layer_id}] CPU (pod): {pod_name} = {result_dict[(layer_id, pod_name)]:.3f}")
            except Exception as e:
                logger.debug(f"  Error parsing CPU: {e}")
        return len(result_dict) > 0
    
    def _query_cpu_with_pod_info(self, layer_id: str, prom_url: str, namespace: str, result_dict: Dict) -> bool:
        """Query CPU with kube_pod_info join."""
        logger = BlazeLogger.get_instance()
        
        node_label_key = self._discover_node_label(prom_url, namespace)
        node_lbl = node_label_key or 'node'
        
        cpu_join_query = (
            f"sum by ({node_lbl}) ("
            f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod!="",container!=""}}[5m])'
            ") * on(pod) group_left(" + node_lbl + ") ("
            f'kube_pod_info{{namespace="{namespace}"}}'
            ")"
        )
        
        logger.debug(f"  [{layer_id}] Retrying CPU with kube_pod_info join on '{node_lbl}'")
        try:
            cpu_join_results = self.query_client.query_at_url(cpu_join_query, prom_url)
            if cpu_join_results:
                for r in cpu_join_results:
                    try:
                        node = r['metric'].get(node_lbl) or r['metric'].get('node')
                        node = self.base_updater._normalize_node_id(node)
                        if node:
                            result_dict[(layer_id, node)] = float(r['value'][1])
                            logger.debug(f"  [{layer_id}] CPU (join): {node} = {result_dict[(layer_id, node)]:.3f}")
                    except Exception:
                        pass
                return len(result_dict) > 0
        except Exception:
            pass
        return False
    
    def _query_ram_metrics(self, layer_id: str, prom_url: str, namespace: str, result_dict: Dict) -> bool:
        """Query RAM by node with fallbacks. Returns True if metrics found."""
        logger = BlazeLogger.get_instance()
        
        # Primary: RAM by node
        ram_query = f'sum(container_memory_working_set_bytes{{namespace="{namespace}",container!=""}}) by (node)'
        ram_results = self.query_client.query_at_url(ram_query, prom_url)
        
        if ram_results:
            return self._parse_ram_results(layer_id, ram_results, result_dict)
        
        # Fallback 1: RAM by pod
        ram_query_fallback = f'sum by (pod)(container_memory_working_set_bytes{{namespace="{namespace}",pod!="",container!=""}})'
        logger.debug(f"  [{layer_id}] Retrying RAM query with 'pod' label")
        ram_results = self.query_client.query_at_url(ram_query_fallback, prom_url)
        
        if ram_results:
            return self._parse_ram_pod_results(layer_id, ram_results, result_dict)
        
        # Fallback 2: Join with kube_pod_info
        return self._query_ram_with_pod_info(layer_id, prom_url, namespace, result_dict)
    
    def _parse_ram_results(self, layer_id: str, results: List, result_dict: Dict) -> bool:
        """Parse RAM by node results."""
        logger = BlazeLogger.get_instance()
        
        found = 0
        for r in results:
            try:
                node_raw = r['metric'].get('node', '')
                node = self.base_updater._normalize_node_id(node_raw)
                if node:
                    result_dict[(layer_id, node)] = float(r['value'][1]) / (1024 ** 3)
                    found += 1
                    logger.debug(f"  [{layer_id}] RAM: {node} = {result_dict[(layer_id, node)]:.3f} GB")
            except Exception as e:
                logger.debug(f"  Error parsing RAM: {e}")
        return found > 0
    
    def _parse_ram_pod_results(self, layer_id: str, results: List, result_dict: Dict) -> bool:
        """Parse RAM by pod results."""
        logger = BlazeLogger.get_instance()
        
        for r in results:
            try:
                pod_name = r['metric'].get('pod', '')
                if pod_name:
                    result_dict[(layer_id, pod_name)] = float(r['value'][1]) / (1024 ** 3)
                    logger.debug(f"  [{layer_id}] RAM (pod): {pod_name} = {result_dict[(layer_id, pod_name)]:.3f} GB")
            except Exception as e:
                logger.debug(f"  Error parsing RAM: {e}")
        return len(result_dict) > 0
    
    def _query_ram_with_pod_info(self, layer_id: str, prom_url: str, namespace: str, result_dict: Dict) -> bool:
        """Query RAM with kube_pod_info join."""
        logger = BlazeLogger.get_instance()
        
        node_label_key = self._discover_node_label(prom_url, namespace)
        node_lbl = node_label_key or 'node'
        
        ram_join_query = (
            f"sum by ({node_lbl}) ("
            f'container_memory_working_set_bytes{{namespace="{namespace}",pod!="",container!=""}}'
            ") * on(pod) group_left(" + node_lbl + ") ("
            f'kube_pod_info{{namespace="{namespace}"}}'
            ")"
        )
        
        logger.debug(f"  [{layer_id}] Retrying RAM with kube_pod_info join on '{node_lbl}'")
        try:
            ram_join_results = self.query_client.query_at_url(ram_join_query, prom_url)
            if ram_join_results:
                for r in ram_join_results:
                    try:
                        node = r['metric'].get(node_lbl) or r['metric'].get('node')
                        node = self.base_updater._normalize_node_id(node)
                        if node:
                            result_dict[(layer_id, node)] = float(r['value'][1]) / (1024 ** 3)
                            logger.debug(f"  [{layer_id}] RAM (join): {node} = {result_dict[(layer_id, node)]:.3f} GB")
                    except Exception:
                        pass
                return len(result_dict) > 0
        except Exception:
            pass
        return False
    
    def _discover_node_label(self, prom_url: str, namespace: str) -> Optional[str]:
        """Discover the node label key from kube_pod_info."""
        try:
            results = self.query_client.query_at_url(f'kube_pod_info{{namespace="{namespace}"}}', prom_url)
            if results:
                for r in results:
                    for candidate in ('node', 'node_name', 'exported_node', 'kubernetes_node', 'kubernetes_io_hostname', 'host'):
                        if candidate in r['metric']:
                            return candidate
        except Exception:
            pass
        return None
