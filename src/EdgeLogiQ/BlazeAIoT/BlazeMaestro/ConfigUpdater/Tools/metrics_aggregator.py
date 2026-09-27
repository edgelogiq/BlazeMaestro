"""Aggregates metrics across multiple clusters."""
from typing import Dict, Tuple

from BlazeAIoT.Core.Utils import BlazeLogger
from ..service_metrics_collector import ServiceMetricsCollector


class MetricsAggregator:
    """Helper class to aggregate metrics across multiple clusters."""
    
    def __init__(self, base_updater):
        self.base_updater = base_updater
    
    def aggregate_cluster_metrics(self, cluster_configs, metrics_facade, 
                                  service_collector, namespace: str) -> Tuple[Dict, Dict]:
        """Aggregate metrics from all clusters. Returns (metrics, snapshot)."""
        logger = BlazeLogger.get_instance()
        
        aggregated = {}
        snapshot = {'namespace': namespace, 'workloads': {}, 'source': 'aggregated'}
        
        for cluster in cluster_configs:
            try:
                ctx = cluster.get('context')
                kubeconfig = cluster.get('kubeconfig_path')
                layer_id = cluster.get('layer_id')
                
                collector = ServiceMetricsCollector(
                    service_collector.query_client,
                    kube_context=ctx,
                    kubeconfig_path=kubeconfig,
                    cluster_id=layer_id
                )
                metrics = collector.collect(namespace)
                cluster_snapshot = getattr(collector, 'last_discovery_snapshot', {}) or {}
                
                self._merge_metrics(aggregated, metrics, namespace)
                snapshot['workloads'].update(cluster_snapshot.get('workloads', {}))
                
            except Exception as e:
                logger.warning(f"Cluster {cluster.get('name', layer_id)} failed: {e}")
        
        logger.debug(f"Aggregated metrics for {len(aggregated)} workloads")
        return aggregated, snapshot
    
    def _merge_metrics(self, aggregated: Dict, cluster_metrics: Dict, namespace: str):
        """Merge cluster metrics into aggregated dict, handling deduplication."""
        logger = BlazeLogger.get_instance()
        
        for workload_name, metrics in cluster_metrics.items():
            if workload_name not in aggregated:
                aggregated[workload_name] = {
                    'workload_type': metrics.get('workload_type', 'Deployment'),
                    'by_node': {},
                    'app': metrics.get('app'),
                    'pods': list(metrics.get('pods', []) or [])
                }
            
            # Merge workload type preference (prefer DaemonSet)
            if metrics.get('workload_type') == 'DaemonSet':
                aggregated[workload_name]['workload_type'] = 'DaemonSet'
            
            # Preserve app label
            if not aggregated[workload_name].get('app') and metrics.get('app'):
                aggregated[workload_name]['app'] = metrics.get('app')
            
            # Accumulate pods with deduplication
            existing_pods = {p.get('name') for p in aggregated[workload_name]['pods']}
            for pod in metrics.get('pods', []) or []:
                if pod.get('name') not in existing_pods:
                    aggregated[workload_name]['pods'].append(pod)
                    existing_pods.add(pod.get('name'))
            
            # Merge by_node metrics
            for node_name, node_vals in (metrics.get('by_node', {}) or {}).items():
                if node_vals.get('pod_count', 0) == 0:
                    continue
                
                if node_name in aggregated[workload_name]['by_node']:
                    target = aggregated[workload_name]['by_node'][node_name]
                    target['cpu_usage'] += float(node_vals.get('cpu_usage', 0.0))
                    target['ram_usage_gb'] += float(node_vals.get('ram_usage_gb', 0.0))
                    target['pod_count'] += int(node_vals.get('pod_count', 0))
                    if 'exec_time' in node_vals and 'exec_time' not in target:
                        target['exec_time'] = node_vals['exec_time']
                    if 'execution_rate_hz' in node_vals and 'execution_rate_hz' not in target:
                        target['execution_rate_hz'] = node_vals['execution_rate_hz']
                else:
                    aggregated[workload_name]['by_node'][node_name] = {
                        'cpu_usage': float(node_vals.get('cpu_usage', 0.0)),
                        'ram_usage_gb': float(node_vals.get('ram_usage_gb', 0.0)),
                        'pod_count': int(node_vals.get('pod_count', 0))
                    }
                    if 'exec_time' in node_vals:
                        aggregated[workload_name]['by_node'][node_name]['exec_time'] = node_vals['exec_time']
                    if 'execution_rate_hz' in node_vals:
                        aggregated[workload_name]['by_node'][node_name]['execution_rate_hz'] = node_vals['execution_rate_hz']
