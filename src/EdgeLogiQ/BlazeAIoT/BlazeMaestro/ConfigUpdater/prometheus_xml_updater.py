"""
Update XML configuration files with real data from Prometheus.

This module queries Prometheus for actual service resource usage and node capacity,
then updates platform.xml and services.xml to reflect reality.

Refactored using composition pattern with specialized helper classes.

ARCHITECTURE - Clean Metrics Association (Jan 2026):
====================================================

PROBLEM SOLVED:
- Metrics were being mixed between pod instances on different nodes
- Multiple replicas of same service were losing their individual data
- Deduplication was incorrectly removing pods on different nodes

SOLUTION - Pod-Node Metrics Manager:
- Maintains (pod_name, node_name) → metrics mapping in memory
- Ensures each pod instance on each node has its own metrics
- Prevents any cross-contamination between different instances

UPDATE FLOW:
1. EXTRACT: Read pod names from existing XML (preserve baseline)
2. DISCOVER: Query Prometheus for active pods → create NEW instances ONLY
3. PRESERVE: Keep existing instances with their metrics as baseline
4. MAP: Build (pod_name, node_name) → metrics lookup from Prometheus
5. UPDATE: Apply metrics ONLY to correct (pod_name, node) pair (never mix)
6. WRITE: Save updated XML with segregated metrics

CRITICAL RULES:
✓ Each pod instance on each node must have distinct metrics entry
✓ Discovery only ADDS new pod instances, never REMOVES existing
✓ Metrics lookup by (pod_name, node_name) pair, only
✓ If metrics not found for a pod-node pair, log warning and skip (no fallback)
✓ Deduplication only removes same pod on SAME node with different representations (IP vs hostname)
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_xml_updater import BaseXMLUpdater
from .prometheus_query_client import PrometheusQueryClient
from .node_metrics_collector import NodeMetricsCollector
from .service_metrics_collector import ServiceMetricsCollector
from .platform_xml_updater import PlatformXMLUpdater
from .services_xml_updater import ServicesXMLUpdater
from .node_resolver import NodeResolver
from .instance_deduplicator import InstanceDeduplicator
from .Tools import (
    MetricsAggregator,
    InstanceMetricsUpdater,
    InstanceSeeder,
    ConfigUpdateHelper,
    PodNodeMetricsManager
)


class PrometheusXMLUpdater(BaseXMLUpdater):
    """Orchestrate XML configuration updates from Prometheus metrics."""

    DEFAULT_LAYER_PREFIXES = {'edge', 'fog', 'cloud', 'local'}

    def __init__(self, prometheus_url: str = None, verify_ssl: bool = False,
                 kube_context: str = None, kubeconfig_path: str = None,
                 cluster_urls: Dict[str, str] = None,
                 metrics_facade = None, output_dir: str = None):
        """Initialize PrometheusXMLUpdater with collectors and updaters."""
        super().__init__()
        self.prometheus_url = prometheus_url
        self.verify_ssl = verify_ssl
        self.cluster_urls = cluster_urls or {}
        self.metrics_facade = metrics_facade
        self.output_dir = output_dir or 'out'
        
        # Track iteration for latency matrix export
        self._iteration_counter = 0
        
        # Initialize query client
        self.query_client = PrometheusQueryClient(
            prometheus_url=prometheus_url,
            verify_ssl=verify_ssl
        )
        
        # Initialize collectors
        self.node_collector = NodeMetricsCollector(self.query_client)
        self.service_collector = ServiceMetricsCollector(
            self.query_client,
            kube_context=kube_context,
            kubeconfig_path=kubeconfig_path
        )
        
        # Initialize specialized XML updaters and helpers
        self.platform_updater = PlatformXMLUpdater(base_updater=self)
        self.services_updater = ServicesXMLUpdater(base_updater=self)
        self.config_helper = ConfigUpdateHelper(base_updater=self)
        
        # Initialize pod-node metrics manager for clean metric association
        self.metrics_manager = PodNodeMetricsManager()
        
        # Initialize node resolver (handles IP to name resolution)
        prometheus_urls = list(cluster_urls.values()) if cluster_urls else ([prometheus_url] if prometheus_url else [])
        cluster_configs = metrics_facade.cluster_configs if metrics_facade and hasattr(metrics_facade, 'cluster_configs') else []
        self.node_resolver = NodeResolver(
            prometheus_urls=prometheus_urls,
            cluster_configs=cluster_configs,
            verify_ssl=verify_ssl
        )
        
        # Initialize deduplicator (handles duplicate removal, needs node_resolver for IP resolution)
        self.deduplicator = InstanceDeduplicator(node_resolver=self.node_resolver)

        # Cache for instance metrics updates (avoid re-parsing every iteration)
        self._instances_tree = None
        self._instances_root = None
        self._instances_read_path = None
        self._instances_write_path = None
        self._instances_initialized = False

    # ============= Query Delegation =============
    def query_prometheus(self, query: str) -> Optional[List[Dict]]:
        """Query Prometheus using configured URL."""
        return self.query_client.query(query)

    def query_prometheus_at_url(self, query: str, url: str) -> Optional[List[Dict]]:
        """Query Prometheus at specific URL."""
        client = PrometheusQueryClient(prometheus_url=url, verify_ssl=self.verify_ssl)
        return client.query(query)

    # ============= Metrics Collection =============
    def get_node_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get node CPU/RAM/GPU metrics."""
        return self.node_collector.collect()

    def get_service_metrics(self, namespace: str = 'default', 
                          return_snapshot: bool = False) -> Any:
        """Get service metrics, multi-cluster aware via K8s discovery."""
        # Note: For XML pod metrics, use layer-aware routing via _collect_xml_pods_by_layer
        # This method is used for K8s discovery (STEP 2b) to get exec metrics
        if self.metrics_facade and hasattr(self.metrics_facade, 'cluster_configs'):
            cluster_configs = getattr(self.metrics_facade, 'cluster_configs', []) or []
            if len(cluster_configs) > 1:
                aggregator = MetricsAggregator(self)
                metrics, snapshot = aggregator.aggregate_cluster_metrics(
                    cluster_configs=cluster_configs,
                    metrics_facade=self.metrics_facade,
                    service_collector=self.service_collector,
                    namespace=namespace
                )
                if return_snapshot:
                    return metrics, snapshot
                return metrics
        
        metrics = self.service_collector.collect(namespace=namespace)
        if return_snapshot:
            return metrics, {'timestamp': datetime.now().isoformat()}
        return metrics

    # ============= XML Updates =============
    def update_platform_xml(self, xml_path: str, backup: bool = True) -> bool:
        """Update platform.xml with node metrics."""
        node_metrics = self.get_node_metrics()
        usage_cpu, usage_ram = self.config_helper.query_node_usage_metrics()
        node_metrics = self.config_helper.merge_usage_into_metrics(node_metrics, usage_cpu, usage_ram)
        allowed_layers = self.config_helper.get_allowed_layers(node_metrics=node_metrics)
        return self.platform_updater.update(xml_path, node_metrics, allowed_layers)

    def update_services_xml(self, xml_path: str, namespace: str = 'default', 
                           backup: bool = True) -> bool:
        """Update services.xml with service metrics."""
        service_metrics, snapshot = self.get_service_metrics(namespace, return_snapshot=True)
        return self.services_updater.update(xml_path, service_metrics, namespace, 
                                           discovery_snapshot=snapshot)

    def update_all_configs(self, platform_xml: str, services_xml: str, 
                          namespace: str = 'default') -> bool:
        """Update platform.xml, services.xml, and instance metrics."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        BlazeLogger.get_instance().info("=" * 60)
        BlazeLogger.get_instance().info("UPDATING CONFIGURATION FROM PROMETHEUS")
        BlazeLogger.get_instance().info("=" * 60)

        platform_ok = self.update_platform_xml(platform_xml)
        services_ok = self.update_services_xml(services_xml, namespace)
        
        is_empty = self.config_helper.is_services_xml_empty(services_xml)
        services_update_xml = services_xml.replace('.xml', '_update.xml')
        target = services_xml if is_empty else services_update_xml
        instances_ok = self.update_instance_metrics(target, namespace)

        if platform_ok and services_ok and instances_ok:
            BlazeLogger.get_instance().debug("✓ Configuration updated successfully")
            return True
        else:
            BlazeLogger.get_instance().warning("⚠ Some updates failed")
            return False

    # ============= Instance Metrics =============
    def initialize_instance_metrics_cache(self, xml_path: str = None, namespace: str = 'default') -> bool:
        """Initialize by loading services.xml ONCE into memory. Call this at startup."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        if xml_path is None:
            BlazeLogger.get_instance().warning("No xml_path provided for initialization")
            return False
        
        # Determine base and update paths
        base_path = xml_path.replace('_update.xml', '.xml')
        update_path = xml_path if xml_path.endswith('_update.xml') else xml_path.replace('.xml', '_update.xml')
        
        # Load base services.xml (source of truth with all historical data)
        # Keep in memory and update throughout lifecycle
        read_path = base_path
        write_path = update_path
        
        if not os.path.exists(read_path):
            BlazeLogger.get_instance().error(f"Cannot initialize: {read_path} not found")
            return False
        
        BlazeLogger.get_instance().info(f"[INIT] Loading services XML from {read_path} (base file with historical data)")
        self._instances_tree = ET.parse(read_path)
        self._instances_root = self._instances_tree.getroot()
        self._instances_read_path = read_path
        self._instances_write_path = write_path
        self._instances_initialized = True
        BlazeLogger.get_instance().info(f"[INIT] Cached services XML in memory, will write updates to {write_path}")
        
        # Discover available metrics for debugging Rancher/Prometheus setup
        logger = BlazeLogger.get_instance()
        logger.info("[INIT] Discovering available metrics in Prometheus...")
        try:
            available = self.service_collector.discover_available_metrics()
            for pattern, metrics in available.items():
                if metrics:
                    logger.info(f"[INIT] Available metrics matching '{pattern}': {len(metrics)} found")
        except Exception as e:
            logger.warning(f"[INIT] Could not discover metrics: {e}")
        
        return True
    
    def update_instance_metrics(self, xml_path: str = None, namespace: str = 'default',
                               metrics_provider=None, **kwargs) -> bool:
        """Update instance metrics in cached memory. Follows workflow: discover -> fetch metrics -> update in-memory -> write."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        if metrics_provider is not None:
            self.metrics_facade = metrics_provider
        
        # Initialize if not already done
        if not self._instances_initialized:
            if xml_path is None:
                xml_path = kwargs.get('services_xml_path', 'services.xml')
            if not self.initialize_instance_metrics_cache(xml_path, namespace):
                return False
        
        # Use cached tree and paths
        BlazeLogger.get_instance().debug(f"[ITERATION] Updating metrics in cached XML tree, will write to {self._instances_write_path}")
        
        result = self._update_instance_metrics_impl(
            self._instances_read_path, 
            self._instances_write_path, 
            namespace=namespace,
            tree=self._instances_tree,
            root=self._instances_root
        )
        
        # Export latency matrix after metrics update (if metrics_provider available)
        nodes = kwargs.get('nodes')
        if result and self.metrics_facade and nodes:
            self._iteration_counter += 1
            self._export_latency_matrix_if_enabled(nodes, self._iteration_counter)
        
        return result

    def _extract_pods_from_xml(self, root: ET.Element) -> List[Dict]:
        """Extract pod names, nodes, and layers from XML for layer-aware Prometheus queries."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        xml_pods = []
        for service in root.findall('service'):
            service_id = service.get('id')
            instances = service.find('instances')
            if instances is None:
                continue
                
            for container in instances.findall('container'):
                pod_name = container.get('pod_name')
                node_name = container.get('node')
                layer = container.get('layer')  # Don't default to 'default' - preserve existing value
                
                if pod_name and node_name:
                    xml_pods.append({
                        'service_id': service_id,
                        'pod_name': pod_name,
                        'node_name': self._normalize_node_id(node_name),
                        'layer': layer
                    })
        
        logger.debug(f"[XML_EXTRACT] Extracted {len(xml_pods)} pods from XML with layer info")
        return xml_pods
    
    def _collect_xml_pods_by_layer(self, xml_pods: List[Dict], namespace: str) -> Dict[str, Dict]:
        """Collect metrics for XML pods grouped by layer, querying the correct Prometheus for each layer."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        # Group pods by layer
        pods_by_layer = {}
        for pod in xml_pods:
            layer = pod.get('layer')
            if not layer:  # Skip pods with no layer - can't route to Prometheus
                continue
            if layer not in pods_by_layer:
                pods_by_layer[layer] = []
            pods_by_layer[layer].append(pod)
        
        # Collect metrics from each layer's Prometheus
        all_metrics = {}
        for layer, layer_pods in pods_by_layer.items():
            try:
                # Get Prometheus URL for this layer from cluster_urls
                prometheus_url = self.cluster_urls.get(layer)
                
                if not prometheus_url:
                    continue
                
                # Create layer-specific collector
                layer_query_client = PrometheusQueryClient(
                    prometheus_url=prometheus_url,
                    verify_ssl=self.verify_ssl
                )
                layer_collector = ServiceMetricsCollector(
                    layer_query_client,
                    kube_context=self.service_collector.kube_context,
                    kubeconfig_path=self.service_collector.kubeconfig_path
                )
                
                # Query this layer's Prometheus
                layer_metrics = layer_collector.collect_from_xml_pods(layer_pods, namespace)
                
                # Merge into all_metrics
                for service_id, svc_data in layer_metrics.items():
                    if service_id not in all_metrics:
                        all_metrics[service_id] = {
                            'workload_type': svc_data.get('workload_type', 'Deployment'),
                            'by_node': {},
                            'app': svc_data.get('app'),
                            'pods': []
                        }
                    
                    # Merge pods
                    all_metrics[service_id]['pods'].extend(svc_data.get('pods', []))
                    
                    # Merge by_node metrics
                    for node_name, node_data in svc_data.get('by_node', {}).items():
                        all_metrics[service_id]['by_node'][node_name] = node_data
                
            except Exception as e:
                logger.error(f"[LAYER_ROUTING] Failed to query {layer} layer at {prometheus_url}: {e}", exc_info=True)
        
        return all_metrics
    
    def _update_instance_metrics_impl(self, read_path: str, write_path: str = None,
                                     namespace: str = 'default',
                                     tree: Optional[ET.ElementTree] = None,
                                     root: Optional[ET.Element] = None) -> bool:
        """Core implementation following 5-step workflow:
        1. XML already cached in memory (done in initialization)
        2. Discovery: detect pod changes and add missing instances (never delete)
        3. Fetch metrics from Prometheus
        4. Update in-memory metrics (last/current, min/max)
        5. Write to services_update.xml
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        # Node resolver cache is persistent - IPs don't change
        # Only build mapping once (on first call)
        
        if write_path is None:
            write_path = read_path
        
        if tree is None or root is None:
            logger.error("Tree and root must be provided (should be cached)")
            return False
        
        try:
            # STEP 1: Extract pod names from XML
            logger.debug("[STEP 1] Extracting pod names from XML")
            xml_pods = self._extract_pods_from_xml(root)
            logger.debug(f"[STEP 1] ✓ Extracted {len(xml_pods)} pods from XML")
            
            # STEP 2: Query Prometheus for XML pods (layer-aware)
            logger.debug("[STEP 2] Querying Prometheus for XML pods (layer-aware)")
            service_metrics = self._collect_xml_pods_by_layer(xml_pods, namespace)
            logger.debug(f"[STEP 2] ✓ Queried metrics for {len(service_metrics)} services across all layers")
            
            # Also run K8s discovery for exec metrics and new pods
            logger.debug("[STEP 2b] Running K8s discovery for exec metrics and new pods")
            k8s_metrics, _ = self.get_service_metrics(namespace, return_snapshot=True)
            
            # Build IP-to-name mapping for merging (using node resolver)
            ip_to_name = self.node_resolver.get_mapping(root)
            # Also build reverse mapping (name to IP)
            name_to_ip = {v: k for k, v in ip_to_name.items() if '.' in k}  # Only IP keys
            
            # Merge exec metrics from K8s discovery into XML pod metrics
            for service_id, k8s_data in k8s_metrics.items():
                if service_id in service_metrics:
                    # Merge exec_time and execution_rate_hz
                    for node_name, k8s_node_data in k8s_data.get('by_node', {}).items():
                        # Try both the node name and its IP equivalent
                        target_nodes = [node_name]
                        if node_name in name_to_ip:
                            target_nodes.append(name_to_ip[node_name])
                        
                        for target_node in target_nodes:
                            if target_node in service_metrics[service_id]['by_node']:
                                target = service_metrics[service_id]['by_node'][target_node]
                                if k8s_node_data.get('exec_time', 0.0) > 0:
                                    target['exec_time'] = k8s_node_data['exec_time']
                                    logger.debug(f"[EXEC_MERGE] {service_id}@{target_node}: merged exec_time={target['exec_time']}")
                                if k8s_node_data.get('execution_rate_hz', 0.0) > 0:
                                    target['execution_rate_hz'] = k8s_node_data['execution_rate_hz']
                                break  # Found a match, no need to try other variants
            
            # Seed new instances from K8s discovery
            seeder = InstanceSeeder(self)
            seeder.seed_instances_from_discovery(root, k8s_metrics, namespace)
            
            # Also seed pods discovered via SERVICE_LABEL (from Prometheus)
            # First, resolve node IPs to names in the pods list AND in by_node dictionary
            logger.debug("[STEP 2c] Seeding pods discovered via Prometheus SERVICE_LABEL")
            logger.debug(f"[STEP 2c] IP-to-name mapping: {ip_to_name}")
            
            for service_id, svc_data in service_metrics.items():
                pods = svc_data.get('pods', [])
                by_node = svc_data.get('by_node', {})
                logger.debug(f"[STEP 2c] Service {service_id}: {len(pods)} pods, by_node keys={list(by_node.keys())}")
                
                # Build service-specific mapping: resolved_name → all keys in THIS service's by_node
                resolved_to_keys = {}
                
                # First pass: resolve pods and build mapping of what keys need consolidation
                for pod in pods:
                    node_identifier = pod.get('node')
                    if node_identifier:
                        resolved = ip_to_name.get(node_identifier) or ip_to_name.get(node_identifier.lower())
                        if resolved and resolved != node_identifier:
                            pod['node'] = resolved
                            logger.debug(f"[STEP 2c] ✓ Resolved pod {pod['name']}: {node_identifier} → {resolved}")
                            
                            # Track which by_node keys need to be consolidated (per service)
                            if resolved not in resolved_to_keys:
                                resolved_to_keys[resolved] = set()
                            resolved_to_keys[resolved].add(node_identifier)
                            resolved_to_keys[resolved].add(node_identifier.lower())
                        elif not resolved:
                            logger.warning(f"[STEP 2c] ✗ Failed to resolve IP {node_identifier} for {service_id}")
                
                # Second pass: resolve ALL IP keys in by_node (not just from pods)
                # This handles cases where by_node has IPs that weren't in the pods list
                for ip_key in list(by_node.keys()):
                    if ip_key and '.' in ip_key:  # Likely an IP
                        resolved = ip_to_name.get(ip_key) or ip_to_name.get(ip_key.lower())
                        if resolved and resolved != ip_key:
                            ip_metrics = by_node.pop(ip_key)
                            if resolved in by_node:
                                # Both exist - merge them (take non-zero preferentially)
                                for metric_key in ['cpu_usage', 'ram_usage_gb', 'pod_count']:
                                    ip_val = ip_metrics.get(metric_key, 0)
                                    resolved_val = by_node[resolved].get(metric_key, 0)
                                    if ip_val > 0 and (resolved_val == 0 or ip_val > resolved_val):
                                        by_node[resolved][metric_key] = ip_val
                                # For exec metrics: only use IP version if resolved has 0
                                for metric_key in ['exec_time', 'execution_rate_hz']:
                                    ip_val = ip_metrics.get(metric_key, 0)
                                    resolved_val = by_node[resolved].get(metric_key, 0)
                                    if ip_val > 0 and resolved_val == 0:
                                        by_node[resolved][metric_key] = ip_val
                                logger.debug(f"[STEP 2c] {service_id}: Merged IP {ip_key} into {resolved}")
                            else:
                                # Only IP exists - move it
                                by_node[resolved] = ip_metrics
                                logger.debug(f"[STEP 2c] {service_id}: Resolved by_node key {ip_key} → {resolved}")
                
                # Third pass: consolidate by_node dict keys under resolved names FOR THIS SERVICE
                for resolved_name, key_variants in resolved_to_keys.items():
                    # Start with existing entry if resolved name already exists (preserves exec metrics from STEP 2b)
                    if resolved_name in by_node:
                        consolidated_metrics = by_node[resolved_name].copy()
                    else:
                        consolidated_metrics = {
                            'cpu_usage': 0.0, 'ram_usage_gb': 0.0,
                            'exec_time': 0.0, 'execution_rate_hz': 0.0, 'pod_count': 0
                        }
                    
                    # Merge from all IP/variant keys (but don't overwrite exec metrics with 0)
                    for key_variant in list(key_variants):
                        if key_variant in by_node and key_variant != resolved_name:
                            source = by_node.pop(key_variant)
                            # Merge: take non-zero values preferentially
                            for metric_key in ['cpu_usage', 'ram_usage_gb', 'pod_count']:
                                source_val = source.get(metric_key, 0)
                                if source_val > 0 and (consolidated_metrics.get(metric_key, 0) == 0 or source_val > consolidated_metrics.get(metric_key, 0)):
                                    consolidated_metrics[metric_key] = source_val
                            # For exec metrics: only overwrite if source has non-zero AND current is zero
                            for metric_key in ['exec_time', 'execution_rate_hz']:
                                source_val = source.get(metric_key, 0)
                                current_val = consolidated_metrics.get(metric_key, 0)
                                if source_val > 0 and current_val == 0:
                                    consolidated_metrics[metric_key] = source_val
                            logger.debug(f"[STEP 2c] {service_id}: Merged by_node key {key_variant} → {resolved_name}")
                    
                    # Write consolidated metrics under the resolved name
                    by_node[resolved_name] = consolidated_metrics
                    logger.debug(f"[STEP 2c] {service_id}: Final by_node[{resolved_name}]: cpu={consolidated_metrics['cpu_usage']:.4f}, exec_time={consolidated_metrics.get('exec_time', 0):.6f}")
            
            seeder.seed_instances_from_discovery(root, service_metrics, namespace)
            logger.debug("[STEP 2c] ✓ Discovery complete, instances seeded")
            
            # ==============================================================
            # STEP 2d: Resolve IPs and Deduplicate (only on first run or when needed)
            # ==============================================================
            # Resolve all node IPs to names in XML (IPs don't change, so only needed once)
            resolved_count = self.node_resolver.resolve_all_in_xml(root)
            if resolved_count > 0:
                logger.info(f"[STEP 2d] ✓ Resolved {resolved_count} node IPs to names")
            
            # Remove duplicate instances (same pod + same node) - only if we found duplicates
            dedup_stats = self.deduplicator.deduplicate(root)
            if len(dedup_stats) > 0:
                logger.info(f"[STEP 2e] ✓ Removed duplicates from {len(dedup_stats)} services")
            
            # STEP 3: Fetch metrics from Prometheus
            logger.debug("[STEP 3] Fetching metrics from Prometheus")
            metrics_lookup = self._build_pod_node_metrics_lookup(service_metrics, root)
            logger.debug(f"[STEP 3] ✓ Fetched metrics: {len(metrics_lookup)} pod-node entries")
            
            # Reset metrics manager for this update cycle
            self.metrics_manager.reset_updates()
            
            # Register node IP resolutions with metrics manager
            ip_to_name = self.node_resolver.get_mapping(root)
            for node_ip, node_name in ip_to_name.items():
                if node_ip != node_name and node_name is not None:  # Skip if node_name is None
                    self.metrics_manager.register_node_resolution(node_ip, node_name)
            
            for (pod_name, node_name), metrics in metrics_lookup.items():
                # Find service ID for this pod
                service_id = None
                for service in root.findall('service'):
                    instances = service.find('instances')
                    if instances:
                        for c in instances.findall('container'):
                            if c.get('pod_name') == pod_name:
                                service_id = service.get('id')
                                break
                    if service_id:
                        break
                
                self.metrics_manager.add_metrics(pod_name, node_name, metrics, service_id)
            
            # STEP 4: Update in-memory metrics (last/current, min/max)
            # STEP 4: Update in-memory metrics (all IPs already resolved in STEP 2d)
            logger.debug("[STEP 4] Updating in-memory metrics")
            updated_count = 0
            skipped_no_data = 0
            
            for service in root.findall('service'):
                instances = service.find('instances')
                if instances is None:
                    continue
                
                service_id = service.get('id')
                
                # For each instance, get metrics ONLY from its specific (pod_name, node) pair
                for container in instances.findall('container'):
                    pod_name = container.get('pod_name')
                    node_name = container.get('node')  # Already resolved in STEP 2d
                    
                    if not node_name:
                        logger.debug(f"[STEP 4] Skipped {service_id}/container (no valid node)")
                        continue
                    
                    metrics_entry = None
                    metrics_source = "none"
                    
                    # CRITICAL: Only look up metrics for THIS SPECIFIC pod on THIS SPECIFIC node
                    if pod_name:
                        # Normalize for lookup (case-insensitive)
                        metrics_entry = self.metrics_manager.get_metrics(pod_name, node_name.lower())
                        if metrics_entry:
                            metrics_source = f"pod-specific ({pod_name}@{node_name})"
                        else:
                            # No metrics found for this pod on this node
                            logger.debug(
                                f"[STEP 4] No metrics found for {service_id}/{pod_name}@{node_name}"
                            )
                    else:
                        # Container without pod_name - no metrics available
                        logger.debug(f"[STEP 4] No pod_name for {service_id}@{node_name}, skipping metrics update")
                    
                    # Update via specialized updater if we have meaningful data
                    if metrics_entry:
                        # Check if metrics have meaningful values (not all zeros)
                        has_data = False
                        for key in ['cpu_usage', 'ram_usage_gb', 'exec_time', 'execution_rate_hz']:
                            val = metrics_entry.get(key)
                            if val is not None and float(val) > 0:
                                has_data = True
                                break
                        
                        if has_data:
                            updater = InstanceMetricsUpdater(self)
                            if updater.update_instance_metrics(container, metrics_entry, 
                                                              service_id, pod_name or 'unknown', node_name):
                                updated_count += 1
                                self.metrics_manager.mark_updated(pod_name or 'unknown', node_name)
                                logger.debug(f"[STEP 4] Updated {service_id}/{pod_name or 'unknown'}@{node_name} from {metrics_source}")
                        else:
                            skipped_no_data += 1
                            if skipped_no_data <= 5:  # Only log first few to avoid spam
                                logger.debug(f"[STEP 4] Skipped {service_id}/{pod_name or 'no-pod'}@{node_name} (no meaningful data)")
            
            logger.info(f"[STEP 4] ✓ Updated {updated_count} instances in memory{f', skipped {skipped_no_data} (no data)' if skipped_no_data > 0 else ''}")
            
            # STEP 4b: Sanitize XML - remove any None attribute values
            logger.debug("[STEP 4b] Sanitizing XML (removing None values)")
            
            # First pass: FIND all None values before removing
            none_found = []
            for service in root.findall('service'):
                instances = service.find('instances')
                if instances is None:
                    continue
                for container in instances.findall('container'):
                    for attr_name, attr_value in container.attrib.items():
                        if attr_value is None:
                            pod_name = container.get('pod_name', 'unknown')
                            service_id = service.get('id', 'unknown')
                            none_found.append(f"{service_id}/{pod_name}@{attr_name}")
                            logger.error(f"[STEP 4b] FOUND None attribute before sanitizing: {service_id}/{pod_name}@{attr_name}=None")
            
            if none_found:
                logger.error(f"[STEP 4b] Total None attributes found: {len(none_found)}")
            
            sanitized_count = self._sanitize_xml_attributes(root)
            if sanitized_count > 0:
                logger.warning(f"[STEP 4b] Removed {sanitized_count} None attribute values")

            # Remove empty instances blocks to prevent accumulation
            self._remove_empty_instances_blocks(root)
            
            # STEP 5: Write to services_update.xml
            logger.debug("[STEP 5] Writing updated XML to disk")
            try:
                self._indent_xml(root)
                logger.debug("[STEP 5] XML indentation complete")
            except Exception as e:
                logger.error(f"[STEP 5] Error during XML indentation: {e}", exc_info=True)
                raise
            
            try:
                tree.write(write_path, encoding='utf-8', xml_declaration=True)
                logger.info(f"[STEP 5] ✓ Wrote updates to {write_path}")
            except Exception as e:
                logger.error(f"[STEP 5] Error writing XML to disk: {e}", exc_info=True)
                # Try to identify which container has a None attribute
                for service in root.findall('service'):
                    instances = service.find('instances')
                    if instances is None:
                        continue
                    for container in instances.findall('container'):
                        for attr_name, attr_value in container.attrib.items():
                            if attr_value is None:
                                logger.error(f"[STEP 5] Found None attribute: {service.get('id')}/{container.get('pod_name')}@{attr_name}=None")
                raise
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating {read_path}: {e}")
            return False

    # ============= Utility Methods =============
    def _build_pod_node_metrics_lookup(self, service_metrics: Dict, root: ET.Element = None) -> Dict:
        """Build (pod_name, node_name) → metrics lookup. Resolve node IPs to names."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        # Get IP-to-name mapping from node resolver
        ip_to_name = self.node_resolver.get_mapping(root)
        logger.debug(f"[METRICS_LOOKUP] Using node resolver mapping with {len(ip_to_name)} entries")
        
        lookup = {}
        for service_name, svc_data in service_metrics.items():
            pods = svc_data.get('pods', []) or []
            by_node = svc_data.get('by_node', {}) or {}
            workload_exec_time = svc_data.get('exec_time')
            workload_exec_rate = svc_data.get('execution_rate_hz')
            # Count pods per node to avoid applying node-level exec metrics to multiple pods
            pod_count_by_node = {}
            for pod in pods:
                pod_node = pod.get('node')
                if pod_node:
                    resolved = ip_to_name.get(pod_node) or ip_to_name.get(pod_node.lower()) or pod_node
                    pod_count_by_node[resolved] = pod_count_by_node.get(resolved, 0) + 1
            
            # Build lookup from pods
            for pod in pods:
                pod_name = pod.get('name')
                node_identifier = pod.get('node')  # Could be IP or name
                if pod_name and node_identifier:
                    # Try to resolve IP to name
                    resolved_node = ip_to_name.get(node_identifier) or ip_to_name.get(node_identifier.lower()) or node_identifier
                    
                    # Safety check - resolved_node should never be None
                    if resolved_node is None:
                        logger.warning(f"[METRICS_LOOKUP] Resolved node is None for {pod_name}@{node_identifier}, skipping")
                        continue
                    
                    key = (pod_name, resolved_node)
                    # Prefer per-pod metrics directly from the pod entry
                    pod_cpu = pod.get('cpu_usage', 0.0) or 0.0
                    pod_ram = pod.get('ram_usage_gb', 0.0) or 0.0
                    pod_exec_time = pod.get('exec_time')
                    pod_exec_rate = pod.get('execution_rate_hz')

                    # Use node-level exec metrics only if there's a single pod on that node
                    node_metrics = {}
                    node_pod_count = pod_count_by_node.get(resolved_node, 0)
                    if node_pod_count <= 1:
                        node_metrics = by_node.get(resolved_node, {}) or by_node.get(node_identifier, {})
                    
                    final_exec_time = pod_exec_time or node_metrics.get('exec_time') or workload_exec_time
                    final_exec_rate = pod_exec_rate or node_metrics.get('execution_rate_hz') or workload_exec_rate
                    
                    # Safety: convert None to 0 for exec metrics
                    if final_exec_time is None:
                        final_exec_time = 0.0
                    if final_exec_rate is None:
                        final_exec_rate = 0.0
                    
                    lookup[key] = {
                        'cpu_usage': float(pod_cpu),
                        'ram_usage_gb': float(pod_ram),
                        'exec_time': final_exec_time,
                        'execution_rate_hz': final_exec_rate
                    }
                    logger.debug(f"[METRICS_LOOKUP] Pod entry: {pod_name} @ {resolved_node} (resolved from {node_identifier})")
            
            # Also include nodes with exec metrics even if no pod exists
            for node_identifier, node_metrics in by_node.items():
                # Check if this node has useful metrics
                has_exec = node_metrics.get('exec_time') is not None and float(node_metrics.get('exec_time', 0)) > 0
                has_cpu = node_metrics.get('cpu_usage') is not None and float(node_metrics.get('cpu_usage', 0)) > 0
                has_ram = node_metrics.get('ram_usage_gb') is not None and float(node_metrics.get('ram_usage_gb', 0)) > 0
                
                # Skip node-level metrics that don't have a real pod
                # They should be merged into real pods during STEP 2b
        
        pods_with_exec = sum(1 for v in lookup.values() if v.get('exec_time') is not None and float(v.get('exec_time', 0)) > 0)
        logger.info(f"Metrics lookup: {len(lookup)} entries, {pods_with_exec} with exec_time")
        return lookup
    
    def _sanitize_xml_attributes(self, root: ET.Element) -> int:
        """Remove any None attribute values from XML tree."""
        sanitized_count = 0
        for elem in root.iter():
            # Check all attributes
            attrs_to_remove = []
            for attr_name, attr_value in elem.attrib.items():
                if attr_value is None or str(attr_value).lower() == 'none':
                    attrs_to_remove.append(attr_name)
            
            # Remove None attributes
            for attr_name in attrs_to_remove:
                del elem.attrib[attr_name]
                sanitized_count += 1
        
        return sanitized_count

    def _normalize_node_id(self, node_name: str) -> Optional[str]:
        """Normalize node identifier."""
        return node_name.strip().lower() if node_name else None

    def _infer_layer_prefix(self, node_name: str) -> Optional[str]:
        """Infer layer prefix from node name."""
        if not node_name:
            return None
        name_lower = node_name.lower()
        # Frank nodes are physical edge devices
        if name_lower.startswith('frank'):
            return 'edge'
        for prefix in self.DEFAULT_LAYER_PREFIXES:
            if name_lower.startswith(prefix):
                return prefix
        return None
    
    def _sanitize_xml_attributes(self, root: ET.Element) -> int:
        """Remove any None attribute values from XML tree (prevents serialization errors)."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        sanitized_count = 0
        
        for elem in root.iter():
            # Check all attributes of this element
            attrs_to_remove = []
            for attr_name, attr_value in list(elem.attrib.items()):  # Use list() to avoid dict size change during iteration
                if attr_value is None:
                    attrs_to_remove.append(attr_name)
                    sanitized_count += 1
                    logger.error(f"[SANITIZE] Found and removing None value from {elem.tag}@{attr_name} in element: {elem.attrib}")
            
            # Remove problematic attributes
            for attr_name in attrs_to_remove:
                del elem.attrib[attr_name]
                logger.warning(f"[SANITIZE] Removed None attribute: {elem.tag}@{attr_name}")
        
        return sanitized_count
    
    def _export_latency_matrix_if_enabled(self, nodes=None, iteration=None) -> None:
        """Export latency matrix to CSV after metrics update."""
        try:
            from ..Metrics.latency_matrix_exporter import LatencyMatrixExporter
            import os
            
            if not self.metrics_facade or not nodes:
                return
            
            # Create output directory (use timestamped dir from csv_logger if available)
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Use fixed filename (append mode) instead of timestamp
            csv_path = f"{output_dir}/latency_matrix_prometheus.csv"
            
            try:
                latency_matrix = self.metrics_facade.get_latency_matrix(nodes)
                LatencyMatrixExporter.export_to_csv(
                    latency_matrix=latency_matrix,
                    nodes=nodes,
                    output_path=csv_path,
                    source='prometheus',
                    iteration=iteration,
                    metadata={'mode': 'kubernetes'}
                )
            except Exception as e:
                from BlazeAIoT.Core.Utils import BlazeLogger
                BlazeLogger.get_instance().debug(f"Latency matrix export failed: {e}")
        except Exception:
            pass



