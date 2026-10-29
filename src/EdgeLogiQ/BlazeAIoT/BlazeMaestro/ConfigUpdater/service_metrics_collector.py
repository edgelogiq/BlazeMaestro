"""
Service metrics collector for gathering workload resource usage from Prometheus.

Discovers Kubernetes workloads (Deployments, DaemonSets, StatefulSets) and queries
their pod-level CPU/RAM metrics.
"""

from typing import Dict, Optional, List


class ServiceMetricsCollector:
    """Collects service/pod resource usage from Prometheus."""

    def __init__(self, query_client, kube_context: str = None, 
                 kubeconfig_path: str = None, cluster_id: str = None):
        """Initialize the service metrics collector."""
        self.query_client = query_client
        self.kube_context = kube_context
        self.kubeconfig_path = kubeconfig_path
        self.cluster_id = cluster_id
        self.last_discovery_snapshot: Optional[Dict] = None

    def collect(self, namespace: str = 'default') -> Dict[str, Dict]:
        """
        Get service/pod resource usage from Prometheus.

        Discovers workloads from Kubernetes, then queries metrics for their pods.

        Args:
            namespace: Kubernetes namespace to filter pods

        Returns:
            Dictionary mapping service IDs to their metrics
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        metrics = {}

        # Discover workloads from Kubernetes
        workload_pods = self._discover_workload_pods(namespace)

        # Store discovery snapshot for reuse
        self.last_discovery_snapshot = {
            'namespace': namespace,
            'workloads': workload_pods or {},
            'source': 'k8s_discovery'
        }

        if not workload_pods:
            BlazeLogger.get_instance().warning(f"No workloads in {namespace}, using generic pod query")
            metrics = self._get_generic_pod_metrics(namespace)
            self.last_discovery_snapshot['source'] = 'generic_pod_metrics'
        else:
            # Query metrics for each discovered workload
            for workload_name, workload_info in workload_pods.items():
                if not workload_info:
                    continue

                workload_type = workload_info.get('type', 'Deployment')
                pod_info_list = workload_info.get('pods', [])
                
                # Always create entry, even for scaled-to-0 deployments
                metrics[workload_name] = {
                    'workload_type': workload_type,
                    'by_node': {},
                    'app': workload_info.get('app'),
                    'pods': pod_info_list
                }

                # Query CPU/RAM for each pod
                for pod_info in pod_info_list:
                    pod_name = pod_info['name']
                    node_name = pod_info['node']
                    self._update_pod_metrics(metrics[workload_name], namespace, pod_name, node_name)

            # Collect execution metrics (blazeaiot_execution_time_us, execution_rate_hz)
            self._collect_execution_metrics(metrics, namespace, workload_pods)

        # Merge daemonset variants and multi-deployment variants
        metrics = self._merge_daemonset_variants(metrics)
        metrics = self._merge_multi_deployment_variants(metrics)

        # Align cached snapshot to merged shape
        self._update_snapshot(metrics, namespace)

        BlazeLogger.get_instance().info(f"Collected metrics for {len(metrics)} workloads")
        return metrics

    def collect_from_xml_pods(self, xml_pods: List[Dict], namespace: str = 'blazeaiot') -> Dict[str, Dict]:
        """
        Query Prometheus metrics for specific pods listed in the XML.
        
        This bypasses Kubernetes discovery and queries Prometheus directly for active pods
        by service labels, solving cross-cluster discovery issues.
        
        Args:
            xml_pods: List of dicts with keys: service_id (used to get service list)
            namespace: Kubernetes namespace for metrics
            
        Returns:
            Dictionary mapping service IDs to their metrics
        """
        # Extract unique service IDs from XML
        service_ids = list(set(pod['service_id'] for pod in xml_pods if pod.get('service_id')))
        
        metrics = {}
        
        # For each service, query Prometheus by labels to find ACTIVE pods only
        for service_id in service_ids:
            metrics[service_id] = {
                'workload_type': 'Deployment',
                'by_node': {},
                'app': service_id,
                'pods': []
            }
            
            # Query Prometheus by app label - only returns pods with active metrics
            self._update_service_by_label(metrics[service_id], service_id, namespace)
        
        # Collect execution metrics
        self._collect_execution_metrics(metrics, namespace, {})
        
        return metrics

    def _update_service_by_label(self, service_metrics: Dict, service_id: str, namespace: str):
        """Query metrics for ALL pods of a service using pod name patterns."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        # Query by pod name pattern (pod labels don't propagate to cAdvisor metrics)
        # Use rate() to get actual CPU usage rate in cores, not cumulative counter
        # Note: Syntax is "sum by (labels) (rate(...))" not "sum(rate(...)) by (labels)"
        cpu_queries = [
            f'sum by (pod, instance) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{service_id}.*",container!=""}}[1m]))',
            f'sum by (pod, instance) (rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{service_id}.*",container!=""}}[2m]))',
            f'sum by (pod, instance) (irate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{service_id}.*",container!=""}}[2m]))',
        ]
        logger.debug(f"[SERVICE_LABEL] {service_id}: Querying pods matching: {service_id}.*")
        
        cpu_results = None
        for cpu_query in cpu_queries:
            try:
                cpu_results = self._query_with_cluster_routing(cpu_query)
                if cpu_results and len(cpu_results) > 0:
                    break
            except Exception as e:
                logger.debug(f"[SERVICE_LABEL] {service_id}: Query error: {e}")
                continue
        
        if cpu_results and len(cpu_results) > 0:
            logger.debug(f"[SERVICE_LABEL] {service_id}: Found {len(cpu_results)} metrics")
            # Results are already summed by (pod, instance) from the query
            for result in cpu_results:
                try:
                    metric_labels = result.get('metric', {})
                    pod_name = metric_labels.get('pod')
                    node_ip = metric_labels.get('instance', '').split(':')[0]  # Extract node IP from instance
                    
                    if not pod_name or not node_ip:
                        continue
                    
                    try:
                        cpu_cores = float(result['value'][1])  # Already summed and rate-calculated
                    except (ValueError, KeyError, IndexError):
                        continue
                    
                    if cpu_cores > 0:
                        if node_ip not in service_metrics['by_node']:
                            service_metrics['by_node'][node_ip] = {
                                'cpu_usage': 0.0, 'ram_usage_gb': 0.0, 'pod_count': 0,
                                'exec_time': 0.0, 'execution_rate_hz': 0.0
                            }
                        service_metrics['by_node'][node_ip]['cpu_usage'] = cpu_cores
                        logger.debug(f"[SERVICE_LABEL] {service_id}/{pod_name}@{node_ip}: {cpu_cores:.4f} cores")
                        
                        # Add pod to pods list for seeding
                        service_metrics['pods'].append({
                            'name': pod_name,
                            'node': node_ip,
                            'cpu_usage': cpu_cores,
                            'ram_usage_gb': 0.0
                        })
                        
                        # Also query RAM for this pod
                        ram_query = f'sum(container_memory_working_set_bytes{{pod="{pod_name}",namespace="{namespace}",container!=""}})'
                        try:
                            ram_results = self._query_with_cluster_routing(ram_query)
                            if ram_results and len(ram_results) > 0:
                                try:
                                    ram_bytes = float(ram_results[0]['value'][1])
                                    if ram_bytes > 0:
                                        ram_gb = ram_bytes / (1024 ** 3)
                                        service_metrics['by_node'][node_ip]['ram_usage_gb'] = ram_gb
                                        # Update pod entry with RAM
                                        for pod in service_metrics['pods']:
                                            if pod['name'] == pod_name:
                                                pod['ram_usage_gb'] = ram_gb
                                                break
                                        logger.debug(f"[SERVICE_LABEL] {service_id}/{pod_name}@{node_ip}: {ram_gb:.4f} GB RAM")
                                except (ValueError, KeyError, IndexError):
                                    pass
                        except Exception as e:
                            logger.debug(f"[SERVICE_LABEL] {service_id}/{pod_name}: RAM query error: {e}")
                except (ValueError, KeyError, IndexError):
                    continue
        else:
            logger.debug(f"[SERVICE_LABEL] {service_id}: No results for pod pattern")

    def _update_pod_metrics(self, workload_metrics: Dict, namespace: str, 
                           pod_name: str, node_name: str):
        """Update metrics for a single pod. Supports Rancher Monitoring and standard Kubernetes metrics."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        if node_name not in workload_metrics['by_node']:
            workload_metrics['by_node'][node_name] = {
                'cpu_usage': 0.0, 'ram_usage_gb': 0.0, 'pod_count': 0,
                'exec_time': 0.0, 'execution_rate_hz': 0.0
            }
        
        node_data = workload_metrics['by_node'][node_name]
        
        # Query CPU usage - try multiple sources (Rancher, Kubelet, cAdvisor)
        cpu_queries = [
            # Rancher Monitoring - pod level
            f'sum(rancher_pod_cpu_usage{{namespace="{namespace}",pod="{pod_name}"}})',
            # Standard Kubernetes - container_cpu_usage_seconds_total rate
            f'sum(rate(container_cpu_usage_seconds_total{{pod="{pod_name}",namespace="{namespace}",container!=""}}[1m]))',
            # Alternative: without container exclusion
            f'sum(rate(container_cpu_usage_seconds_total{{pod="{pod_name}",namespace="{namespace}"}}[1m]))',
        ]
        
        cpu_val = 0.0
        for i, cpu_query in enumerate(cpu_queries):
            cpu_result = self._query_with_cluster_routing(cpu_query)
            if cpu_result and len(cpu_result) > 0:
                try:
                    cpu_val = float(cpu_result[0]['value'][1])
                    if cpu_val > 0:
                        node_data['cpu_usage'] = cpu_val  # Replace, don't accumulate
                        logger.debug(f"[CPU_METRICS] {pod_name}@{node_name}: {cpu_val:.4f} cores (query #{i+1})")
                        break
                except (ValueError, KeyError, IndexError):
                    continue
            elif i == 0:  # Log first query failure for debugging
                logger.debug(f"[CPU_METRICS] {pod_name}@{node_name}: query #{i+1} returned no results")
        
        if cpu_val == 0.0:
            logger.debug(f"[CPU_METRICS] {pod_name}@{node_name}: no CPU data - keeping historical value")
        
        # Query memory usage - try multiple sources
        mem_queries = [
            # Rancher Monitoring - pod level
            f'sum(rancher_pod_memory_usage_bytes{{namespace="{namespace}",pod="{pod_name}"}})',
            # Standard Kubernetes - working set bytes
            f'sum(container_memory_working_set_bytes{{pod="{pod_name}",namespace="{namespace}",container!=""}})',
            # Alternative: without container exclusion
            f'sum(container_memory_working_set_bytes{{namespace="{namespace}",pod="{pod_name}"}})',
        ]
        
        for i, mem_query in enumerate(mem_queries):
            mem_result = self._query_with_cluster_routing(mem_query)
            if mem_result and len(mem_result) > 0:
                try:
                    mem_bytes = float(mem_result[0]['value'][1])
                    if mem_bytes > 0:
                        mem_gb = mem_bytes / (1024 ** 3)
                        node_data['ram_usage_gb'] = mem_gb  # Replace, don't accumulate
                        logger.debug(f"[RAM_METRICS] {pod_name}@{node_name}: {mem_gb:.4f} GB ({mem_bytes:.0f} bytes, query #{i+1})")
                        break
                except (ValueError, KeyError, IndexError):
                    continue
        
        if node_data['ram_usage_gb'] == 0.0:
            logger.debug(f"[RAM_METRICS] {pod_name}@{node_name}: no RAM data - keeping historical value")
        
        node_data['pod_count'] += 1

    def discover_available_metrics(self, patterns: list = None) -> Dict[str, list]:
        """Discover what metrics are available in Prometheus. Useful for debugging.
        
        Args:
            patterns: List of metric name patterns to search for (e.g., ['rancher', 'container', 'pod'])
        
        Returns:
            Dictionary of pattern -> list of matching metric names
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        logger = BlazeLogger.get_instance()
        
        if not patterns:
            patterns = ['rancher_pod', 'container_cpu', 'container_memory', 'pod_cpu', 'pod_memory']
        
        results = {}
        
        try:
            # For each pattern, try to find matching metrics
            for pattern in patterns:
                query = f'{{__name__=~"{pattern}.*"}}'
                try:
                    metrics_result = self.query_client.query(query)
                    if metrics_result:
                        metric_names = set()
                        for result in metrics_result:
                            if '__name__' in result.get('metric', {}):
                                metric_names.add(result['metric']['__name__'])
                        results[pattern] = sorted(list(metric_names))
                except Exception as e:
                    logger.warning(f"[METRIC_DISCOVERY] Error querying pattern '{pattern}': {e}")
        except Exception as e:
            logger.warning(f"[METRIC_DISCOVERY] Error discovering metrics: {e}")
        
        return results

    def _discover_workload_pods(self, namespace: str) -> Dict[str, dict]:
        """Discover running pods organized by parent workload."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        workload_pods = {}

        try:
            from kubernetes import client, config
        except ImportError:
            BlazeLogger.get_instance().warning("kubernetes not available")
            return workload_pods

        try:
            # Load kubeconfig
            if self.kubeconfig_path and self.kubeconfig_path != 'null':
                config.load_kube_config(config_file=self.kubeconfig_path, context=self.kube_context)
            elif self.kube_context:
                config.load_kube_config(context=self.kube_context)
            else:
                config.load_kube_config()

            v1 = client.CoreV1Api()
            apps_v1 = client.AppsV1Api()

            # Discover Deployments
            deployments = apps_v1.list_namespaced_deployment(namespace, timeout_seconds=10)
            for deployment in deployments.items:
                pod_list = self._get_pods_for_workload(v1, namespace, deployment.spec.selector.match_labels)
                workload_pods[deployment.metadata.name] = {
                    'type': 'Deployment',
                    'pods': pod_list,
                    'app': self._extract_app_label(deployment.spec.template.metadata)
                }

            # Discover DaemonSets
            daemonsets = apps_v1.list_namespaced_daemon_set(namespace, timeout_seconds=10)
            for daemonset in daemonsets.items:
                pod_list = self._get_pods_for_workload(v1, namespace, daemonset.spec.selector.match_labels, 
                                                       daemonset.metadata.name)
                workload_pods[daemonset.metadata.name] = {
                    'type': 'DaemonSet',
                    'pods': pod_list,
                    'app': self._extract_app_label(daemonset.spec.template.metadata)
                }

            BlazeLogger.get_instance().debug(f"Discovered workloads: {list(workload_pods.keys())}")
            
            # Debug: Show discovered pods for key workloads
            for wl_name, wl_info in workload_pods.items():
                if 'robot' in wl_name.lower():
                    pod_list = wl_info.get('pods', [])
                    pod_names = [p['name'] for p in pod_list]
                    BlazeLogger.get_instance().debug(f"[DISCOVERY] {wl_name}: {len(pod_names)} pods - {pod_names}")

        except Exception as e:
            BlazeLogger.get_instance().error(f"Failed to discover workloads: {e}")

        return workload_pods

    def _get_pods_for_workload(self, v1, namespace: str, selector: Dict, 
                              daemonset_name: str = None) -> List[Dict]:
        """Get running pods matching a selector, with optional daemonset owner check."""
        label_selector = ','.join([f"{k}={v}" for k, v in selector.items()])
        pods = v1.list_namespaced_pod(namespace, label_selector=label_selector, timeout_seconds=10)
        
        running_pods = []
        for p in pods.items:
            if p.status.phase != 'Running':
                continue
            
            # If daemonset_name provided, verify ownership
            if daemonset_name:
                if not self._is_owned_by_daemonset(p, daemonset_name):
                    continue
            
            running_pods.append({
                'name': p.metadata.name,
                'node': p.spec.node_name,
                'ip': p.status.pod_ip
            })
        
        return running_pods

    def _is_owned_by_daemonset(self, pod, daemonset_name: str) -> bool:
        """Check if pod is owned by daemonset via ownerReferences, labels, or name prefix."""
        # Check ownerReferences
        owners = getattr(pod.metadata, 'owner_references', None) or []
        for o in owners:
            if getattr(o, 'kind', '') == 'DaemonSet' and getattr(o, 'name', '') == daemonset_name:
                return True
        
        # Check app labels
        labels = getattr(pod.metadata, 'labels', {}) or {}
        app_name = labels.get('app.kubernetes.io/name') or labels.get('app')
        if app_name == daemonset_name:
            return True
        
        # Check name prefix
        if pod.metadata.name and pod.metadata.name.startswith(daemonset_name + '-'):
            return True
        
        return False

    def _extract_app_label(self, metadata) -> str:
        """Extract app label from pod metadata."""
        try:
            labels = (metadata.labels or {}) if metadata else {}
            return labels.get('app.kubernetes.io/name') or labels.get('app')
        except Exception:
            return None

    def _query_with_cluster_routing(self, query: str) -> Optional[List[Dict]]:
        """Query metrics, routing to cluster-specific Prometheus if needed."""
        if not self.cluster_id:
            return self.query_client.query(query)
        
        cluster_urls = getattr(self.query_client, 'cluster_urls', {})
        if self.cluster_id in cluster_urls:
            try:
                return self.query_client.query_at_url(query, cluster_urls[self.cluster_id], layer_name=self.cluster_id)
            except Exception:
                pass
        
        return self.query_client.query(query)

    def _get_generic_pod_metrics(self, namespace: str) -> Dict[str, Dict]:
        """Fallback: query pod metrics generically without workload discovery."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        metrics = {}

        # Query pod metrics by pod name
        cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}",container!=""}}[1m])) by (pod)'
        mem_query = f'sum(container_memory_working_set_bytes{{namespace="{namespace}",container!=""}}) by (pod)'
        
        cpu_results = self._query_with_cluster_routing(cpu_query)
        mem_results = self._query_with_cluster_routing(mem_query)

        # Build pod-level metrics
        pod_metrics = {}
        if cpu_results:
            for result in cpu_results:
                pod_name = result['metric'].get('pod')
                if pod_name:
                    pod_metrics[pod_name] = {'cpu_usage': float(result['value'][1])}

        if mem_results:
            for result in mem_results:
                pod_name = result['metric'].get('pod')
                if pod_name:
                    if pod_name not in pod_metrics:
                        pod_metrics[pod_name] = {}
                    mem_gb = float(result['value'][1]) / (1024 ** 3)
                    pod_metrics[pod_name]['ram_usage_gb'] = mem_gb

        # Convert to workload format (infer workload from pod name)
        for pod_name, pod_data in pod_metrics.items():
            workload_name = self._infer_workload_from_pod_name(pod_name)
            
            if workload_name not in metrics:
                metrics[workload_name] = {
                    'workload_type': 'Deployment',
                    'by_node': {},
                    'app': workload_name.split('-')[0],
                    'pods': [{'name': pod_name, 'node': 'generic', 'ip': None}]
                }
            else:
                metrics[workload_name]['pods'].append({'name': pod_name, 'node': 'generic', 'ip': None})

            # Accumulate metrics under generic node
            node_data = metrics[workload_name]['by_node'].setdefault('generic', {
                'cpu_usage': 0.0, 'ram_usage_gb': 0.0, 'pod_count': 0,
                'exec_time': 0.0, 'execution_rate_hz': 0.0
            })
            node_data['cpu_usage'] += pod_data.get('cpu_usage', 0.0)
            node_data['ram_usage_gb'] += pod_data.get('ram_usage_gb', 0.0)
            node_data['pod_count'] += 1

        BlazeLogger.get_instance().debug(f"Generic pod query: {len(pod_metrics)} pods → {len(metrics)} workloads")
        return metrics

    def _infer_workload_from_pod_name(self, pod_name: str) -> str:
        """Infer workload name from pod name (handle daemonset variants)."""
        for suffix in ('-cpu-daemonset-', '-gpu-daemonset-', '-jetson-daemonset-'):
            if suffix in pod_name:
                return pod_name.split(suffix)[0] + suffix.rstrip('-')
        return pod_name

    def _merge_daemonset_variants(self, metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """Merge daemonset variants (cpu/gpu/jetson) under common base."""
        
        merged = {}
        merge_count = 0

        for name, metrics_data in metrics.items():
            if metrics_data.get('workload_type') != 'DaemonSet':
                merged[name] = metrics_data
                continue

            # Strip daemonset variant suffixes
            base_key = name
            for suffix in ('-cpu-daemonset', '-gpu-daemonset', '-jetson-daemonset', '-daemonset'):
                if name.endswith(suffix):
                    base_key = name[: -len(suffix)]
                    merge_count += 1 if base_key != name else 0
                    break

            if base_key not in merged:
                merged[base_key] = {
                    'workload_type': 'DaemonSet',
                    'by_node': {},
                    'app': metrics_data.get('app'),
                    'pods': []
                }

            target = merged[base_key]
            target['pods'].extend(metrics_data.get('pods', []) or [])

            # Merge per-node metrics
            for node_name, node_data in (metrics_data.get('by_node', {}) or {}).items():
                if node_name not in target['by_node']:
                    target['by_node'][node_name] = {
                        'cpu_usage': 0.0, 'ram_usage_gb': 0.0, 'pod_count': 0,
                        'exec_time': 0.0, 'execution_rate_hz': 0.0
                    }
                
                target_node = target['by_node'][node_name]
                target_node['cpu_usage'] += float(node_data.get('cpu_usage', 0.0))
                target_node['ram_usage_gb'] += float(node_data.get('ram_usage_gb', 0.0))
                target_node['pod_count'] += int(node_data.get('pod_count', 0))

        return merged

    def _merge_multi_deployment_variants(self, metrics: Dict[str, Dict]) -> Dict[str, Dict]:
        """Merge multi-deployment variants (eth0, wifi, etc.) under base service."""
        
        common_suffixes = {'eth0', 'eth1', 'wifi', 'main', 'gpu', 'cpu', 'primary', 'secondary'}
        merged = {}
        variant_map = {}

        # Identify variants
        for workload_name in metrics.keys():
            for suffix in common_suffixes:
                if workload_name.endswith('-' + suffix):
                    base_name = workload_name[: -(len(suffix) + 1)]
                    variant_map[workload_name] = base_name
                    break

        # Merge variants into base
        processed = set()
        for workload_name, metrics_data in metrics.items():
            if workload_name in processed:
                continue

            if workload_name in variant_map:
                base_name = variant_map[workload_name]
                processed.add(workload_name)

                if base_name not in merged:
                    merged[base_name] = metrics.get(base_name, {
                        'workload_type': metrics_data.get('workload_type', 'Deployment'),
                        'by_node': {}, 'app': metrics_data.get('app'), 'pods': []
                    })

                target = merged[base_name]
                target['pods'].extend(metrics_data.get('pods', []) or [])

                # Merge per-node metrics
                for node_name, node_data in (metrics_data.get('by_node', {}) or {}).items():
                    if node_name not in target['by_node']:
                        target['by_node'][node_name] = {
                            'cpu_usage': 0.0, 'ram_usage_gb': 0.0, 'pod_count': 0,
                            'exec_time': 0.0, 'execution_rate_hz': 0.0
                        }
                    
                    target_node = target['by_node'][node_name]
                    target_node['cpu_usage'] += float(node_data.get('cpu_usage', 0.0))
                    target_node['ram_usage_gb'] += float(node_data.get('ram_usage_gb', 0.0))
                    target_node['pod_count'] += int(node_data.get('pod_count', 0))
            else:
                if workload_name not in merged:
                    merged[workload_name] = metrics_data
                    processed.add(workload_name)

        return merged

    def _collect_execution_metrics(self, metrics: Dict, namespace: str, workload_pods: Dict):
        """Collect execution time and rate metrics from Prometheus."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        try:
            # Query execution time and rate metrics
            exec_time_query = f'blazeaiot_execution_time_us{{namespace="{namespace}"}}'
            exec_time_results = self._query_with_cluster_routing(exec_time_query)
            if not exec_time_results:
                exec_time_results = self._query_with_cluster_routing('blazeaiot_execution_time_us')

            exec_rate_query = f'blazeaiot_execution_rate_hz{{namespace="{namespace}"}}'
            exec_rate_results = self._query_with_cluster_routing(exec_rate_query)
            if not exec_rate_results:
                exec_rate_results = self._query_with_cluster_routing('blazeaiot_execution_rate_hz')

            if not exec_time_results:
                return

            # Group metrics by service
            service_exec_times = {}
            service_exec_rates = {}

            for result in exec_time_results:
                metric = result.get('metric', {})
                service_name = (metric.get('serviceName') or metric.get('service') 
                               or metric.get('app') or metric.get('service_id'))
                node_name = metric.get('nodeName') or metric.get('node')
                
                
                if not service_name:
                    continue
                
                service_key = f"{service_name}@{node_name}" if node_name else service_name
                exec_time_us = float(result.get('value', [0, '0'])[1])
                
                if service_key not in service_exec_times:
                    service_exec_times[service_key] = {
                        'exec_time_us': 0.0, 'service_name': service_name, 'node_name': node_name
                    }
                service_exec_times[service_key]['exec_time_us'] += exec_time_us

            for result in exec_rate_results or []:
                metric = result.get('metric', {})
                service_name = (metric.get('serviceName') or metric.get('service')
                               or metric.get('app') or metric.get('service_id'))
                node_name = metric.get('nodeName') or metric.get('node')
                
                if not service_name:
                    continue
                
                service_key = f"{service_name}@{node_name}" if node_name else service_name
                service_exec_rates[service_key] = float(result.get('value', [0, '0'])[1])

            # Update metrics with execution data
            updated_count = 0
            for workload_name, workload_metrics in metrics.items():
                by_node = workload_metrics.get('by_node', {}) or {}
                matched_any = False

                # Prefer matching to existing nodes (from pod discovery)
                for node_name, node_data in by_node.items():
                    matched = False
                    for service_key, exec_data in service_exec_times.items():
                        if self._matches_service_instance(workload_name, node_name, exec_data, service_key):
                            node_data['exec_time'] = exec_data['exec_time_us'] / 1_000_000.0
                            if service_key in service_exec_rates:
                                node_data['execution_rate_hz'] = service_exec_rates[service_key]
                            updated_count += 1
                            matched = True
                            matched_any = True
                            break

                # Fallback: if no nodes or no matches, attach exec metrics by service name
                if not by_node or not matched_any:
                    for service_key, exec_data in service_exec_times.items():
                        exec_node = exec_data.get('node_name')
                        if not exec_node:
                            continue
                        if self._matches_service_instance(workload_name, exec_node, exec_data, service_key):
                            node_data = by_node.get(exec_node)
                            if node_data is None:
                                node_data = {
                                    'cpu_usage': 0.0,
                                    'ram_usage_gb': 0.0,
                                    'pod_count': 0,
                                    'exec_time': 0.0,
                                    'execution_rate_hz': 0.0
                                }
                                by_node[exec_node] = node_data
                            node_data['exec_time'] = exec_data['exec_time_us'] / 1_000_000.0
                            if service_key in service_exec_rates:
                                node_data['execution_rate_hz'] = service_exec_rates[service_key]
                            updated_count += 1
                            matched_any = True
                            break

                workload_metrics['by_node'] = by_node

        except Exception as e:
            BlazeLogger.get_instance().warning(f"Failed to collect execution metrics: {e}")

    def _matches_service_instance(self, workload_name: str, node_name: str, 
                                 exec_data: Dict, service_key: str) -> bool:
        """Check if service instance matches execution metrics data."""
        def _norm(s: str) -> str:
            s = (s or '').lower().replace('-', '').replace('_', '')
            return ''.join(c for c in s if c.isalnum())

        def _infer_layer_prefix(identifier: str | None) -> str | None:
            if not identifier or '-' not in identifier:
                return None
            prefix = identifier.split('-', 1)[0].lower()
            if prefix == 'frank':
                return 'edge'
            return prefix

        def _extract_index(identifier: str | None) -> int | None:
            if not identifier:
                return None
            import re
            m = re.search(r"(\d+)$", identifier)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None

        s1 = _norm(workload_name)
        s2 = _norm(exec_data.get('service_name', ''))
        name_match = s1 == s2 or s1 in s2 or s2 in s1
        
        if not name_match:
            return False

        # Check node match if both have node info
        if node_name and exec_data.get('node_name'):
            node1 = node_name.split('.')[0].lower()
            node2 = exec_data['node_name'].split('.')[0].lower()
            if node1 == node2 or node1 in node2 or node2 in node1:
                return True

            # Fallback: match by inferred layer prefix + numeric index
            node1_prefix = _infer_layer_prefix(node1)
            node2_prefix = _infer_layer_prefix(node2)
            node1_idx = _extract_index(node1)
            node2_idx = _extract_index(node2)

            if node1_idx is not None and node2_idx is not None:
                if node1_prefix and node2_prefix:
                    return node1_prefix == node2_prefix and node1_idx == node2_idx
                return node1_idx == node2_idx

            return False

        return True

    def _update_snapshot(self, metrics: Dict, namespace: str):
        """Update cached discovery snapshot to merged metrics shape."""
        try:
            if metrics:
                self.last_discovery_snapshot = {
                    'namespace': namespace,
                    'workloads': {
                        name: {
                            'pods': data.get('pods', []),
                            'by_node': data.get('by_node', {}),
                            'workload_type': data.get('workload_type')
                        }
                        for name, data in metrics.items()
                    },
                    'source': 'merged_metrics'
                }
        except Exception:
            pass
