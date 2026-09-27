"""
Node metrics collector for gathering CPU, RAM, GPU, and layer information from Prometheus and Kubernetes.
"""

from typing import Dict, Optional



class NodeMetricsCollector:
    """Collects node resource metrics from Prometheus and Kubernetes."""

    def __init__(self, query_client):
        """
        Initialize the node metrics collector.

        Args:
            query_client: PrometheusQueryClient instance for executing queries
        """
        self.query_client = query_client

    def collect(self) -> Dict[str, Dict[str, float]]:
        """
        Get node resource metrics from Prometheus.

        Returns:
            Dictionary mapping node names to their metrics (cpu_cores, ram_gb, gpu_count, layer, cpu_freq_hz)
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        metrics = {}

        # Build a list of Prometheus sources to query. If multi-layer URLs are configured, hit all of them.
        prom_sources = []
        if getattr(self.query_client, 'cluster_urls', None):
            for layer_id, prom_url in self.query_client.cluster_urls.items():
                prom_sources.append((layer_id, prom_url))
        else:
            prom_sources.append((None, None))

        # Merge helper so later sources fill missing fields without clobbering existing data
        def _merge(node: str, key: str, value):
            if not node:
                return
            if node not in metrics:
                metrics[node] = {}
            if metrics[node].get(key) in (None, 0, ''):
                metrics[node][key] = value

        def _query(query: str, prom_url: Optional[str]):
            if prom_url:
                return self.query_client.query_at_url(query, prom_url)
            return self.query_client.query(query)

        # Global aggregates across sources
        cpu_capacity_by_instance: Dict[str, float] = {}
        mem_total_by_instance: Dict[str, float] = {}
        mem_avail_by_instance: Dict[str, float] = {}
        gpu_total_by_instance: Dict[str, float] = {}
        gpu_free_by_instance: Dict[str, float] = {}
        cpu_usage_by_instance: Dict[str, float] = {}
        instance_to_node: Dict[str, str] = {}
        instance_to_layer: Dict[str, str] = {}

        for layer_hint, prom_url in prom_sources:
            # Build a mapping from exporter instance (IP or host:port) to Kubernetes node name and layer when available
            kube_node_info_query = 'kube_node_info'
            try:
                kube_results = _query(kube_node_info_query, prom_url)
                if kube_results:
                    for result in kube_results:
                        inst = result['metric'].get('instance', '').split(':')[0]
                        node_label = result['metric'].get('node') or result['metric'].get('kubernetes_io_hostname')
                        if inst and node_label:
                            instance_to_node.setdefault(inst, node_label)
                            layer_label = result['metric'].get('blaze_layer') or result['metric'].get('layer')
                            if not layer_label and layer_hint:
                                layer_label = layer_hint
                            if layer_label:
                                instance_to_layer[inst] = layer_label.lower()
            except Exception:
                pass

            # Query for CPU capacity (number of cores)
            cpu_query = 'machine_cpu_cores'
            cpu_results = _query(cpu_query, prom_url)
            if cpu_results:
                for result in cpu_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    node = instance_to_node.get(inst, inst)
                    cores = float(result['value'][1])
                    cpu_capacity_by_instance[inst] = cores
                    _merge(node, 'cpu_cores', cores)
                    if inst in instance_to_layer:
                        _merge(node, 'layer', instance_to_layer[inst])

            # CPU usage fraction by instance (1 - idle)
            # Prefer irate for k3s/Rancher edge which matches Grafana dashboards; fallback to rate.
            cpu_idle_query_rate = 'avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)'
            cpu_idle_query_irate = 'avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) by (instance)'
            cpu_idle_results = _query(cpu_idle_query_irate, prom_url)
            if not cpu_idle_results:
                cpu_idle_results = _query(cpu_idle_query_rate, prom_url)
            if cpu_idle_results:
                for result in cpu_idle_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    try:
                        idle_frac = float(result['value'][1])
                        cpu_usage_by_instance[inst] = max(0.0, 1.0 - idle_frac)
                    except Exception:
                        continue

            # Query for CPU frequency (Hz) if available to derive speed
            freq_query = 'avg(node_cpu_frequency_hertz) by (instance)'
            freq_results = _query(freq_query, prom_url)
            if freq_results:
                for result in freq_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    node = instance_to_node.get(inst, inst)
                    freq_hz = float(result['value'][1])
                    _merge(node, 'cpu_freq_hz', freq_hz)

            # Query for memory capacity (in bytes, convert to GB)
            mem_query = 'machine_memory_bytes'
            mem_results = _query(mem_query, prom_url)
            if mem_results:
                for result in mem_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    node = instance_to_node.get(inst, inst)
                    memory_bytes = float(result['value'][1])
                    mem_total_by_instance[inst] = memory_bytes
                    memory_gb = memory_bytes / (1024 ** 3)
                    _merge(node, 'ram_gb', memory_gb)
                    if inst in instance_to_layer:
                        _merge(node, 'layer', instance_to_layer[inst])

            # Memory available per instance (bytes)
            mem_avail_query = 'node_memory_MemAvailable_bytes'
            mem_avail_results = _query(mem_avail_query, prom_url)
            if mem_avail_results:
                for result in mem_avail_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    try:
                        mem_avail_by_instance[inst] = float(result['value'][1])
                    except Exception:
                        continue

            # Query for GPU count (if available)
            gpu_query = 'count(DCGM_FI_DEV_GPU_UTIL) by (instance)'
            gpu_results = _query(gpu_query, prom_url)
            if gpu_results:
                for result in gpu_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    node = instance_to_node.get(inst, inst)
                    gpu_count = float(result['value'][1])
                    _merge(node, 'gpu_count', gpu_count)
                    if inst in instance_to_layer:
                        _merge(node, 'layer', instance_to_layer[inst])

            # GPU totals/free (bytes) for usage debug
            gpu_total_query = 'sum(node_nvsmi_memory_total_bytes) by (instance)'
            gpu_free_query = 'sum(node_nvsmi_memory_free_bytes) by (instance)'
            gpu_total_results = _query(gpu_total_query, prom_url)
            gpu_free_results = _query(gpu_free_query, prom_url)
            if gpu_total_results:
                for result in gpu_total_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    try:
                        gpu_total_by_instance[inst] = float(result['value'][1])
                    except Exception:
                        continue
            if gpu_free_results:
                for result in gpu_free_results:
                    inst = result['metric'].get('instance', '').split(':')[0]
                    try:
                        gpu_free_by_instance[inst] = float(result['value'][1])
                    except Exception:
                        continue

        # Always enrich with Kubernetes node capacities (fills gaps and fixes naming)
        k8s_metrics = self._get_k8s_node_metrics()
        # Build IP→node name map from Kubernetes addresses if available
        ip_to_node: Dict[str, str] = getattr(self, '_ip_to_node', {})
        added_from_k8s = 0
        enriched_from_k8s = 0
        if k8s_metrics:
            for node, kvals in k8s_metrics.items():
                if node not in metrics:
                    metrics[node] = kvals.copy()
                    added_from_k8s += 1
                    continue

                # Fill missing/zero fields from Kubernetes (cpu_cores, ram_gb, gpu_count, layer)
                for key, val in kvals.items():
                    if key not in metrics[node] or metrics[node].get(key) in (None, 0, ''):
                        metrics[node][key] = val
                        enriched_from_k8s += 1

        # Derive used values for debugging (capacity - available)
        # Consider every instance we observed (even if kube_node_info mapping is missing)
        all_instances = set(instance_to_node.keys())
        all_instances.update(cpu_capacity_by_instance.keys())
        all_instances.update(mem_total_by_instance.keys())
        all_instances.update(mem_avail_by_instance.keys())
        all_instances.update(gpu_total_by_instance.keys())
        all_instances.update(gpu_free_by_instance.keys())
        all_instances.update(cpu_usage_by_instance.keys())

        for inst in all_instances:
            node = instance_to_node.get(inst)
            if not node:
                # If inst is an IP:port or IP, strip port and map via Kubernetes IP→node
                ip = inst.split(':')[0]
                node = ip_to_node.get(ip, inst)
            if node not in metrics:
                metrics[node] = {}
            # CPU used (cores)
            cap_cpu = metrics[node].get('cpu_cores', cpu_capacity_by_instance.get(inst, 0.0))
            usage_frac = cpu_usage_by_instance.get(inst)
            if usage_frac is not None and cap_cpu > 0:
                metrics[node]['cpu_used'] = max(0.0, usage_frac * cap_cpu)

            # RAM used (GiB)
            total_bytes = mem_total_by_instance.get(inst)
            avail_bytes = mem_avail_by_instance.get(inst)
            if total_bytes is not None and avail_bytes is not None and total_bytes >= avail_bytes:
                used_gb = (total_bytes - avail_bytes) / (1024 ** 3)
                metrics[node]['ram_used_gb'] = max(0.0, used_gb)

            # GPU used (GiB)
            total_gpu_bytes = gpu_total_by_instance.get(inst)
            free_gpu_bytes = gpu_free_by_instance.get(inst)
            if total_gpu_bytes is not None and free_gpu_bytes is not None and total_gpu_bytes >= free_gpu_bytes:
                used_gpu_gb = (total_gpu_bytes - free_gpu_bytes) / (1024 ** 3)
                metrics[node]['gpu_used_gb'] = max(0.0, used_gpu_gb)

        # Fallback: derive used CPU/RAM per node from container aggregates when exporter usage is unavailable
        try:
            cpu_used_by_node = {}
            mem_used_by_node = {}
            # For environments lacking 'node' labels on container metrics, build pod->node map
            pod_to_node_global: Dict[str, str] = {}
            # Track node label key discovered from kube_pod_info per source (e.g., 'node', 'node_name')
            node_label_key_global: Optional[str] = None

            for layer_hint, prom_url in prom_sources:
                # Build pod->node mapping from kube_pod_info
                try:
                    pod_info_results = _query('kube_pod_info', prom_url)
                    if pod_info_results:
                        for r in pod_info_results:
                            pod = r['metric'].get('pod')
                            # Discover and cache the node label key used by this Prometheus
                            if node_label_key_global is None:
                                for candidate in ('node', 'node_name', 'exported_node', 'kubernetes_node', 'kubernetes_io_hostname', 'host'):
                                    if candidate in r['metric']:
                                        node_label_key_global = candidate
                                        break
                            node = None
                            if node_label_key_global:
                                node = r['metric'].get(node_label_key_global)
                            if not node:
                                node = r['metric'].get('node') or r['metric'].get('host') or r['metric'].get('kubernetes_io_hostname')
                            if pod and node:
                                pod_to_node_global.setdefault(pod, node)
                except Exception:
                    pass

                cpu_used_results = _query('sum(rate(container_cpu_usage_seconds_total{container!=""}[5m])) by (node)', prom_url)
                if cpu_used_results:
                    for result in cpu_used_results:
                        node = result['metric'].get('node')
                        if node:
                            try:
                                cpu_used_by_node[node] = float(result['value'][1])
                            except Exception:
                                pass

                mem_used_results = _query('sum(container_memory_working_set_bytes{container!=""}) by (node)', prom_url)
                if mem_used_results:
                    for result in mem_used_results:
                        node = result['metric'].get('node')
                        if node:
                            try:
                                mem_used_by_node[node] = float(result['value'][1]) / (1024 ** 3)
                            except Exception:
                                pass

                # Additional fallback for clusters that lack 'node' label: group by instance (kubelet IP) and map to node
                if not cpu_used_results:
                    # PromQL join: pod metrics joined with kube_pod_info to recover node label
                    node_lbl = node_label_key_global or 'node'
                    cpu_join_query = (
                        f"sum by ({node_lbl}) ("
                        'rate(container_cpu_usage_seconds_total{pod!="",container!=""}[5m])'
                        f") * on(pod) group_left({node_lbl}) ("
                        'kube_pod_info'
                        ")"
                    )
                    try:
                        cpu_join_results = _query(cpu_join_query, prom_url)
                        if cpu_join_results:
                            for result in cpu_join_results:
                                node = result['metric'].get(node_lbl) or result['metric'].get('node')
                                if node:
                                    try:
                                        cpu_used_by_node[node] = cpu_used_by_node.get(node, 0.0) + float(result['value'][1])
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    cpu_inst_results = _query('sum(rate(container_cpu_usage_seconds_total{pod!="",container!=""}[5m])) by (instance)', prom_url)
                    if cpu_inst_results:
                        for result in cpu_inst_results:
                            inst = result['metric'].get('instance', '').split(':')[0]
                            node = instance_to_node.get(inst)
                            if node:
                                try:
                                    cpu_used_by_node[node] = cpu_used_by_node.get(node, 0.0) + float(result['value'][1])
                                except Exception:
                                    pass
                    # Also aggregate by pod and translate via pod->node when instance mapping is insufficient
                    cpu_pod_results = _query('sum(rate(container_cpu_usage_seconds_total{pod!="",container!=""}[5m])) by (pod)', prom_url)
                    if cpu_pod_results:
                        for result in cpu_pod_results:
                            pod = result['metric'].get('pod')
                            node = pod_to_node_global.get(pod)
                            if node:
                                try:
                                    cpu_used_by_node[node] = cpu_used_by_node.get(node, 0.0) + float(result['value'][1])
                                except Exception:
                                    pass

                if not mem_used_results:
                    node_lbl = node_label_key_global or 'node'
                    mem_join_query = (
                        f"sum by ({node_lbl}) ("
                        'container_memory_working_set_bytes{pod!="",container!=""}'
                        f") * on(pod) group_left({node_lbl}) ("
                        'kube_pod_info'
                        ")"
                    )
                    try:
                        mem_join_results = _query(mem_join_query, prom_url)
                        if mem_join_results:
                            for result in mem_join_results:
                                node = result['metric'].get(node_lbl) or result['metric'].get('node')
                                if node:
                                    try:
                                        mem_used_by_node[node] = mem_used_by_node.get(node, 0.0) + float(result['value'][1]) / (1024 ** 3)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    mem_inst_results = _query('sum(container_memory_working_set_bytes{pod!="",container!=""}) by (instance)', prom_url)
                    if mem_inst_results:
                        for result in mem_inst_results:
                            inst = result['metric'].get('instance', '').split(':')[0]
                            node = instance_to_node.get(inst)
                            if node:
                                try:
                                    mem_used_by_node[node] = mem_used_by_node.get(node, 0.0) + float(result['value'][1]) / (1024 ** 3)
                                except Exception:
                                    pass
                    # Also aggregate memory by pod and translate via pod->node
                    mem_pod_results = _query('sum(container_memory_working_set_bytes{pod!="",container!=""}) by (pod)', prom_url)
                    if mem_pod_results:
                        for result in mem_pod_results:
                            pod = result['metric'].get('pod')
                            node = pod_to_node_global.get(pod)
                            if node:
                                try:
                                    mem_used_by_node[node] = mem_used_by_node.get(node, 0.0) + float(result['value'][1]) / (1024 ** 3)
                                except Exception:
                                    pass

            for node, cpu_used in cpu_used_by_node.items():
                if node not in metrics:
                    metrics[node] = {}
                # Only set when missing or zero
                if metrics[node].get('cpu_used', 0.0) <= 0.0:
                    metrics[node]['cpu_used'] = max(0.0, cpu_used)

            for node, ram_used in mem_used_by_node.items():
                if node not in metrics:
                    metrics[node] = {}
                if metrics[node].get('ram_used_gb', 0.0) <= 0.0:
                    metrics[node]['ram_used_gb'] = max(0.0, ram_used)
        except Exception:
            pass

        # Ensure debug fields exist even when usage metrics are missing so all layers get stamped
        for node in list(metrics.keys()):
            metrics[node].setdefault('cpu_used', 0.0)
            metrics[node].setdefault('ram_used_gb', 0.0)
            if metrics[node].get('gpu_count'):
                metrics[node].setdefault('gpu_used_gb', 0.0)
        
        return metrics

    def _get_k8s_node_metrics(self) -> Dict[str, Dict[str, float]]:
        """Fallback: pull node capacities directly from Kubernetes if available."""
        kmetrics: Dict[str, Dict[str, float]] = {}
        try:
            from kubernetes import client, config
        except ImportError:
            return kmetrics

        try:
            # Collect node info across all configured clusters (fog + edge)
            contexts = []
            mf = getattr(self.query_client, 'metrics_facade', None)
            if mf and getattr(mf, 'cluster_contexts', None):
                contexts = list(mf.cluster_contexts.values())

            # Fallback: single context from query_client
            if not contexts:
                contexts.append({
                    'kubeconfig_path': getattr(self.query_client, 'kubeconfig_path', None),
                    'context': getattr(self.query_client, 'kube_context', None),
                })

            self._ip_to_node = {}

            for ctx in contexts:
                kubeconfig_path = ctx.get('kubeconfig_path')
                kube_context = ctx.get('context')

                if kubeconfig_path and kubeconfig_path != 'null':
                    config.load_kube_config(config_file=kubeconfig_path, context=kube_context)
                elif kube_context:
                    config.load_kube_config(context=kube_context)
                else:
                    config.load_kube_config()

                v1 = client.CoreV1Api()
                nodes = v1.list_node(timeout_seconds=10)

                for n in nodes.items:
                    name = n.metadata.name
                    cap = n.status.capacity or {}
                    labels = n.metadata.labels or {}
                    layer_label = labels.get('blaze.layer') or labels.get('layer')
                    for addr in (n.status.addresses or []):
                        if addr.address:
                            self._ip_to_node[addr.address] = name
                    
                    # Check if node is schedulable
                    schedulable = True
                    if n.spec.unschedulable:
                        schedulable = False
                    taints = n.spec.taints or []
                    for taint in taints:
                        if taint.effect in ['NoSchedule', 'NoExecute']:
                            schedulable = False
                            break

                    cpu_str = cap.get('cpu', '0')
                    if cpu_str.endswith('m'):
                        cpu_val = int(cpu_str[:-1]) / 1000
                    else:
                        cpu_val = float(cpu_str)

                    mem_str = cap.get('memory', '0')
                    mem_gb = 0.0
                    if mem_str.endswith('Ki'):
                        mem_gb = int(mem_str[:-2]) / (1024 ** 2)
                    elif mem_str.endswith('Mi'):
                        mem_gb = int(mem_str[:-2]) / 1024
                    elif mem_str.endswith('Gi'):
                        mem_gb = float(mem_str[:-2])
                    elif mem_str.endswith('Ti'):
                        mem_gb = float(mem_str[:-2]) * 1024
                    else:
                        try:
                            mem_gb = int(mem_str) / (1024 ** 3)
                        except Exception:
                            mem_gb = 0.0

                    gpu_count = 0
                    for gpu_key in ['nvidia.com/gpu', 'amd.com/gpu', 'intel.com/gpu']:
                        if gpu_key in cap:
                            try:
                                gpu_count += int(cap[gpu_key])
                            except Exception:
                                pass

                    if name not in kmetrics:
                        kmetrics[name] = {
                            'cpu_cores': cpu_val,
                            'ram_gb': mem_gb,
                            'schedulable': schedulable,
                        }
                    else:
                        # Prefer non-zero fills if already present
                        kmetrics[name].setdefault('cpu_cores', cpu_val)
                        kmetrics[name].setdefault('ram_gb', mem_gb)
                        kmetrics[name].setdefault('schedulable', schedulable)

                    if layer_label:
                        kmetrics[name]['layer'] = layer_label.lower()
                    if gpu_count > 0:
                        kmetrics[name]['gpu_count'] = float(gpu_count)

            return kmetrics
        except Exception:
            return kmetrics
