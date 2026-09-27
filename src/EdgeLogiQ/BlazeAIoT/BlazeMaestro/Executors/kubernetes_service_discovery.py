"""
Kubernetes Service Discovery Module

Implements Single Responsibility Principle (SRP) for service discovery from Kubernetes.
Follows Open/Closed Principle (OCP) - extensible for different discovery strategies.
"""

from typing import Dict, List, Optional, Any
from BlazeAIoT.Core.Utils import BlazeLogger


class KubernetesServiceDiscovery:
    """
    Discovers services from Kubernetes clusters.
    
    Responsibilities:
    - Query Kubernetes API for workloads (Deployments, DaemonSets, StatefulSets)
    - Extract service metadata from labels and annotations
    - Map pods to nodes for allocation tracking
    """
    
    def __init__(self, clusters_config: List[Dict[str, Any]], services_config: Dict[str, Any]):
        """
        Initialize Kubernetes service discovery.
        
        Args:
            clusters_config: List of cluster configurations
            services_config: Services configuration (namespace, label keys, etc.)
        """
        self._logger = BlazeLogger.get_instance()
        self.clusters_config = clusters_config
        self.services_config = services_config
        self.namespace = services_config.get('namespace', 'default')
        self.label_keys = services_config.get('label_keys', [
            'blaze-service-id',
            'app',
            'app.kubernetes.io/name',
            'k8s-app',
        ])
        self.label_aliases = services_config.get('label_aliases', {})
        self.log_labels = bool(services_config.get('log_labels', False))
    
    def discover_current_allocation(self, service_ids: set) -> Dict[str, List[Dict[str, str]]]:
        """
        Discover current service allocation from Kubernetes.
        
        Returns a mapping: service_id -> [{'node': node_name, 'pod': pod_name, 'layer': layer_id}, ...]
        Only includes Ready pods (Running with Ready condition and all containers ready).
        If services_config.label_selector is provided without a value (e.g., just a key), we ignore it and query all pods.
        
        Args:
            service_ids: Set of known service IDs to match against
            
        Returns:
            Dict mapping service_id to list of pod placements
        """
        self._logger.debug(f"[K8s-Discovery] START - querying {self.namespace} namespace")
        
        try:
            from kubernetes import client, config
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            self._logger.warning("Kubernetes client not installed, skipping discovery")
            return {}
        
        allocation: Dict[str, List[Dict[str, str]]] = {}
        self._logger.debug(f"[K8s-Discovery] START - querying {self.namespace} namespace")
        self._logger.debug(f"[K8s-Discovery] Service IDs to match: {service_ids}")
        self._logger.debug(f"[K8s-Discovery] Clusters config: {len(self.clusters_config)} clusters")
        
        total_matched = 0
        for cluster in self.clusters_config:
            context = cluster.get('context')
            kubeconfig_path = cluster.get('kubeconfig_path')
            layer_id = cluster.get('layer_id', 'unknown')
            
            self._logger.debug(f"[K8s-Discovery] Processing cluster: name={cluster.get('name')}, context={context}, layer={layer_id}")
            
            try:
                # Load kubeconfig
                if kubeconfig_path and kubeconfig_path != 'null':
                    config.load_kube_config(config_file=kubeconfig_path, context=context)
                else:
                    config.load_kube_config(context=context)
                
                v1 = client.CoreV1Api()
                apps_v1 = client.AppsV1Api()
                
                # Discover pods
                # Only use label_selector if it includes a value (key=value); otherwise query all pods
                configured_selector = self.services_config.get('label_selector')
                label_selector = configured_selector if configured_selector and '=' in configured_selector else None

                if label_selector:
                    pods = v1.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
                    self._logger.debug(f"  [{context}] Pods with selector '{label_selector}': {len(pods.items)}")
                else:
                    pods = v1.list_namespaced_pod(namespace=self.namespace)
                    self._logger.debug(f"  [{context}] Pods (no selector): {len(pods.items)}")
                    # Debug: list robot-localization pods found
                    robot_loc_pods = [p for p in pods.items if any(v == 'robot-localization' for v in (p.metadata.labels or {}).values())]
                    if robot_loc_pods:
                        self._logger.debug(f"    Found {len(robot_loc_pods)} robot-localization pod(s):")
                        for pod in robot_loc_pods:
                            self._logger.debug(f"      - {pod.metadata.name} (phase={pod.status.phase}, node={pod.spec.node_name})")
                
                # Match pods to services
                matched_count = self._match_pods_to_services(pods, service_ids, layer_id, allocation)
                total_matched += matched_count
                
                if self.log_labels:
                    print(f"  [{context}] Matched {matched_count} pods to services")
                
                # Fallback: discover from workload names
                self._discover_from_workloads(apps_v1, service_ids, layer_id, allocation)
                
            except Exception as e:
                self._logger.error(f"Error querying cluster {context}: {e}")
                self._logger.debug(f"  [{context}] ERROR: {e}")
                continue
        
        self._logger.debug(f"[K8s-Discovery] COMPLETE - total matched: {total_matched} pods")
        self._logger.debug(f"[k8s-discovery] Allocation entries: {len(allocation)} (matched pods: {total_matched})")
        return allocation
    
    def _match_pods_to_services(
        self,
        pods: Any,
        service_ids: set,
        layer_id: str,
        allocation: Dict[str, List[Dict[str, str]]]
    ) -> int:
        """
        Match pods to services based on labels.
        
        Returns:
            Number of matched pods
        """
        matched_count = 0
        
        for pod in pods.items:
            labels = pod.metadata.labels or {}

            # Temporary targeted logging for robot-localization troubleshooting
            label_values = list(labels.values())
            is_robot_localization = any(v == 'robot-localization' for v in label_values)

            active = self._pod_is_active(pod)
            if not active:
                if is_robot_localization:
                    phase = getattr(pod.status, 'phase', '')
                    conditions = getattr(pod.status, 'conditions', None) or []
                    ready_cond = [f"{c.type}:{c.status}" for c in conditions]
                    cs = getattr(pod.status, 'container_statuses', None) or []
                    cs_ready = [getattr(c, 'ready', False) for c in cs]
                    print(f"    ✗ robot-localization pod {pod.metadata.name} not active (phase={phase}, ready_conditions={ready_cond}, containers_ready={cs_ready})")
                continue
            
            node_name = pod.spec.node_name
            pod_name = pod.metadata.name
            
            # Try to match pod to service
            matched_service_id = self._match_pod_labels(labels, service_ids)
            
            if is_robot_localization:
                print(f"    [robot-localization] pod {pod_name}: active=True, matched_to={matched_service_id}, labels={labels}, service_ids={service_ids}")
            
            if matched_service_id:
                if matched_service_id not in allocation:
                    allocation[matched_service_id] = []
                
                allocation[matched_service_id].append({
                    'node': node_name,
                    'pod': pod_name,
                    'start_time': pod.status.start_time,
                    'layer': layer_id
                })
                matched_count += 1
                
                if self.log_labels or matched_service_id == 'robot-localization':
                    print(f"    ✓ Matched pod {pod_name} → service '{matched_service_id}' (node={node_name}, layer={layer_id})")
            else:
                if self.log_labels:
                    candidates = [labels.get(k) for k in self.label_keys if labels.get(k)]
                    print(f"    ✗ No match for pod {pod_name} (candidates={candidates})")
        
        return matched_count
    
    def _match_pod_labels(self, labels: Dict[str, str], service_ids: set) -> Optional[str]:
        """
        Match pod labels to a service ID.
        
        Args:
            labels: Pod labels
            service_ids: Set of known service IDs
            
        Returns:
            Matched service ID or None
        """
        # Try exact match on label keys
        candidates = [labels.get(k) for k in self.label_keys if labels.get(k)]
        for cand in candidates:
            if cand in service_ids:
                return cand

        # Try app name and instance variations
        app_name = labels.get('app.kubernetes.io/name')
        app_instance = labels.get('app.kubernetes.io/instance')
        for val in (app_name, app_instance):
            if val:
                # Exact match
                if val in service_ids:
                    return val
                # Substring match (for cases like robot-localization-xyz)
                for sid in service_ids:
                    if sid in val:
                        return sid

        # Try app label
        app_label = labels.get('app')
        if app_label:
            if app_label in service_ids:
                return app_label

        # Try k8s-app label
        k8s_app_label = labels.get('k8s-app')
        if k8s_app_label:
            if k8s_app_label in service_ids:
                return k8s_app_label

        # Try aliases
        if self.label_aliases:
            for label_key, alias_map in self.label_aliases.items():
                label_value = labels.get(label_key)
                if label_value and label_value in alias_map:
                    aliased_sid = alias_map[label_value]
                    if aliased_sid in service_ids:
                        return aliased_sid

        return None
    
    def _pod_is_active(self, pod: Any) -> bool:
        """Check if pod is active and Ready (all containers ready).

        Criteria:
        - Running AND Ready condition True AND all container_statuses.ready True
        - Pending pods are excluded from active allocation mapping
        """
        try:
            phase = (pod.status.phase or '').lower()
            if phase != 'running':
                return False

            # Ready condition
            conditions = getattr(pod.status, 'conditions', None) or []
            has_ready_condition = any(
                (getattr(c, 'type', '') == 'Ready' and str(getattr(c, 'status', '')).lower() == 'true')
                for c in conditions
            )

            # All containers ready
            container_statuses = getattr(pod.status, 'container_statuses', None) or []
            containers_ready = len(container_statuses) > 0 and all(
                getattr(cs, 'ready', False) for cs in container_statuses
            )

            return bool(has_ready_condition and containers_ready)

        except Exception:
            # If any field is missing, treat as not active
            return False
    
    def _discover_from_workloads(
        self,
        apps_v1: Any,
        service_ids: set,
        layer_id: str,
        allocation: Dict[str, List[Dict[str, str]]]
    ) -> None:
        """Fallback: discover from deployment/daemonset names."""
        try:
            # Deployments
            deployments = apps_v1.list_namespaced_deployment(namespace=self.namespace)
            for dep in deployments.items:
                name = dep.metadata.name
                matched_sid = self._match_workload_name(name, service_ids)
                if matched_sid and matched_sid not in allocation:
                    allocation[matched_sid] = [{'node': '', 'pod': name, 'layer': layer_id}]
                    if self.log_labels:
                        print(f"    • Fallback match deployment {name} → service '{matched_sid}'")
            
            # DaemonSets
            daemonsets = apps_v1.list_namespaced_daemon_set(namespace=self.namespace)
            for ds in daemonsets.items:
                name = ds.metadata.name
                matched_sid = self._match_workload_name(name, service_ids)
                if matched_sid and matched_sid not in allocation:
                    allocation[matched_sid] = [{'node': '', 'pod': name, 'layer': layer_id}]
                    if self.log_labels:
                        print(f"    • Fallback match daemonset {name} → service '{matched_sid}'")
        
        except Exception as e:
            self._logger.debug(f"Could not discover from workloads: {e}")
    
    def _match_workload_name(self, name: str, service_ids: set) -> Optional[str]:
        """Match workload name to service ID."""
        for sid in service_ids:
            if name == sid or name.startswith(sid):
                return sid
        return None
    
    def build_pod_node_mapping(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Build mapping of service -> {pod_name -> {node, layer}} from actual Kubernetes workloads.
        
        Returns:
            Dict mapping service_id to dict of pod_name -> {'node': node_id, 'layer': layer_id}
        """
        try:
            from kubernetes import client, config
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            self._logger.debug("Kubernetes client not available, skipping pod discovery")
            return {}
        
        pod_node_mapping = {}
        
        print(f"[K8s Discovery] Starting discovery in namespace: {self.namespace}")
        print(f"[K8s Discovery] Clusters config: {len(self.clusters_config)} clusters")
        
        for cluster in self.clusters_config:
            context = cluster.get('context')
            kubeconfig_path = cluster.get('kubeconfig_path')
            cluster_name = cluster.get('name', context)
            layer_id = cluster.get('layer_id', 'unknown')
            
            print(f"[K8s Discovery] Processing cluster={cluster_name}, context={context}, layer={layer_id}")
            
            try:
                # Load kubeconfig
                if kubeconfig_path and kubeconfig_path != 'null':
                    config.load_kube_config(config_file=kubeconfig_path, context=context)
                else:
                    config.load_kube_config(context=context)
                
                v1 = client.CoreV1Api()
                apps_v1 = client.AppsV1Api()
                
                # Discover DaemonSets and their pods
                self._discover_daemonset_pods(apps_v1, v1, cluster_name, layer_id, pod_node_mapping)
                
                # Discover Deployments and their pods
                self._discover_deployment_pods(apps_v1, v1, cluster_name, layer_id, pod_node_mapping)
                
            except Exception as e:
                self._logger.warning(f"Could not query cluster {cluster_name}: {e}")
                print(f"[K8s Discovery] ERROR on {cluster_name}: {e}")
                continue
        
        if pod_node_mapping:
            total_pods = sum(len(pods) for pods in pod_node_mapping.values())
            self._logger.debug(f"✓ Discovered {total_pods} running pods across {len(pod_node_mapping)} services")
        else:
            print(f"[K8s Discovery] WARNING: No pods discovered in any cluster")
        
        return pod_node_mapping
    
    def _discover_daemonset_pods(
        self,
        apps_v1: Any,
        v1: Any,
        cluster_name: str,
        layer_id: str,
        pod_node_mapping: Dict[str, Dict[str, Dict[str, str]]]
    ) -> None:
        """Discover DaemonSet pods."""
        try:
            daemonsets = apps_v1.list_namespaced_daemon_set(namespace=self.namespace)
            print(f"  [{cluster_name}] Found {len(daemonsets.items)} DaemonSets in {self.namespace}")
            
            for ds in daemonsets.items:
                workload_name = ds.metadata.name
                labels = ds.metadata.labels or {}
                service_id = self._extract_service_id(labels) or workload_name
                
                # Get pods for this DaemonSet
                label_selector = ','.join([f"{k}={v}" for k, v in ds.spec.selector.match_labels.items()])
                pods = v1.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
                
                # Debug: log pods found for this daemonset
                if pods.items:
                    is_robot_loc = 'robot-localization' in workload_name
                    if is_robot_loc:
                        print(f"    [Discovery] DaemonSet {workload_name} has {len(pods.items)} pods")
                
                for pod in pods.items:
                    is_robot_loc = 'robot-localization' in workload_name
                    
                    active = self._pod_is_active(pod)
                    if is_robot_loc:
                        phase = getattr(pod.status, 'phase', '')
                        conditions = getattr(pod.status, 'conditions', None) or []
                        ready_cond = [(c.type, c.status) for c in conditions]
                        cs = getattr(pod.status, 'container_statuses', None) or []
                        cs_ready = [getattr(c, 'ready', False) for c in cs]
                        print(f'    [robot-localization] pod {pod.metadata.name}: active={active}, phase={phase}, ready_conditions={ready_cond}, containers_ready={cs_ready}')
                    
                    if active:
                        pod_name = pod.metadata.name
                        node_name = pod.spec.node_name or 'unknown'
                        
                        if service_id not in pod_node_mapping:
                            pod_node_mapping[service_id] = {}
                        
                        pod_node_mapping[service_id][pod_name] = {
                            'node': node_name,
                            'layer': layer_id
                        }
                        
                        self._logger.debug(f"  [{cluster_name}] Pod {pod_name} ({workload_name}->{service_id}) on {node_name} ({layer_id})")
        
        except Exception as e:
            self._logger.debug(f"Could not discover daemonsets in {cluster_name}: {e}")
    
    def _discover_deployment_pods(
        self,
        apps_v1: Any,
        v1: Any,
        cluster_name: str,
        layer_id: str,
        pod_node_mapping: Dict[str, Dict[str, Dict[str, str]]]
    ) -> None:
        """Discover Deployment pods."""
        try:
            deployments = apps_v1.list_namespaced_deployment(namespace=self.namespace)
            print(f"  [{cluster_name}] Found {len(deployments.items)} Deployments in namespace '{self.namespace}':")
            if deployments.items:
                for d in deployments.items:
                    print(f"    - {d.metadata.name}")
            
            for deploy in deployments.items:
                workload_name = deploy.metadata.name
                labels = deploy.metadata.labels or {}
                service_id = self._extract_service_id(labels) or workload_name
                
                # Get pods for this Deployment
                label_selector = ','.join([f"{k}={v}" for k, v in deploy.spec.selector.match_labels.items()])
                pods = v1.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
                
                # Debug: for robot-localization, print ALL pods found, active or not
                if 'robot-localization' in workload_name:
                    print(f"    [robot-localization] Deployment found, selector='{label_selector}', pods found={len(pods.items)}")
                    for pod in pods.items:
                        phase = getattr(pod.status, 'phase', '')
                        conditions = getattr(pod.status, 'conditions', None) or []
                        ready_cond = {c.type: c.status for c in conditions}
                        cs = getattr(pod.status, 'container_statuses', None) or []
                        cs_ready = [f"{c.name}:{c.ready}" for c in cs]
                        node_name = pod.spec.node_name or 'unscheduled'
                        pod_labels = pod.metadata.labels or {}
                        extracted_id = self._extract_service_id(pod_labels)
                        print(f"      pod {pod.metadata.name}: phase={phase}, node={node_name}, ready_cond={ready_cond}, containers={cs_ready}")
                        print(f"        labels={pod_labels}")
                        print(f"        extracted_service_id={extracted_id}, deployment_service_id={service_id}")
                
                for pod in pods.items:
                    is_robot_loc = 'robot-localization' in workload_name
                    
                    active = self._pod_is_active(pod)
                    
                    if active:
                        pod_name = pod.metadata.name
                        node_name = pod.spec.node_name or 'unknown'
                        
                        if service_id not in pod_node_mapping:
                            pod_node_mapping[service_id] = {}
                        
                        pod_node_mapping[service_id][pod_name] = {
                            'node': node_name,
                            'layer': layer_id
                        }
                        
                        self._logger.debug(f"  [{cluster_name}] Pod {pod_name} ({workload_name}->{service_id}) on {node_name} ({layer_id})")
        
        except Exception as e:
            self._logger.debug(f"Could not discover deployments in {cluster_name}: {e}")
            print(f"  [{cluster_name}] ERROR listing deployments: {e}")
    
    def _extract_service_id(self, metadata_labels: Dict[str, str]) -> Optional[str]:
        """Extract service ID from workload labels using priority order."""
        if not metadata_labels:
            return None
        
        for label_key in self.label_keys:
            if label_key in metadata_labels:
                return metadata_labels[label_key]
        
        return None
