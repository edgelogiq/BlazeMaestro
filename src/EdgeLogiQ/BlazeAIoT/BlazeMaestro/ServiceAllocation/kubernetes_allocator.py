"""
Kubernetes service allocator implementation.

Uses the Kubernetes Python client to manipulate Deployment replicas for
service allocation.

Allocation flow (same-cluster):
1. Read current replica count and nodeSelector
2. Patch deployment with nodeAffinity to pin replica to target_node
3. Monitor pod status until Running and containers Ready
4. Scale source deployment to 0 (if provided)

Allocation flow (cross-cluster or no source):
1. Scale target deployment to 1 replica
2. Monitor pod status until Running and containers Ready
3. Scale source deployment to 0 (if provided)
"""

from typing import Optional, Tuple
from contextlib import suppress
from .base_allocator import BaseAllocator

from BlazeAIoT.Core.Utils import BlazeLogger


class KubernetesAllocator(BaseAllocator):
    """
    Kubernetes implementation of service allocator.
    
    Uses Kubernetes API to update Deployment replica counts.
    Supports multi-cluster configuration - routes allocation requests
    to the appropriate cluster based on node/layer mapping.
    """
    
    def __init__(
        self,
        cluster_configs: Optional[list] = None,
        nodes: Optional[list] = None,
        verbose: bool = False,
        replica_check_timeout: float = 60.0,
        confirmation_timeout: float = 300.0
    ):
        """
        Initialize Kubernetes allocator with multi-cluster support.
        
        Args:
            cluster_configs: List of cluster config dicts with keys:
                - kubeconfig_path: Path to kubeconfig
                - context: Kubernetes context
                - layer_id: Layer identifier
                - name: Cluster name
            nodes: List of node objects (used to build node->cluster mapping)
            verbose: Enable verbose logging
            replica_check_timeout: Timeout (seconds) for waiting on replica readiness
            confirmation_timeout: Max seconds to wait for pod readiness
        """
        super().__init__(service_manager_broker=None, verbose=verbose)
        self.cluster_configs = cluster_configs or []
        self.replica_check_timeout = replica_check_timeout
        self.confirmation_timeout = confirmation_timeout
        self._logger = BlazeLogger.get_instance()
        
        # Build node_id -> cluster_config mapping
        self.node_to_cluster = {}
        if nodes and cluster_configs:
            for node in nodes:
                node_id = getattr(node, 'node_id', '')
                layer_id = getattr(node, 'layer_id', None)
                if not layer_id and '-' in node_id:
                    layer_id = node_id.split('-')[0]
                # Find matching cluster by layer_id
                for cluster in cluster_configs:
                    cluster_layer = cluster.get('layer_id')
                    if cluster_layer == layer_id:
                        self.node_to_cluster[node_id] = cluster
                        print(f"      ✓ Mapped {node_id} to {cluster.get('name')}")
                        break
        
        # Cache for AppsV1Api clients per cluster context
        self._apps_api_cache = {}
        # Cache for CoreV1Api clients per cluster context
        self._core_api_cache = {}

        # Try to import Kubernetes client lazily
        self._k8s_client = None
        self._k8s_config = None
        with suppress(Exception):
            from kubernetes import client as _client, config as _config  # type: ignore
            self._k8s_client = _client
            self._k8s_config = _config
    
    def allocate(
        self,
        service_id: str,
        target_node: str,
        target_layer: str,
        source_node: Optional[str] = None,
        source_layer: Optional[str] = None,
        namespace: str = "default",
        deployment_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Allocate service to target node.
        
        Same-cluster flow (source_deployment_name == deployment_name):
        1. Read current replicas & nodeSelector
        2. Scale to current+1 and add nodeAffinity for target_node
        3. Wait until pod is Ready on target_node (all containers ready)
        4. Scale to 1 and update nodeSelector to target_node only
        
        Cross-cluster flow (different deployment_name or no source):
        1. Scale target deployment to 1 replica
        2. Wait until pod is Ready on target_node (all containers ready)
        3. Scale source deployment to 0
        
        Args:
            service_id: Service ID
            target_node: Target Kubernetes node name
            target_layer: Target layer
            source_node: Source node (for logging/metadata)
            source_layer: Source layer
            namespace: Kubernetes namespace (default: "default")
            deployment_name: Deployment name (defaults to service_id)
            **kwargs: 
                - source_deployment_name: Source deployment (if different from deployment_name)
                - same_cluster: Force same-cluster logic (default: auto-detect)
                - confirmation_timeout: Override default timeout
        
        Returns:
            Tuple[bool, Optional[str]]: (success status, failure reason if failed)
        """
        
        deployment_name = deployment_name or service_id
        source_deployment_name = kwargs.get('source_deployment_name', deployment_name)
        
        # Determine same_cluster based on node-to-cluster mapping
        # Query the platform to find which cluster each node belongs to
        if 'same_cluster' in kwargs:
            same_cluster = kwargs.get('same_cluster')
        elif source_node is None:
            # Initial placement (no source) must use cross-cluster flow so target
            # deployment is explicitly scaled up to 1 replica.
            same_cluster = False
        else:
            # Default: same cluster if nodes belong to same cluster in platform
            target_cluster = self.node_to_cluster.get(target_node)
            source_cluster = self.node_to_cluster.get(source_node) if source_node else None
            
            # Same cluster if:
            # 1. Both nodes map to same cluster config, OR
            # 2. Same deployment AND same layer (fallback)
            if target_cluster and source_cluster:
                same_cluster = (target_cluster.get('context') == source_cluster.get('context'))
            elif target_cluster and source_node:
                # If we can't find source cluster, assume different
                same_cluster = False
            else:
                # No cluster info available - use layer check
                same_cluster = (deployment_name == source_deployment_name) and (source_layer == target_layer)
        
        confirmation_timeout: float = float(
            kwargs.get('confirmation_timeout', self.confirmation_timeout)
        )

        try:
            if same_cluster:
                # Same-cluster: scale up, add affinity, wait, scale down with new selector
                return self._allocate_same_cluster(
                    deployment_name=deployment_name,
                    namespace=namespace,
                    target_node=target_node,
                    service_id=service_id,
                    target_layer=target_layer,
                    confirmation_timeout=confirmation_timeout
                )
            else:
                # Cross-cluster or different deployments
                return self._allocate_cross_cluster(
                    deployment_name=deployment_name,
                    source_deployment_name=source_deployment_name,
                    namespace=namespace,
                    target_node=target_node,
                    source_node=source_node,
                    service_id=service_id,
                    target_layer=target_layer,
                    source_layer=source_layer,
                    confirmation_timeout=confirmation_timeout
                )
        
        except Exception as e:
            error_msg = str(e)
            return False, error_msg
    
    def _allocate_same_cluster(
        self,
        deployment_name: str,
        namespace: str,
        target_node: str,
        service_id: str,
        target_layer: str,
        confirmation_timeout: float
    ) -> Tuple[bool, Optional[str]]:
        """Handle same-cluster allocation (scale up, wait, scale down)."""
        import sys
        
        try:
            api = self._get_apps_api(target_node=target_node)
        except Exception as e:
            print(f"[KubernetesAllocator] Failed to get API: {e}")
            sys.stdout.flush()
            return False, f"Failed to get Kubernetes API client: {e}"
        
        # Step 1: Read current deployment
        try:
            dep = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        except Exception as e:
            print(f"[KubernetesAllocator] Step 1: Failed to read deployment: {e}")
            sys.stdout.flush()
            return False, f"Deployment '{deployment_name}' not found in namespace '{namespace}': {e}"
        current_replicas = dep.spec.replicas or 0
        
        # Step 2: Scale up to current+1 and add nodeAffinity for target_node
        # Step 2: Patch template with nodeAffinity but keep replicas at current level
        # Let rolling update handle the pod replacement naturally
        # maxSurge=1 allows up to 2 pods temporarily (1 old + 1 new)
        # maxUnavailable=0 ensures new pod starts before old one terminates
        
        # Build node affinity to prefer target_node for new replica
        node_affinity = {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{
                    "matchExpressions": [{
                        "key": "kubernetes.io/hostname",
                        "operator": "In",
                        "values": [target_node]
                    }]
                }]
            }
        }
        
        # Keep replicas at current level - let rolling update handle replacement
        patch_body = {
            "spec": {
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": 1,
                        "maxUnavailable": 0
                    }
                },
                "template": {
                    "spec": {
                        "nodeName": None,  # Clear hard node constraint
                        "nodeSelector": {
                            "kubernetes.io/hostname": target_node
                        },
                        "affinity": {
                            "nodeAffinity": node_affinity
                        }
                    }
                }
            }
        }
        
        try:
            api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=patch_body
            )
        except Exception as e:
            print(f"[KubernetesAllocator] Step 2: Patch failed: {e}")
            sys.stdout.flush()
            return False, f"Failed to patch deployment template: {e}"
        
        # Verify the patch was applied
        try:
            import time
            time.sleep(1)  # Give Kubernetes a moment to process
            updated_dep = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            # Replicas should stay at current level during rolling update
        except Exception as e:
            self._logger.warning(f"Could not verify template patch: {e}")
        
        # Step 3: Wait for pod to be Ready on target_node
        try:
            import time
            v1 = self._get_core_api(target_node=target_node)
            
            # First, find what labels the deployment's pods actually have
            # List all pods in namespace and find ones matching deployment
            all_pods = v1.list_namespaced_pod(namespace=namespace)
            
            # Find pods for this service by checking all pods
            service_pods = []
            for pod in all_pods.items:
                pod_labels = pod.metadata.labels or {}
                # Check if pod belongs to this deployment
                if pod_labels.get('app.kubernetes.io/name') == service_id or \
                   pod_labels.get('app') == service_id or \
                   pod_labels.get('app.kubernetes.io/instance') == service_id:
                    service_pods.append(pod)
            
            if not service_pods:
                try:
                    pods = v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"app.kubernetes.io/name={service_id}"
                    )
                    service_pods = pods.items
                except Exception as e:
                    print(f"[KubernetesAllocator] Label selector failed: {e}")
            
            start_time = time.time()
            pod_ready = False
            loop_iterations = 0
            
            sys.stdout.flush()
            
            while time.time() - start_time < confirmation_timeout:
                loop_iterations += 1
                elapsed = time.time() - start_time
                
                try:
                    # Re-query pods each time to check status
                    all_pods = v1.list_namespaced_pod(namespace=namespace)
                    
                    # Find pods for this service
                    for pod in all_pods.items:
                        pod_labels = pod.metadata.labels or {}
                        # Check if pod belongs to this deployment
                        if pod_labels.get('app.kubernetes.io/name') == service_id or \
                           pod_labels.get('app') == service_id or \
                           pod_labels.get('app.kubernetes.io/instance') == service_id:
                            
                            if pod.spec.node_name == target_node:
                                # Check pod Ready condition and all containers ready
                                conditions = pod.status.conditions or []
                                has_ready_condition = any(
                                    (c.type == "Ready" and str(c.status).lower() == "true") for c in conditions
                                )
                                container_statuses = pod.status.container_statuses or []
                                containers_ready = (
                                    len(container_statuses) > 0 and all(cs.ready for cs in container_statuses)
                                )
                                if has_ready_condition and containers_ready:
                                    pod_ready = True
                                    break
                except Exception:
                    # Continue polling until timeout
                    pass
                
                if pod_ready:
                    break
                
                time.sleep(2)  # Check every 2 seconds
            
            if not pod_ready:
                elapsed = time.time() - start_time
                return False, f"Pod failed to become Ready on {target_node} within {elapsed:.1f}s (timeout={confirmation_timeout}s)"
        
        except Exception as e:
            return False, f"Error while waiting for pod readiness: {e}"
        
        return True, None
    
    def _allocate_cross_cluster(
        self,
        deployment_name: str,
        source_deployment_name: str,
        namespace: str,
        target_node: str,
        source_node: Optional[str],
        service_id: str,
        target_layer: str,
        source_layer: Optional[str],
        confirmation_timeout: float
    ) -> Tuple[bool, Optional[str]]:
        """Handle cross-cluster allocation (scale target, wait, scale source down)."""
        import sys
        
        # Step 1: Scale target deployment to 1 replica with nodeSelector
        try:
            api = self._get_apps_api(target_node=target_node)
        except Exception as e:
            print(f"[KubernetesAllocator] Failed to get API for target: {e}")
            sys.stdout.flush()
            return False, f"Failed to get Kubernetes API client: {e}"
        
        # Clear any existing affinity rules when using nodeSelector
        # IMPORTANT: Also remove nodeName to allow scheduler to place pod based on nodeSelector
        patch_body = {
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "nodeName": None,  # Remove hard node constraint to allow scheduling
                        "nodeSelector": {
                            "kubernetes.io/hostname": target_node
                        },
                        "affinity": None  # Remove any existing affinity
                    }
                }
            }
        }
        
        try:
            api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=patch_body
            )
        except Exception as e:
            return False, f"Failed to scale target deployment: {e}"
        
        # Verify the scale was applied
        try:
            import time
            time.sleep(1)  # Give Kubernetes a moment to process
            updated_dep = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
            actual_replicas = updated_dep.spec.replicas or 0
            if actual_replicas != 1:
                return False, f"Scale verification failed: expected 1 replica, got {actual_replicas}"
        except Exception as e:
            return False, f"Could not verify scale: {e}"
        
        
        # Step 2: Wait for pod to be Ready on target_node (cross-cluster readiness check)
        try:
            import time
            v1 = self._get_core_api(target_node=target_node)
            start_time = time.time()
            pod_ready = False
            wait_iterations = 0


            while time.time() - start_time < confirmation_timeout:
                wait_iterations += 1
                elapsed = time.time() - start_time

                all_pods = v1.list_namespaced_pod(namespace=namespace)
                for pod in all_pods.items:
                    labels = pod.metadata.labels or {}
                    if labels.get('app.kubernetes.io/name') == service_id or \
                       labels.get('app') == service_id or \
                       labels.get('app.kubernetes.io/instance') == service_id:
                        if pod.spec.node_name == target_node:
                            conditions = pod.status.conditions or []
                            has_ready_condition = any((c.type == "Ready" and str(c.status).lower() == "true") for c in conditions)
                            container_statuses = pod.status.container_statuses or []
                            containers_ready = len(container_statuses) > 0 and all(cs.ready for cs in container_statuses)
                            if has_ready_condition and containers_ready:
                                pod_ready = True
                                break
                if pod_ready:
                    break
                time.sleep(2)

            if not pod_ready:
                elapsed = time.time() - start_time
                return False, f"Pod failed to become Ready on {target_node} within {elapsed:.1f}s (timeout={confirmation_timeout}s)"

        except Exception as e:
            return False, f"Error while waiting for pod readiness (cross-cluster): {e}"

        # Step 3: Scale source deployment to 0 (if source provided and different cluster/node)
        if source_node and source_layer and (source_node != target_node or source_layer != target_layer or source_deployment_name != deployment_name):
            if not self._set_replica_count(source_deployment_name, namespace, 0, node_hint=source_node):
                BlazeLogger.get_instance().warning(f"Failed to scale down source deployment {source_deployment_name}")
                # Don't fail allocation if source scale-down fails
        
        return True, None
    
    def deallocate(
        self,
        service_id: str,
        node: str,
        layer: str,
        namespace: str = "default",
        deployment_name: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Deallocate service from a node by scaling deployment to 0.
        
        Args:
            service_id: Service ID
            node: Node to deallocate from
            layer: Layer
            namespace: Kubernetes namespace
            deployment_name: Deployment name (defaults to service_id)
            **kwargs: Additional arguments
        
        Returns:
            bool: True if successful, False otherwise
        """
        deployment_name = deployment_name or service_id
        try:
            # Scale deployment to 0
            if not self._set_replica_count(deployment_name, namespace, 0):
                raise RuntimeError(f"Failed to scale {deployment_name} to 0 replicas")
            
            return True
        
        except Exception as e:
            if self.verbose:
                print(f"[KubernetesAllocator] Deallocation failed: {e}")
            return False
    
    def get_replica_count(
        self,
        service_id: str,
        node: str,
        namespace: str = "default",
        deployment_name: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Get current replica count for deployment.
        
        Args:
            service_id: Service ID
            node: Node (unused, for interface compatibility)
            namespace: Kubernetes namespace
            deployment_name: Deployment name (defaults to service_id)
            **kwargs: Additional arguments
        
        Returns:
            int: Current replica count
        """
        deployment_name = deployment_name or service_id

        try:
            api = self._get_apps_api(target_node=node)
            scale = api.read_namespaced_deployment_scale(name=deployment_name, namespace=namespace)
            replicas = getattr(getattr(scale, 'status', None), 'replicas', None)
            if replicas is None:
                # Fallback: try reading full deployment
                dep = api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
                replicas = getattr(getattr(dep, 'status', None), 'replicas', 0)
            return int(replicas or 0)
        except Exception as e:
            if self.verbose:
                print(f"[KubernetesAllocator] Failed to get replica count via API: {e}")
            return 0
    
    def _set_replica_count(
        self,
        deployment_name: str,
        namespace: str,
        replicas: int,
        timeout: float = 30.0,
        node_hint: Optional[str] = None
    ) -> bool:
        """
        Set replica count for a Deployment using the Kubernetes API.

        Args:
            deployment_name: Name of Deployment
            namespace: Kubernetes namespace
            replicas: Desired replica count
            timeout: Unused (kept for signature compatibility)

        Returns:
            bool: True if the API call succeeded
        """
        try:
            api = self._get_apps_api(target_node=node_hint)
            body = {"spec": {"replicas": int(replicas)}}
            api.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            return True
        except Exception as e:
            if self.verbose:
                print(f"[KubernetesAllocator] API patch scale failed: {e}")
            return False

    def _get_core_api(self, target_node: Optional[str] = None):
        """Get or initialize Kubernetes CoreV1Api client for the appropriate cluster.
        Mirrors _get_apps_api but returns CoreV1Api.
        """
        if self._k8s_client is None or self._k8s_config is None:
            raise RuntimeError("kubernetes client library is not available. Install 'kubernetes' package.")

        # Determine which cluster to use based on target_node
        cluster_config = None
        if target_node and target_node in self.node_to_cluster:
            cluster_config = self.node_to_cluster[target_node]
        elif self.cluster_configs:
            cluster_config = self.cluster_configs[0]

        if not cluster_config:
            cache_key = 'default-core'
            if cache_key in self._core_api_cache:
                return self._core_api_cache[cache_key]

            with suppress(Exception):
                self._k8s_config.load_incluster_config()
            if not self._k8s_client.Configuration().host:
                self._k8s_config.load_kube_config()

            api = self._k8s_client.CoreV1Api()
            self._core_api_cache[cache_key] = api
            return api

        context = cluster_config.get('context')
        kubeconfig_path = cluster_config.get('kubeconfig_path')
        cache_key = f"core:{kubeconfig_path}:{context}"

        if cache_key in self._core_api_cache:
            return self._core_api_cache[cache_key]

        if kubeconfig_path and kubeconfig_path != 'null':
            self._k8s_config.load_kube_config(config_file=kubeconfig_path, context=context)
        else:
            with suppress(Exception):
                self._k8s_config.load_incluster_config()
            if not self._k8s_client.Configuration().host:
                self._k8s_config.load_kube_config(context=context)

        api = self._k8s_client.CoreV1Api()
        self._core_api_cache[cache_key] = api
        return api

    def _get_apps_api(self, target_node: Optional[str] = None):
        """Get or initialize Kubernetes AppsV1Api client for the appropriate cluster.
        
        Args:
            target_node: Node ID to determine which cluster to use
            
        Returns:
            AppsV1Api client for the appropriate cluster
        """
        if self._k8s_client is None or self._k8s_config is None:
            raise RuntimeError("kubernetes client library is not available. Install 'kubernetes' package.")
        
        # Determine which cluster to use based on target_node
        cluster_config = None
        if target_node and target_node in self.node_to_cluster:
            cluster_config = self.node_to_cluster[target_node]
        elif self.cluster_configs:
            # Fallback to first cluster if node mapping not available
            cluster_config = self.cluster_configs[0]
        
        if not cluster_config:
            # Last resort: try default kubeconfig
            cache_key = 'default'
            if cache_key in self._apps_api_cache:
                return self._apps_api_cache[cache_key]
            
            with suppress(Exception):
                self._k8s_config.load_incluster_config()
            if not self._k8s_client.Configuration().host:
                self._k8s_config.load_kube_config()
            
            api = self._k8s_client.AppsV1Api()
            self._apps_api_cache[cache_key] = api
            return api
        
        # Use cluster-specific config
        context = cluster_config.get('context')
        kubeconfig_path = cluster_config.get('kubeconfig_path')
        cache_key = f"{kubeconfig_path}:{context}"
        
        # Return cached client if available
        if cache_key in self._apps_api_cache:
            return self._apps_api_cache[cache_key]
        
        # Load config for this cluster
        if kubeconfig_path and kubeconfig_path != 'null':
            self._k8s_config.load_kube_config(config_file=kubeconfig_path, context=context)
        else:
            self._k8s_config.load_kube_config(context=context)
        
        # Create and cache API client
        api = self._k8s_client.AppsV1Api()
        self._apps_api_cache[cache_key] = api
        
        return api
