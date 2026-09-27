"""
Kubernetes Platform implementation.

Discovers topology from one or more Kubernetes clusters.
"""

from typing import List, Tuple, Optional, Dict
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.platform import Platform
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.layer import Layer
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node


class KubernetesPlatform(Platform):
    """
    Platform implementation for Kubernetes clusters.

    Discovers nodes from one or more Kubernetes clusters and organizes them
    into layers based on labels.
    """

    def __init__(
        self,
        kubeconfig_paths: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
        layer_label: str = "blaze.layer",
        namespace: Optional[str] = None
    ):
        """
        Initialize Kubernetes platform.

        Args:
            kubeconfig_paths: List of paths to kubeconfig files (None = default)
            contexts: List of context names to use (None = current context)
            layer_label: Label key used to identify node layer (default: "blaze.layer")
            namespace: Namespace to filter nodes (None = all namespaces)
        """
        self.kubeconfig_paths = kubeconfig_paths or [None]
        self.contexts = contexts or [None]
        self.layer_label = layer_label
        self.namespace = namespace

    def build(self) -> Tuple[List[Layer], List[Node]]:
        """
        Build topology by discovering nodes from Kubernetes clusters.

        Returns:
            Tuple of (layers, nodes)
        """
        # Disable SSL verification globally BEFORE loading any kubeconfig or creating clients
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            from kubernetes.client import configuration
            configuration.ssl_verify = False
        except ImportError:
            pass

        try:
            from kubernetes import client, config
        except ImportError:
            raise ImportError(
                "Kubernetes client library not installed. "
                "Install with: pip install kubernetes"
            )

        layers_dict: Dict[str, Layer] = {}
        all_nodes = []

        # Iterate over clusters/contexts
        for kubeconfig_path, context in zip(self.kubeconfig_paths, self.contexts):
            # Load kube config
            try:
                print(f"  Connecting to context: {context or 'default'}...")
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path, context=context)
                else:
                    config.load_kube_config(context=context)

                print(f"    ✓ Connected")

                # Create API client (SSL verification already disabled globally)
                v1 = client.CoreV1Api()

                # List all nodes
                print(f"    Listing nodes...")
                k8s_nodes = v1.list_node(timeout_seconds=10)
                print(f"    ✓ Found {len(k8s_nodes.items)} nodes")
            except Exception as e:
                print(f"    ✗ Failed to list nodes: {e}")
                print(f"    ℹ Skipping this context")
                continue

            # Process each node
            for k8s_node in k8s_nodes.items:
                node = self._parse_k8s_node(k8s_node, layers_dict)
                if node:
                    all_nodes.append(node)

        return list(layers_dict.values()), all_nodes

    def _parse_k8s_node(self, k8s_node, layers_dict: Dict[str, Layer]) -> Optional[Node]:
        """
        Parse a Kubernetes node into a Node object.

        Args:
            k8s_node: Kubernetes node object
            layers_dict: Dictionary of layers (will be updated)

        Returns:
            Node object or None if node should be skipped
        """
        # Get node name
        node_name = k8s_node.metadata.name

        # Get layer from labels
        labels = k8s_node.metadata.labels or {}
        layer_id = labels.get(self.layer_label, 'default')
        
        # Map frank layer to edge (frank nodes are edge devices)
        if layer_id == 'frank':
            layer_id = 'edge'

        # Create layer if it doesn't exist
        if layer_id not in layers_dict:
            layers_dict[layer_id] = Layer(layer_id)

        # Extract resources
        capacity = k8s_node.status.capacity

        # CPU (cores)
        cpu_str = capacity.get('cpu', '0')
        cpu = self._parse_cpu(cpu_str)

        # Memory (convert to GB)
        memory_str = capacity.get('memory', '0')
        ram = self._parse_memory(memory_str)

        # GPU (if available)
        gpu = 0
        for gpu_key in ['nvidia.com/gpu', 'amd.com/gpu', 'intel.com/gpu']:
            if gpu_key in capacity:
                try:
                    gpu_count = int(capacity[gpu_key])
                    # Assume 8GB per GPU if not specified
                    gpu = gpu_count * 8
                    break
                except ValueError:
                    pass

        # Get worker label first (before taint checks)
        worker_label = labels.get('node-role.kubernetes.io/worker')
        worker_val = str(worker_label).lower() if worker_label is not None else ''
        has_worker_role = worker_val in ('true', 'yes', '1')

        # Check if node is schedulable (check for NoSchedule taint)
        # But allow worker nodes even if tainted (they're explicitly designated for workloads)
        taints = k8s_node.spec.taints or []
        schedulable = True
        if not has_worker_role:
            for taint in taints:
                if taint.effect in ['NoSchedule', 'NoExecute']:
                    schedulable = False
                    break

        # Also check if node is marked as unschedulable
        # But allow worker nodes to override this
        if k8s_node.spec.unschedulable and not has_worker_role:
            schedulable = False

        # Treat nodes as non-schedulable only if:
        # 1. They have control-plane labels AND
        # 2. They do NOT have an explicit worker role
        # This allows mixed nodes (both control-plane + worker) to be schedulable
        if schedulable and not has_worker_role:
            control_plane_labels = {
                'node-role.kubernetes.io/control-plane',
                'node-role.kubernetes.io/master',
            }
            has_control_plane = False
            if labels.get('kubernetes.io/role', '') == 'master':
                has_control_plane = True
            elif labels.get('node.kubernetes.io/role', '') == 'master':
                has_control_plane = True
            else:
                for lbl in control_plane_labels:
                    if lbl in labels:
                        has_control_plane = True
                        break
                if not has_control_plane:
                    for key in labels.keys():
                        if 'control-plane' in key or key.endswith('/master') or key == 'master':
                            has_control_plane = True
                            break
            
            if has_control_plane:
                schedulable = False

        # Debug: show schedulability decision (stdout so it is always visible)
        control_labels_dbg = {k: v for k, v in labels.items() if 'role' in k or 'control' in k or k.endswith('/master')}
        taints_dbg = [getattr(t, 'effect', None) for t in taints]
        worker_label_val = labels.get('node-role.kubernetes.io/worker')

        # Create node
        node = Node(node_name, cpu=cpu, ram=ram, gpu=gpu, schedulable=schedulable)

        # Add to layer
        layers_dict[layer_id].add_node(node)

        # Store layer_id on node for allocation emitter
        node.layer_id = layer_id

        # Store Kubernetes metadata
        node.k8s_labels = labels
        node.k8s_annotations = k8s_node.metadata.annotations or {}
        node.k8s_taints = taints

        return node

    def _parse_cpu(self, cpu_str: str) -> int:
        """
        Parse CPU string from Kubernetes.

        Args:
            cpu_str: CPU string (e.g., "4", "4000m")

        Returns:
            Number of CPU cores
        """
        if cpu_str.endswith('m'):
            # Millicores
            return int(cpu_str[:-1]) / 1000
        else:
            return int(cpu_str)

    def _parse_memory(self, memory_str: str) -> float:
        """
        Parse memory string from Kubernetes.

        Args:
            memory_str: Memory string (e.g., "16Gi", "8192Mi", "8589934592")

        Returns:
            Memory in GB
        """
        if memory_str.endswith('Ki'):
            return int(memory_str[:-2]) / (1024 ** 2)
        elif memory_str.endswith('Mi'):
            return int(memory_str[:-2]) / 1024
        elif memory_str.endswith('Gi'):
            return int(memory_str[:-2])
        elif memory_str.endswith('Ti'):
            return int(memory_str[:-2]) * 1024
        else:
            # Assume bytes
            return int(memory_str) / (1024 ** 3)

    def get_name(self) -> str:
        """Get platform name."""
        cluster_count = len(self.kubeconfig_paths)
        return f"Kubernetes ({cluster_count} cluster{'s' if cluster_count > 1 else ''})"

    def get_services(self, namespace: str = "default", label_selector: str = None) -> Dict[str, List]:
        """
        Discover services (DaemonSets and Deployments) from the cluster.

        Args:
            namespace: Kubernetes namespace to query (default: "default")
            label_selector: Optional label selector to filter services

        Returns:
            Dictionary with keys 'daemonsets' and 'deployments', each containing a list of service info
        """
        # Disable SSL verification globally BEFORE loading any kubeconfig or creating clients
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            from kubernetes.client import configuration
            configuration.ssl_verify = False
        except ImportError:
            pass

        try:
            from kubernetes import client, config
        except ImportError:
            raise ImportError("Kubernetes client library not installed")

        result = {
            'daemonsets': [],
            'deployments': []
        }

        for kubeconfig_path, context in zip(self.kubeconfig_paths, self.contexts):
            try:
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path, context=context)
                else:
                    config.load_kube_config(context=context)

                apps_v1 = client.AppsV1Api()

                # Get DaemonSets
                try:
                    daemonsets = apps_v1.list_namespaced_daemon_set(
                        namespace=namespace,
                        label_selector=label_selector,
                        timeout_seconds=10
                    )
                    for ds in daemonsets.items:
                        result['daemonsets'].append({
                            'name': ds.metadata.name,
                            'namespace': ds.metadata.namespace,
                            'node_selector': ds.spec.template.spec.node_selector or {},
                            'containers': self._extract_containers(ds.spec.template.spec.containers)
                        })
                except Exception as e:
                    print(f"Warning: Failed to list DaemonSets from {namespace}: {e}")

                # Get Deployments
                try:
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector=label_selector,
                        timeout_seconds=10
                    )
                    for deploy in deployments.items:
                        result['deployments'].append({
                            'name': deploy.metadata.name,
                            'namespace': deploy.metadata.namespace,
                            'replicas': deploy.spec.replicas,
                            'node_selector': deploy.spec.template.spec.node_selector or {},
                            'containers': self._extract_containers(deploy.spec.template.spec.containers)
                        })
                except Exception as e:
                    print(f"Warning: Failed to list Deployments from {namespace}: {e}")

            except Exception as e:
                print(f"Warning: Failed to connect to context {context}: {e}")
                continue

        return result

    def _extract_containers(self, containers) -> List[Dict]:
        """Extract container specs from Kubernetes container list."""
        result = []
        for container in containers:
            resources = container.resources
            requests = resources.requests if resources and resources.requests else {}
            limits = resources.limits if resources and resources.limits else {}

            cpu_str = requests.get('cpu') or limits.get('cpu') or '0'
            memory_str = requests.get('memory') or limits.get('memory') or '0'

            # Parse GPU
            gpu = 0
            for gpu_key in ['nvidia.com/gpu', 'amd.com/gpu']:
                if gpu_key in requests:
                    gpu = float(requests[gpu_key])
                    break
                elif gpu_key in limits:
                    gpu = float(limits[gpu_key])
                    break

            result.append({
                'name': container.name,
                'image': container.image,
                'cpu': self._parse_cpu(cpu_str),
                'memory_gb': self._parse_memory(memory_str),
                'gpu': gpu
            })

        return result


class MultiKubernetesPlatform(Platform):
    """
    Platform implementation for multiple Kubernetes clusters.

    Each cluster can be mapped to a different layer.
    """

    def __init__(self, cluster_configs: List[Dict]):
        """
        Initialize multi-cluster Kubernetes platform.

        Args:
            cluster_configs: List of cluster configurations, each with:
                - kubeconfig_path: Path to kubeconfig (optional)
                - context: Context name (optional)
                - layer_id: Layer ID for this cluster (required)
                - name: Cluster name (optional)
        """
        self.cluster_configs = cluster_configs

    def build(self) -> Tuple[List[Layer], List[Node]]:
        """
        Build topology from multiple Kubernetes clusters.

        Returns:
            Tuple of (layers, nodes)
        """
        # Disable SSL verification globally BEFORE loading any kubeconfig or creating clients
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            from kubernetes.client import configuration
            configuration.ssl_verify = False
        except ImportError:
            pass

        try:
            from kubernetes import client, config
        except ImportError:
            raise ImportError(
                "Kubernetes client library not installed. "
                "Install with: pip install kubernetes"
            )

        layers_dict: Dict[str, Layer] = {}
        all_nodes = []

        for cluster_config in self.cluster_configs:
            layer_id = cluster_config['layer_id']
            kubeconfig_path = cluster_config.get('kubeconfig_path')
            context = cluster_config.get('context')
            cluster_name = cluster_config.get('name', layer_id)

            # Create layer if it doesn't exist
            if layer_id not in layers_dict:
                layers_dict[layer_id] = Layer(layer_id)

            # Load kube config
            try:
                print(f"  Connecting to cluster: {cluster_name} (context: {context})...")
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path, context=context)
                else:
                    config.load_kube_config(context=context)

                print(f"    ✓ Connected")

                # Disable SSL verification for this API client
                api_client = client.ApiClient()

                # Create insecure SSL context
                import ssl
                import urllib3
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                # Configure the API client's HTTP adapter
                api_client.rest_client.pool_manager.connection_pool_kw['ssl_context'] = ssl_context

                # Create API client with SSL verification disabled
                v1 = client.CoreV1Api(api_client)

                # List all nodes
                print(f"    Listing nodes...")
                k8s_nodes = v1.list_node(timeout_seconds=10)
                print(f"    ✓ Found {len(k8s_nodes.items)} nodes")
            except Exception as e:
                print(f"    ✗ Failed to list nodes: {e}")
                print(f"    ℹ Skipping cluster: {cluster_name}")
                continue

            # Process each node
            for k8s_node in k8s_nodes.items:
                node = self._parse_k8s_node(k8s_node, cluster_name)
                if node:
                    # Set layer_id on node for orchestrator constraint checking
                    node.layer_id = layer_id
                    layers_dict[layer_id].add_node(node)
                    all_nodes.append(node)

        # Debug: summary of discovered nodes
        schedulable_count = sum(1 for n in all_nodes if getattr(n, 'schedulable', True))
        unschedulable_count = len(all_nodes) - schedulable_count
        print(f"\n[K8sTopology] Topology built: {len(all_nodes)} total nodes, {schedulable_count} schedulable, {unschedulable_count} unschedulable")
        
        return list(layers_dict.values()), all_nodes

    def _parse_k8s_node(self, k8s_node, cluster_name: str) -> Optional[Node]:
        """Parse Kubernetes node into Node object."""
        # Use the k8s node name directly (cluster is already identified by layer)
        k8s_node_name = k8s_node.metadata.name
        node_name = k8s_node_name  # Don't prefix with cluster name

        # Get labels for schedulability checks
        labels = k8s_node.metadata.labels or {}

        # Extract resources
        capacity = k8s_node.status.capacity

        cpu_str = capacity.get('cpu', '0')
        cpu = self._parse_cpu(cpu_str)

        memory_str = capacity.get('memory', '0')
        ram = self._parse_memory(memory_str)

        gpu = 0
        for gpu_key in ['nvidia.com/gpu', 'amd.com/gpu', 'intel.com/gpu']:
            if gpu_key in capacity:
                try:
                    gpu_count = int(capacity[gpu_key])
                    gpu = gpu_count * 8  # Assume 8GB per GPU
                    break
                except ValueError:
                    pass

        # Get worker label first (before taint checks)
        worker_label = labels.get('node-role.kubernetes.io/worker')
        worker_val = str(worker_label).lower() if worker_label is not None else ''
        has_worker_role = worker_val in ('true', 'yes', '1')

        # Check if node is schedulable (check for NoSchedule taint)
        # But allow worker nodes even if tainted (they're explicitly designated for workloads)
        taints = k8s_node.spec.taints or []
        schedulable = True
        if not has_worker_role:
            for taint in taints:
                if taint.effect in ['NoSchedule', 'NoExecute']:
                    schedulable = False
                    break

        # Also check if node is marked as unschedulable
        # But allow worker nodes to override this
        if k8s_node.spec.unschedulable and not has_worker_role:
            schedulable = False

        # Treat nodes as non-schedulable only if:
        # 1. They have control-plane labels AND
        # 2. They do NOT have an explicit worker role
        # This allows mixed nodes (both control-plane + worker) to be schedulable
        if schedulable and not has_worker_role:
            control_plane_labels = {
                'node-role.kubernetes.io/control-plane',
                'node-role.kubernetes.io/master',
            }
            has_control_plane = False
            if labels.get('kubernetes.io/role', '') == 'master':
                has_control_plane = True
            elif labels.get('node.kubernetes.io/role', '') == 'master':
                has_control_plane = True
            else:
                for lbl in control_plane_labels:
                    if lbl in labels:
                        has_control_plane = True
                        break
                if not has_control_plane:
                    for key in labels.keys():
                        if 'control-plane' in key or key.endswith('/master') or key == 'master':
                            has_control_plane = True
                            break
            
            if has_control_plane:
                schedulable = False

        # Debug: show schedulability decision
        control_labels_dbg = {k: v for k, v in labels.items() if 'role' in k or 'control' in k or k.endswith('/master')}
        taints_dbg = [getattr(t, 'effect', None) for t in taints]
        worker_label_val = labels.get('node-role.kubernetes.io/worker')

        node = Node(node_name, cpu=cpu, ram=ram, gpu=gpu, schedulable=schedulable)
        node.k8s_cluster = cluster_name
        node.k8s_node_name = k8s_node_name  # Store original k8s node name for Prometheus queries
        node.k8s_labels = k8s_node.metadata.labels or {}

        return node

    def _parse_cpu(self, cpu_str: str) -> int:
        """Parse CPU from Kubernetes format."""
        if cpu_str.endswith('m'):
            return int(cpu_str[:-1]) / 1000
        return int(cpu_str)

    def _parse_memory(self, memory_str: str) -> float:
        """Parse memory from Kubernetes format (returns GB)."""
        if memory_str.endswith('Ki'):
            return int(memory_str[:-2]) / (1024 ** 2)
        elif memory_str.endswith('Mi'):
            return int(memory_str[:-2]) / 1024
        elif memory_str.endswith('Gi'):
            return int(memory_str[:-2])
        elif memory_str.endswith('Ti'):
            return int(memory_str[:-2]) * 1024
        else:
            return int(memory_str) / (1024 ** 3)

    def get_name(self) -> str:
        """Get platform name."""
        return f"Multi-Kubernetes ({len(self.cluster_configs)} clusters)"

    def get_services(self, namespace: str = "default", label_selector: str = None) -> Dict[str, List]:
        """
        Discover services (DaemonSets and Deployments) from all clusters.

        Args:
            namespace: Kubernetes namespace to query (default: "default")
            label_selector: Optional label selector to filter services

        Returns:
            Dictionary with keys 'daemonsets' and 'deployments', each containing a list of dicts with:
                - name: Service name
                - namespace: Namespace
                - cluster: Cluster name
                - replicas: Number of replicas (for deployments)
                - node_selector: Node selector labels
                - containers: List of container specs
        """
        # Disable SSL verification globally BEFORE loading any kubeconfig or creating clients
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            from kubernetes.client import configuration
            configuration.ssl_verify = False
        except ImportError:
            pass

        try:
            from kubernetes import client, config
        except ImportError:
            raise ImportError("Kubernetes client library not installed")

        result = {
            'daemonsets': [],
            'deployments': []
        }

        for cluster_config in self.cluster_configs:
            kubeconfig_path = cluster_config.get('kubeconfig_path')
            context = cluster_config.get('context')
            cluster_name = cluster_config.get('name', cluster_config['layer_id'])

            try:
                if kubeconfig_path:
                    config.load_kube_config(config_file=kubeconfig_path, context=context)
                else:
                    config.load_kube_config(context=context)

                # Disable SSL verification and warnings for self-signed certificates
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                # Disable SSL verification in the client configuration
                from kubernetes.client import configuration
                configuration.ssl_verify = False

                apps_v1 = client.AppsV1Api()

                # Get DaemonSets
                try:
                    daemonsets = apps_v1.list_namespaced_daemon_set(
                        namespace=namespace,
                        label_selector=label_selector,
                        timeout_seconds=10
                    )
                    for ds in daemonsets.items:
                        result['daemonsets'].append({
                            'name': ds.metadata.name,
                            'namespace': ds.metadata.namespace,
                            'cluster': cluster_name,
                            'node_selector': ds.spec.template.spec.node_selector or {},
                            'containers': self._extract_containers(ds.spec.template.spec.containers)
                        })
                except Exception as e:
                    print(f"Warning: Failed to list DaemonSets from {cluster_name}/{namespace}: {e}")

                # Get Deployments
                try:
                    deployments = apps_v1.list_namespaced_deployment(
                        namespace=namespace,
                        label_selector=label_selector,
                        timeout_seconds=10
                    )
                    for deploy in deployments.items:
                        result['deployments'].append({
                            'name': deploy.metadata.name,
                            'namespace': deploy.metadata.namespace,
                            'cluster': cluster_name,
                            'replicas': deploy.spec.replicas,
                            'node_selector': deploy.spec.template.spec.node_selector or {},
                            'containers': self._extract_containers(deploy.spec.template.spec.containers)
                        })
                except Exception as e:
                    print(f"Warning: Failed to list Deployments from {cluster_name}/{namespace}: {e}")

            except Exception as e:
                print(f"Warning: Failed to connect to cluster {cluster_name}: {e}")
                continue

        return result

    def _extract_containers(self, containers) -> List[Dict]:
        """Extract container specs from Kubernetes container list."""
        result = []
        for container in containers:
            resources = container.resources
            requests = resources.requests if resources and resources.requests else {}
            limits = resources.limits if resources and resources.limits else {}

            cpu_str = requests.get('cpu') or limits.get('cpu') or '0'
            memory_str = requests.get('memory') or limits.get('memory') or '0'

            # Parse GPU
            gpu = 0
            for gpu_key in ['nvidia.com/gpu', 'amd.com/gpu']:
                if gpu_key in requests:
                    gpu = float(requests[gpu_key])
                    break
                elif gpu_key in limits:
                    gpu = float(limits[gpu_key])
                    break

            result.append({
                'name': container.name,
                'image': container.image,
                'cpu': self._parse_cpu(cpu_str),
                'memory_gb': self._parse_memory(memory_str),
                'gpu': gpu
            })

        return result

