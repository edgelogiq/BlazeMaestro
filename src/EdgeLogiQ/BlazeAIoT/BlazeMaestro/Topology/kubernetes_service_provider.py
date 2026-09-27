"""
Kubernetes-based Service Provider implementation.

Discovers services from Kubernetes namespaces by querying Deployments,
StatefulSets, and DaemonSets.
"""

from typing import List, Optional
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_provider import ServiceProvider
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service, ContainerSpec, DependencySpec


class KubernetesServiceProvider(ServiceProvider):
    """
    Service provider that discovers services from Kubernetes namespaces.

    Maps Kubernetes workloads (Deployments, StatefulSets, DaemonSets) to
    BlazeAIoT service definitions.
    """

    def __init__(
        self,
        namespace: str = "default",
        kubeconfig_path: Optional[str] = None,
        context: Optional[str] = None,
        label_selector: Optional[str] = None
    ):
        """
        Initialize Kubernetes service provider.

        Args:
            namespace: Kubernetes namespace to query
            kubeconfig_path: Path to kubeconfig file (None = default)
            context: Context name to use (None = current context)
            label_selector: Label selector to filter workloads (e.g., "app=blaze")
        """
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.context = context
        self.label_selector = label_selector

    def get_services(self) -> List[Service]:
        """
        Discover services from Kubernetes namespace.

        Returns:
            List of Service objects
        """
        try:
            from kubernetes import client, config
        except ImportError:
            raise ImportError(
                "Kubernetes client library not installed. "
                "Install with: pip install kubernetes"
            )

        # Load kube config
        if self.kubeconfig_path:
            config.load_kube_config(config_file=self.kubeconfig_path, context=self.context)
        else:
            config.load_kube_config(context=self.context)

        apps_v1 = client.AppsV1Api()
        services = []

        # Get Deployments
        try:
            deployments = apps_v1.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector=self.label_selector
            )
            for deployment in deployments.items:
                service = self._parse_deployment(deployment)
                if service:
                    services.append(service)
        except Exception as e:
            print(f"Warning: Failed to list deployments: {e}")

        # Get StatefulSets
        try:
            statefulsets = apps_v1.list_namespaced_stateful_set(
                namespace=self.namespace,
                label_selector=self.label_selector
            )
            for statefulset in statefulsets.items:
                service = self._parse_statefulset(statefulset)
                if service:
                    services.append(service)
        except Exception as e:
            print(f"Warning: Failed to list statefulsets: {e}")

        # Get DaemonSets
        try:
            daemonsets = apps_v1.list_namespaced_daemon_set(
                namespace=self.namespace,
                label_selector=self.label_selector
            )
            for daemonset in daemonsets.items:
                service = self._parse_daemonset(daemonset)
                if service:
                    services.append(service)
        except Exception as e:
            print(f"Warning: Failed to list daemonsets: {e}")

        return services

    def _parse_deployment(self, deployment) -> Optional[Service]:
        """Parse Kubernetes Deployment into Service."""
        service_id = deployment.metadata.name

        # Extract containers
        containers = self._parse_containers(deployment.spec.template.spec.containers)

        if not containers:
            return None

        # Get placement from annotations/labels
        annotations = deployment.metadata.annotations or {}
        labels = deployment.metadata.labels or {}

        placement = self._get_placement(annotations, labels, is_daemonset=False)

        # Get execution rate from annotations
        execution_rate_hz = float(annotations.get('blaze.execution_rate_hz', '1.0'))

        service = Service(
            service_id=service_id,
            containers=containers,
            execution_rate_hz=execution_rate_hz,
            **placement
        )

        # Store Kubernetes metadata
        service.k8s_type = 'Deployment'
        service.k8s_namespace = self.namespace
        service.k8s_labels = labels

        return service

    def _parse_statefulset(self, statefulset) -> Optional[Service]:
        """Parse Kubernetes StatefulSet into Service."""
        service_id = statefulset.metadata.name

        containers = self._parse_containers(statefulset.spec.template.spec.containers)

        if not containers:
            return None

        annotations = statefulset.metadata.annotations or {}
        labels = statefulset.metadata.labels or {}

        placement = self._get_placement(annotations, labels, is_daemonset=False)
        execution_rate_hz = float(annotations.get('blaze.execution_rate_hz', '1.0'))

        service = Service(
            service_id=service_id,
            containers=containers,
            execution_rate_hz=execution_rate_hz,
            **placement
        )

        service.k8s_type = 'StatefulSet'
        service.k8s_namespace = self.namespace
        service.k8s_labels = labels

        return service

    def _parse_daemonset(self, daemonset) -> Optional[Service]:
        """Parse Kubernetes DaemonSet into Service."""
        service_id = daemonset.metadata.name

        containers = self._parse_containers(daemonset.spec.template.spec.containers)

        if not containers:
            return None

        annotations = daemonset.metadata.annotations or {}
        labels = daemonset.metadata.labels or {}

        # DaemonSets are always daemonsets in our model
        placement = self._get_placement(annotations, labels, is_daemonset=True)
        execution_rate_hz = float(annotations.get('blaze.execution_rate_hz', '1.0'))

        service = Service(
            service_id=service_id,
            containers=containers,
            execution_rate_hz=execution_rate_hz,
            **placement
        )

        service.k8s_type = 'DaemonSet'
        service.k8s_namespace = self.namespace
        service.k8s_labels = labels

        return service

    def _parse_containers(self, k8s_containers) -> List[ContainerSpec]:
        """Parse Kubernetes containers into ContainerSpec objects."""
        containers = []

        for k8s_container in k8s_containers:
            # Extract resource requests (or limits if no requests)
            resources = k8s_container.resources
            requests = resources.requests if resources and resources.requests else {}
            limits = resources.limits if resources and resources.limits else {}

            # CPU
            cpu_str = requests.get('cpu') or limits.get('cpu') or '1'
            cpu = self._parse_cpu(cpu_str)

            # RAM
            memory_str = requests.get('memory') or limits.get('memory') or '1Gi'
            ram_gb = self._parse_memory(memory_str)

            # GPU
            gpu_gb = 0
            for gpu_key in ['nvidia.com/gpu', 'amd.com/gpu']:
                if gpu_key in requests:
                    gpu_gb = float(requests[gpu_key]) * 8  # Assume 8GB per GPU
                    break
                elif gpu_key in limits:
                    gpu_gb = float(limits[gpu_key]) * 8
                    break

            # Note: exec_time is not available from Kubernetes, use default
            container = ContainerSpec(
                id=k8s_container.name,
                cpu=cpu,
                ram_gb=ram_gb,
                gpu_gb=gpu_gb,
                exec_time=1.0,  # Default, will need profiling
                exec_time_available=False  # Mark as estimated
            )

            containers.append(container)

        return containers

    def _get_placement(self, annotations: dict, labels: dict, is_daemonset: bool) -> dict:
        """Extract placement configuration from annotations/labels."""
        # Check for explicit placement annotations
        if 'blaze.placement.type' in annotations:
            placement_type = annotations['blaze.placement.type']

            if placement_type == 'node_lock':
                node_id = annotations.get('blaze.placement.node')
                if node_id:
                    return {'locked_node_id': node_id}

            elif placement_type == 'layer_lock':
                layer_id = annotations.get('blaze.placement.layer')
                if layer_id:
                    return {'locked_layer_id': layer_id}

        # If it's a Kubernetes DaemonSet, treat as Blaze daemonset
        if is_daemonset:
            layer_id = annotations.get('blaze.placement.layer', 'edge')
            return {
                'is_daemonset': True,
                'daemonset_layer_id': layer_id
            }

        # Check for layer from labels
        if 'blaze.layer' in labels:
            return {'locked_layer_id': labels['blaze.layer']}

        # Default: flexible placement
        return {}

    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse CPU from Kubernetes format."""
        if cpu_str.endswith('m'):
            return int(cpu_str[:-1]) / 1000
        return float(cpu_str)

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
        """Get provider name."""
        return f"Kubernetes (namespace: {self.namespace})"


class MultiNamespaceServiceProvider(ServiceProvider):
    """
    Service provider that aggregates services from multiple Kubernetes namespaces.
    """

    def __init__(
        self,
        namespaces: List[str],
        kubeconfig_path: Optional[str] = None,
        context: Optional[str] = None
    ):
        """
        Initialize multi-namespace service provider.

        Args:
            namespaces: List of namespace names to query
            kubeconfig_path: Path to kubeconfig file
            context: Context name to use
        """
        self.providers = [
            KubernetesServiceProvider(ns, kubeconfig_path, context)
            for ns in namespaces
        ]

    def get_services(self) -> List[Service]:
        """
        Get services from all namespaces.

        Returns:
            List of Service objects from all namespaces
        """
        all_services = []
        for provider in self.providers:
            services = provider.get_services()
            all_services.extend(services)
        return all_services

    def get_name(self) -> str:
        """Get provider name."""
        ns_count = len(self.providers)
        return f"Kubernetes ({ns_count} namespace{'s' if ns_count > 1 else ''})"

