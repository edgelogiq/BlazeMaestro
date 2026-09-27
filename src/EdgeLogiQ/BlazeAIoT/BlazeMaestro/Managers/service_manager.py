"""
Service Manager Module - Single Responsibility Principle
Manages service discovery, expansion, and allocation tracking.
"""
from BlazeAIoT.Core.Utils import BlazeLogger
from typing import Dict, List, Optional
import os
import shutil


class ServiceManager:
    """
    Manages service lifecycle including discovery, expansion, and allocation.
    SRP: Responsible only for service-related operations.
    """

    def __init__(self, config: dict, nodes: List, layers: List):
        """
        Initialize service manager.
        
        Args:
            config: Configuration dictionary
            nodes: List of node objects
            layers: List of layer objects
        """
        self.config = config
        self.nodes = nodes
        self.layers = layers
        self._logger = BlazeLogger.get_instance()
        
        self.services = None
        self.current_allocation = {}
        self.current_pod_refs = {}
        self.locked_services = set()

    def expand_services_to_instances(
        self,
        services_xml_path: str,
        metrics_provider,
        kubernetes_discovery
    ) -> bool:
        """
        Expand daemonsets and flexible deployments into explicit per-node instances.
        
        Args:
            services_xml_path: Path to services XML file
            metrics_provider: Metrics provider for fetching real metrics
            kubernetes_discovery: Kubernetes discovery service for pod mapping
            
        Returns:
            True if changes were made, False otherwise
        """
        if not self.nodes or not self.services:
            return False

        try:
            from ..Topology.xml_instance_expander import expand_services_to_instances
            
            # Build nodes by layer
            nodes_by_layer = self._build_nodes_by_layer()

            # Build pod->node mappings from actual running workloads
            pod_node_mapping = kubernetes_discovery.build_pod_node_mapping()

            # Determine output paths
            services_update_xml = self._get_services_update_path(services_xml_path)
            temp_update_xml = services_update_xml + ".tmp"

            # Create update file from template if it doesn't exist
            if not os.path.exists(services_update_xml):
                shutil.copy2(services_xml_path, services_update_xml)
                self._logger.info(f"Created {services_update_xml} from template {services_xml_path}")
            else:
                self._logger.debug(f"Using existing {services_update_xml} to preserve metrics")

            # Read existing content to detect changes
            old_content = self._read_file_content(services_update_xml)

            # Expand services
            expand_services_to_instances(
                services_xml_path,
                nodes_by_layer,
                temp_update_xml,
                metrics_provider=metrics_provider,
                current_allocation=self.current_allocation,
                pod_node_mapping=pod_node_mapping
            )

            # Check if changes were made
            new_content = self._read_file_content(temp_update_xml)
            changed = self._update_if_changed(old_content, new_content, temp_update_xml, services_update_xml)

            if metrics_provider:
                self._logger.debug("  Using real Prometheus metrics (min/max/avg)")
            else:
                self._logger.debug("  Using default metrics (no Prometheus)")

            return changed

        except Exception as e:
            self._logger.error("Service instance expansion failed: %s", e)
            import traceback
            traceback.print_exc()
            return False

    def _build_nodes_by_layer(self) -> Dict[str, List[tuple]]:
        """Build mapping of nodes organized by layer."""
        nodes_by_layer = {}
        for layer in self.layers:
            layer_id = layer.layer_id
            nodes_by_layer[layer_id] = [
                (
                    getattr(n, 'node_id', ''),
                    float(getattr(getattr(n, 'cpu', None), 'capacity', 0)) or float(getattr(n, 'cpu', 0)),
                    float(getattr(getattr(n, 'ram', None), 'capacity', 0)) or float(getattr(n, 'ram', 0))
                )
                for n in layer.get_nodes()
            ]
        return nodes_by_layer

    @staticmethod
    def _get_services_update_path(services_xml_path: str) -> str:
        """Get path for services update XML file."""
        if services_xml_path.endswith('_update.xml'):
            return services_xml_path
        return services_xml_path.replace('.xml', '_update.xml')

    @staticmethod
    def _read_file_content(file_path: str) -> bytes:
        """Read file content as bytes."""
        if not os.path.exists(file_path):
            return b''
        with open(file_path, 'rb') as f:
            return f.read()

    def _update_if_changed(
        self,
        old_content: bytes,
        new_content: bytes,
        temp_path: str,
        target_path: str
    ) -> bool:
        """Update target file if content has changed."""
        if not old_content or old_content != new_content:
            shutil.move(temp_path, target_path)
            self._logger.debug(f"✓ Merged instances into {target_path} (changes detected)")
            return True
        else:
            # No changes; remove temp file
            try:
                os.remove(temp_path)
            except OSError:
                pass
            self._logger.debug("No changes in services_update.xml; skipping overwrite")
            return False

    def set_services(self, services: List):
        """Set the services list."""
        self.services = services

    def set_current_allocation(self, allocation: Dict):
        """Set current service allocation."""
        self.current_allocation = allocation

    def set_current_pod_refs(self, pod_refs: Dict):
        """Set current pod references."""
        self.current_pod_refs = pod_refs

    def add_locked_service(self, service_id: str):
        """Mark a service as locked (cannot be rescheduled)."""
        self.locked_services.add(service_id)

    def get_services(self) -> List:
        """Get the services list."""
        return self.services

    def get_current_allocation(self) -> Dict:
        """Get current service allocation."""
        return self.current_allocation

    def get_current_pod_refs(self) -> Dict:
        """Get current pod references."""
        return self.current_pod_refs

    def get_locked_services(self) -> set:
        """Get set of locked services."""
        return self.locked_services

    def collect_measured_usage_from_services(self) -> dict:
        """
        Collect measured resource usage from all services.
        
        Returns:
            Dictionary mapping service_id to usage metrics
        """
        if not self.services:
            return {}

        measured_usage = {}
        for service in self.services:
            service_id = service.service_id
            
            # Aggregate usage across all containers in the service
            total_cpu = 0.0
            total_ram = 0.0
            container_count = 0
            
            for container in service.containers:
                if hasattr(container, 'measured_cpu') and container.measured_cpu is not None:
                    total_cpu += container.measured_cpu
                if hasattr(container, 'measured_ram') and container.measured_ram is not None:
                    total_ram += container.measured_ram
                container_count += 1
            
            if container_count > 0:
                measured_usage[service_id] = {
                    'cpu': total_cpu,
                    'ram_gb': total_ram,
                    'container_count': container_count
                }
        
        return measured_usage

    def log_all_services_allocation(self):
        """Log current allocation for all services."""
        if not self.services or not self.current_allocation:
            self._logger.debug("No services or allocation to log")
            return

        self._logger.info("=" * 80)
        self._logger.info("CURRENT SERVICE ALLOCATION")
        self._logger.info("=" * 80)
        
        for service in self.services:
            service_id = service.service_id
            allocation = self.current_allocation.get(service_id, [])
            
            if not allocation:
                self._logger.info(f"Service: {service_id} - NOT ALLOCATED")
                continue
            
            self._logger.info(f"Service: {service_id}")
            for i, alloc in enumerate(allocation, 1):
                node = alloc.get('node', 'unknown')
                pod = alloc.get('pod', 'unknown')
                layer = alloc.get('layer', 'unknown')
                self._logger.info(f"  Instance {i}: node={node}, pod={pod}, layer={layer}")
        
        self._logger.info("=" * 80)
