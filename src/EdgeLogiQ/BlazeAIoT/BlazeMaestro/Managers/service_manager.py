"""
Service Manager Module - Single Responsibility Principle
Manages service discovery, expansion, and allocation tracking.
"""
from BlazeAIoT.Core.Utils import BlazeLogger
from typing import Dict, List
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

    def get_services(self) -> List:
        """Get the services list."""
        return self.services

