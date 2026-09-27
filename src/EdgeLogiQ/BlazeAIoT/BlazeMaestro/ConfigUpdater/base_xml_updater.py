"""
Base class for XML configuration updaters.

Provides common infrastructure for updating platform.xml and services.xml
with metrics from different sources (Prometheus, simulator, etc.).

Architecture:
    
    BaseXMLUpdater (Abstract)
    ├── get_node_metrics() [abstract]
    ├── get_service_metrics() [abstract]
    ├── update_platform_xml() [abstract]
    ├── update_services_xml() [abstract]
    └── Common utilities:
        ├── _infer_layer_prefix()
        ├── _extract_index()
        ├── _match_host_to_node()
        ├── _should_prune_layer()
        ├── _prune_inactive_layers()
        ├── _indent_xml()
        └── _get_allowed_layers()
    
    PrometheusXMLUpdater (Concrete)
    ├── Queries Prometheus for node/service metrics
    ├── Pulls Kubernetes node capacities as fallback
    ├── Updates platform_update.xml with real capacities
    └── Updates services_update.xml with workload metrics
    
    SimulatorXMLUpdater (Concrete)
    ├── Uses template specs + overrides for metrics
    ├── Updates instance metrics in services_update.xml
    └── No platform updates (simulator uses static topology)

Usage:
    # Kubernetes mode with Prometheus
    updater = PrometheusXMLUpdater(metrics_facade=prometheus_facade)
    updater.update_platform_xml('platform.xml')
    updater.update_services_xml('services.xml')
    
    # Simulator mode
    updater = SimulatorXMLUpdater()
    updater.update_instance_metrics('services_update.xml', overrides={...})
"""
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any

class BaseXMLUpdater(ABC):
    """Abstract base class for XML configuration updaters."""

    # Layers we know how to reason about when pruning inactive tiers
    DEFAULT_LAYER_PREFIXES = {"cloud", "fog", "edge"}

    def __init__(self):
        """Initialize the base updater."""
        pass

    # Abstract methods that must be implemented by subclasses

    @abstractmethod
    def get_node_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get node resource metrics from the underlying source.

        Returns:
            Dictionary mapping node names to their metrics (cpu_cores, ram_gb, etc.)
        """
        pass

    @abstractmethod
    def get_service_metrics(self, namespace: str = 'default') -> Dict[str, Dict[str, float]]:
        """
        Get service/pod resource usage metrics from the underlying source.

        Args:
            namespace: Kubernetes namespace to filter (if applicable)

        Returns:
            Dictionary mapping service IDs to their metrics
        """
        pass

    @abstractmethod
    def update_platform_xml(self, xml_path: str, backup: bool = True) -> bool:
        """
        Update platform.xml with node metrics.

        Args:
            xml_path: Path to platform.xml file
            backup: Whether to create a backup (implementation-specific)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def update_services_xml(self, xml_path: str, namespace: str = 'default', backup: bool = True) -> bool:
        """
        Update services.xml with service metrics.

        Args:
            xml_path: Path to services.xml file
            namespace: Kubernetes namespace to filter (if applicable)
            backup: Whether to create a backup (implementation-specific)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def update_instance_metrics(self, metrics_provider: Any = None, nodes: List = None, 
                               services: List = None, current_allocation: Dict = None, **kwargs) -> bool:
        """
        Update metrics in a mode-agnostic way.
        
        For simulator mode: updates instance metrics directly in services XML.
        For Kubernetes mode: queries Prometheus and updates XML files.
        
        Args:
            metrics_provider: Metrics facade (simulator or Prometheus)
            nodes: List of Node objects
            services: List of Service objects
            current_allocation: Current service-to-node mapping
            **kwargs: Additional arguments specific to the implementation
            
        Returns:
            True if successful, False otherwise
        """
        pass

    # Common utility methods available to all subclasses

    def _infer_layer_prefix(self, identifier: Optional[str]) -> Optional[str]:
        """Infer a layer prefix from an element identifier like 'cloud-cpu1'."""
        if not identifier or '-' not in identifier:
            return None
        prefix = identifier.split('-', 1)[0].lower()
        # Normalize custom prefixes to known layers
        if prefix == 'frank':
            return 'edge'
        return prefix

    def _extract_index(self, identifier: Optional[str]) -> Optional[int]:
        """Extract a trailing integer index from identifiers like 'fog-cpu2' or 'edge-node-3'."""
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

    def _normalize_node_id(self, identifier: Optional[str]) -> Optional[str]:
        """Normalize a node identifier for matching.

        - Lowercase
        - Drop port (host:port)
        - Drop domain (split by '.') to keep short hostname

        Args:
            identifier: Raw node identifier from Prometheus or XML

        Returns:
            Normalized node id string or None
        """
        if not identifier:
            return None
        try:
            ident = str(identifier).strip().lower()
            # remove port if present
            if ':' in ident:
                ident = ident.split(':', 1)[0]
            # remove domain suffix if present
            if '.' in ident:
                ident = ident.split('.', 1)[0]
            return ident
        except Exception:
            return identifier

    def _match_host_to_node(self, host_id: str, node_metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
        """
        Best-effort matching from platform host_id to collected node metrics.

        Priority:
        1) direct substring/equality (case-insensitive)
        2) same layer prefix + same numeric index
        3) layer label match (blaze.layer) with same numeric index
        4) layer label match

        Args:
            host_id: Host ID from platform.xml
            node_metrics: Dictionary of node metrics keyed by node name

        Returns:
            The first best candidate node name or None
        """
        host_lower = host_id.lower()

        # 1) direct contains/equality match (case-insensitive)
        for node_name in node_metrics:
            node_lower = node_name.lower()
            if host_lower == node_lower or host_lower in node_lower or node_lower in host_lower:
                return node_name

        host_prefix = self._infer_layer_prefix(host_id)
        host_idx = self._extract_index(host_id)

        best_prefix_match = None
        best_layer_match = None

        for node_name in node_metrics:
            node_prefix = self._infer_layer_prefix(node_name)
            node_idx = self._extract_index(node_name)
            node_layer = node_metrics.get(node_name, {}).get('layer')

            # 2) prefix + same index
            if host_prefix and node_prefix == host_prefix and host_idx is not None and node_idx == host_idx:
                return node_name

            # 3) layer label + same index
            if node_layer and host_prefix and node_layer == host_prefix and host_idx is not None and node_idx == host_idx:
                return node_name

            # 4) prefix only (fallback)
            if host_prefix and node_prefix == host_prefix and not best_prefix_match:
                best_prefix_match = node_name

            # 5) layer label only (fallback)
            if node_layer and host_prefix and node_layer == host_prefix and not best_layer_match:
                best_layer_match = node_name

        return best_prefix_match or best_layer_match

    def _normalize_instances_elements(self, root: ET.Element) -> None:
        """Ensure each service has exactly one <instances> element."""
        for service in root.findall('service'):
            instances_blocks = service.findall('instances')
            if len(instances_blocks) <= 1:
                continue

            primary = instances_blocks[0]
            for extra in instances_blocks[1:]:
                for inst in list(extra.findall('container')):
                    primary.append(inst)
                service.remove(extra)

    def _remove_empty_instances_blocks(self, root: ET.Element) -> None:
        """Remove empty <instances> blocks from all services."""
        for service in root.findall('service'):
            instances_blocks = service.findall('instances')
            for instances in instances_blocks:
                # If this block has no children, remove it
                if len(list(instances)) == 0:
                    service.remove(instances)

    def _should_prune_layer(self, layer_prefix: Optional[str], allowed_layers: set) -> bool:
        """True when the prefix is a known layer and not part of the configured layers."""
        if not layer_prefix:
            return False
        return layer_prefix in self.DEFAULT_LAYER_PREFIXES and layer_prefix not in allowed_layers

    def _prune_inactive_layers(self, root: ET.Element, allowed_layers: set) -> Dict[str, Any]:
        """
        Remove hosts/links/routes/routers for layers that are not configured.

        Args:
            root: XML root element
            allowed_layers: Set of layer prefixes to keep

        Returns:
            Dictionary with counts of removed elements by type
        """
        allowed_layers = {l.lower() for l in allowed_layers}
        disallowed_layers = {p for p in self.DEFAULT_LAYER_PREFIXES if p not in allowed_layers}

        result = {
            'hosts': 0,
            'links': 0,
            'routes': 0,
            'routers': 0,
            'removed_layers': [],
            'total_removed': 0
        }

        if not disallowed_layers:
            return result

        removed_layer_hits = set()

        # Remove hosts for disallowed layers
        for zone in root.findall('.//zone'):
            for host in list(zone.findall('host')):
                layer_prefix = self._infer_layer_prefix(host.get('id'))
                if self._should_prune_layer(layer_prefix, allowed_layers):
                    zone.remove(host)
                    result['hosts'] += 1
                    if layer_prefix:
                        removed_layer_hits.add(layer_prefix)

        # Remove routers tied to disallowed layers
        for zone in root.findall('.//zone'):
            for router in list(zone.findall('router')):
                layer_prefix = self._infer_layer_prefix(router.get('id'))
                if self._should_prune_layer(layer_prefix, allowed_layers):
                    zone.remove(router)
                    result['routers'] += 1
                    if layer_prefix:
                        removed_layer_hits.add(layer_prefix)

        # Remove links belonging to disallowed layers
        for zone in root.findall('.//zone'):
            for link in list(zone.findall('link')):
                layer_prefix = self._infer_layer_prefix(link.get('id'))
                if self._should_prune_layer(layer_prefix, allowed_layers):
                    zone.remove(link)
                    result['links'] += 1
                    if layer_prefix:
                        removed_layer_hits.add(layer_prefix)

        # Remove routes that reference removed hosts/routers
        for zone in root.findall('.//zone'):
            for route in list(zone.findall('route')):
                src = route.get('src', '')
                dst = route.get('dst', '')
                src_prefix = self._infer_layer_prefix(src)
                dst_prefix = self._infer_layer_prefix(dst)
                if (self._should_prune_layer(src_prefix, allowed_layers) or
                    self._should_prune_layer(dst_prefix, allowed_layers)):
                    zone.remove(route)
                    result['routes'] += 1

        result['removed_layers'] = list(removed_layer_hits)
        result['total_removed'] = result['hosts'] + result['links'] + result['routes'] + result['routers']
        return result

    def _indent_xml(self, elem, level=0):
        """
        Add pretty-printing indentation to XML elements.

        Args:
            elem: XML element to indent
            level: Current indentation level
        """
        indent = "\n" + ("  " * level)
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for i, child in enumerate(elem):
                self._indent_xml(child, level + 1)
                if not child.tail or not child.tail.strip():
                    child.tail = indent + "  " if i < len(elem) - 1 else indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    def _get_allowed_layers(self, node_metrics: Optional[Dict[str, Dict[str, float]]] = None) -> set:
        """
        Return a lowercase set of configured layers.

        Base implementation infers from node metrics if available.
        Subclasses can override to add source-specific logic.

        Args:
            node_metrics: Optional node metrics to infer layers from

        Returns:
            Set of allowed layer prefixes
        """
        allowed = set()

        # Infer from observed node metrics if present
        if node_metrics:
            for node_name in node_metrics.keys():
                prefix = self._infer_layer_prefix(node_name)
                if prefix and prefix in self.DEFAULT_LAYER_PREFIXES:
                    allowed.add(prefix)

        return allowed
