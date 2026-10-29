"""
Instance Deduplication Module.

Removes duplicate container instances in services.xml where the same pod
appears multiple times on the same node (e.g., with IP vs hostname).

Key rule: After IP resolution, each (pod_name, node) pair should be unique.
"""

import xml.etree.ElementTree as ET
from typing import Dict
from BlazeAIoT.Core.Utils import BlazeLogger


class InstanceDeduplicator:
    """Removes duplicate container instances from XML."""
    
    def __init__(self, node_resolver=None):
        """
        Initialize deduplicator.
        
        Args:
            node_resolver: NodeResolver instance for IP-to-name resolution
        """
        self.logger = BlazeLogger.get_instance()
        self.node_resolver = node_resolver
    
    def deduplicate(self, root: ET.Element) -> Dict[str, int]:
        """
        Remove duplicate instances across all services.
        
        Duplicates are defined as: same pod_name + same node (case-insensitive).
        When duplicates found, keeps first instance and merges non-zero metrics.
        
        Args:
            root: XML root element
            
        Returns:
            Dictionary with service_id → count of removed duplicates
        """
        self.logger.info("[Deduplicator] Starting deduplication")
        
        removal_stats = {}
        
        for service in root.findall('service'):
            instances = service.find('instances')
            if instances is None:
                continue
            
            service_id = service.get('id')
            removed = self._deduplicate_service(service_id, instances)
            
            if removed > 0:
                removal_stats[service_id] = removed
                self.logger.info(f"[Deduplicator] {service_id}: Removed {removed} duplicates")
        
        total_removed = sum(removal_stats.values())
        self.logger.info(f"[Deduplicator] ✓ Total removed: {total_removed} duplicates across {len(removal_stats)} services")
        
        return removal_stats
    
    def _deduplicate_service(self, service_id: str, instances: ET.Element) -> int:
        """
        Deduplicate instances for a single service.
        
        Resolves node IPs to names before comparison to catch duplicates like:
        - (pod-123, "edge-gpu2") and (pod-123, "172.16.200.25")
        
        Args:
            service_id: Service identifier (for logging)
            instances: XML instances element
            
        Returns:
            Number of duplicates removed
        """
        # Get IP resolution mapping if available
        mapping = {}
        if self.node_resolver:
            mapping = self.node_resolver._cache or {}
        
        # Group containers by (pod_name, resolved_node_lowercase)
        seen_pairs = {}  # (pod_name, resolved_node_lower) → container_element
        containers_to_remove = []
        
        for container in instances.findall('container'):
            pod_name = container.get('pod_name')
            node_raw = container.get('node', '')
            
            if not pod_name or not node_raw:
                continue
            
            # Resolve IP to name before comparison
            node_resolved = mapping.get(node_raw) or mapping.get(node_raw.lower()) or node_raw
            node_lower = node_resolved.lower()
            
            pair = (pod_name, node_lower)
            
            if pair in seen_pairs:
                # Duplicate found!
                existing = seen_pairs[pair]
                self.logger.warning(f"[Deduplicator] {service_id}: Duplicate {pod_name}@{node_resolved}")
                
                # Merge non-zero metrics from duplicate into existing
                self._merge_metrics(existing, container)
                
                # Mark for removal
                containers_to_remove.append(container)
            else:
                # First occurrence - keep it
                seen_pairs[pair] = container
        
        # Remove all duplicates from XML
        for container in containers_to_remove:
            instances.remove(container)
        
        return len(containers_to_remove)
    
    def _merge_metrics(self, target: ET.Element, source: ET.Element):
        """
        Merge metrics from source into target, preferring non-zero values.
        
        Args:
            target: Container element to keep (receives merged metrics)
            source: Container element to remove (provides metrics if better)
        """
        metrics = [
            'cpu_usage_min', 'cpu_usage_max', 'cpu_usage_last',
            'ram_usage_min_gb', 'ram_usage_max_gb', 'ram_usage_last_gb',
            'exec_time', 'execution_rate_hz'
        ]
        
        for metric in metrics:
            try:
                target_val = float(target.get(metric, 0))
                source_val = float(source.get(metric, 0))
                
                # If source has data and target doesn't, use source value
                if source_val > 0 and target_val == 0:
                    target.set(metric, str(source_val))
                    self.logger.debug(f"[Deduplicator] Merged {metric}: {source_val}")
            except (ValueError, TypeError):
                # Skip non-numeric values
                pass
