"""Platform XML updater - handles platform.xml updates with Prometheus metrics."""
from datetime import datetime
from typing import Dict, Optional
import xml.etree.ElementTree as ET



class PlatformXMLUpdater:
    """Handles updating platform.xml with node metrics from Prometheus."""

    def __init__(self, base_updater):
        """
        Initialize with a reference to the base updater for utility methods.
        
        Args:
            base_updater: BaseXMLUpdater instance with utility methods
        """
        self.base_updater = base_updater

    def update(self, xml_path: str, node_metrics: Dict[str, Dict[str, float]], 
               allowed_layers: set, update_path_override: Optional[str] = None) -> bool:
        """
        Update platform.xml with real node metrics.

        Creates platform_update.xml with new values.

        Args:
            xml_path: Path to platform.xml file
            node_metrics: Dictionary mapping node names to their metrics
            allowed_layers: Set of layer names that should be kept

        Returns:
            True if successful, False otherwise
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        try:
            # Parse existing XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Create a copy for the update file
            import copy
            new_tree = copy.deepcopy(tree)
            new_root = new_tree.getroot()

            # Update host elements in the new tree
            updated_count = 0
            has_changes = False

            # Prune inactive layers
            if allowed_layers:
                prune_result = self.base_updater._prune_inactive_layers(new_root, allowed_layers)
                if prune_result.get('total_removed', 0) > 0:
                    has_changes = True
                    BlazeLogger.get_instance().debug(
                        "Pruned inactive layers from platform: removed %s hosts, %s links, %s routes, %s routers (layers pruned: %s)",
                        prune_result.get('hosts', 0),
                        prune_result.get('links', 0),
                        prune_result.get('routes', 0),
                        prune_result.get('routers', 0),
                        ','.join(prune_result.get('removed_layers', []))
                    )

            # Update each host with metrics
            for zone in new_root.findall('.//zone'):
                for host in zone.findall('host'):
                    host_id = host.get('id')
                    matched_node = self.base_updater._match_host_to_node(host_id, node_metrics)

                    if matched_node:
                        if self._update_host_metrics(host, host_id, matched_node, node_metrics):
                            updated_count += 1
                            has_changes = True
                        if self._ensure_debug_props(host):
                            has_changes = True
                    else:
                        # Stamp layer even without metrics
                        if self._stamp_layer_prop(host, host_id, None):
                            has_changes = True
                        if self._ensure_debug_props(host):
                            has_changes = True

            # ADD MISSING NODES: If nodes exist in node_metrics but not in platform.xml, add them
            existing_hosts = {host.get('id') for zone in new_root.findall('.//zone') for host in zone.findall('host')}
            missing_nodes = set(node_metrics.keys()) - existing_hosts
            
            if missing_nodes:
                BlazeLogger.get_instance().debug(f"Adding {len(missing_nodes)} discovered nodes to platform: {', '.join(sorted(missing_nodes))}")
                added_count = self._add_missing_nodes_to_platform(new_root, missing_nodes, node_metrics, allowed_layers)
                updated_count += added_count
                has_changes = True

            # Write update file
            update_path = update_path_override or xml_path.replace('.xml', '_update.xml')
            self._write_update_file(new_tree, new_root, update_path, tree, updated_count, has_changes)

            return True

        except Exception as e:
            BlazeLogger.get_instance().error(f"Failed to update platform.xml: {e}")
            return False

    def _update_host_metrics(self, host: ET.Element, host_id: str, matched_node: str,
                            node_metrics: Dict[str, Dict[str, float]]) -> bool:
        """Update a single host element with metrics. Returns True if changed."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        metrics = node_metrics[matched_node]
        node_changed = False

        # Stamp layer property
        if self._stamp_layer_prop(host, host_id, metrics):
            node_changed = True

        # Update CPU cores (keep speed from original platform.xml)
        if 'cpu_cores' in metrics:
            new_cores = str(int(metrics['cpu_cores']))
            old_cores = host.get('core', '')
            if new_cores != old_cores:
                host.set('core', new_cores)
                node_changed = True

        # Update RAM
        if 'ram_gb' in metrics:
            if self._update_property(host, 'ram.GB', f"{metrics['ram_gb']:.1f}"):
                node_changed = True

        # Update GPU count
        if 'gpu_count' in metrics and metrics['gpu_count'] > 0:
            if self._update_property(host, 'gpu.count', str(int(metrics['gpu_count']))):
                node_changed = True

        # Update schedulable property (for orchestration filtering)
        if 'schedulable' in metrics:
            schedulable_value = 'true' if metrics['schedulable'] else 'false'
            if self._update_property(host, 'schedulable', schedulable_value):
                node_changed = True

        # Add node usage debugging info (for debugging, doesn't affect simulation)
        if 'cpu_used' in metrics:
            if self._update_property(host, 'debug.cpu_used', f"{metrics['cpu_used']:.3f}"):
                node_changed = True
        if 'ram_used_gb' in metrics:
            if self._update_property(host, 'debug.ram_used_gb', f"{metrics['ram_used_gb']:.3f}"):
                node_changed = True
        if 'gpu_used_gb' in metrics:
            if self._update_property(host, 'debug.gpu_used_gb', f"{metrics['gpu_used_gb']:.3f}"):
                node_changed = True

        if node_changed:
            BlazeLogger.get_instance().debug(f"Changed host {host_id} with updated metrics")
        
        return node_changed

    def _ensure_debug_props(self, host: ET.Element) -> bool:
        """Ensure debug.* props exist (initialized to 0). Returns True if any added."""
        changed = False
        if host.find(".//prop[@id='debug.cpu_used']") is None:
            prop = ET.SubElement(host, 'prop')
            prop.set('id', 'debug.cpu_used')
            prop.set('value', '0.000')
            changed = True
        if host.find(".//prop[@id='debug.ram_used_gb']") is None:
            prop = ET.SubElement(host, 'prop')
            prop.set('id', 'debug.ram_used_gb')
            prop.set('value', '0.000')
            changed = True
        if host.find(".//prop[@id='gpu.count']") is not None and host.find(".//prop[@id='debug.gpu_used_gb']") is None:
            prop = ET.SubElement(host, 'prop')
            prop.set('id', 'debug.gpu_used_gb')
            prop.set('value', '0.000')
            changed = True
        return changed

    def _stamp_layer_prop(self, host: ET.Element, host_id: str, 
                         metrics: Optional[Dict[str, float]]) -> bool:
        """Stamp layer property on host. Returns True if changed."""
        layer_prop = host.find(".//prop[@id='layer']")
        
        # Determine layer value
        if metrics and 'layer' in metrics:
            layer_value = metrics['layer']
        else:
            layer_value = self.base_updater._infer_layer_prefix(host_id)
        
        # Map frank layer to edge (frank nodes are edge devices)
        if layer_value == 'frank':
            layer_value = 'edge'
        
        if not layer_value:
            return False

        # Create or update layer prop
        if layer_prop is None:
            layer_prop = ET.SubElement(host, 'prop')
            layer_prop.set('id', 'layer')
            layer_prop.set('value', str(layer_value))
            return True
        elif layer_prop.get('value') != str(layer_value):
            layer_prop.set('value', str(layer_value))
            return True
        
        return False

    def _update_property(self, host: ET.Element, prop_id: str, new_value: str) -> bool:
        """Update or create a property on a host. Returns True if changed."""
        prop = host.find(f".//prop[@id='{prop_id}']")
        
        if prop is not None:
            old_value = prop.get('value', '')
            if new_value != old_value:
                prop.set('value', new_value)
                return True
        else:
            # Create property
            prop = ET.SubElement(host, 'prop')
            prop.set('id', prop_id)
            prop.set('value', new_value)
            return True
        
        return False

    def _write_update_file(self, new_tree: ET.ElementTree, new_root: ET.Element,
                          update_path: str, original_tree: ET.ElementTree,
                          updated_count: int, has_changes: bool):
        """Write the update file with proper formatting."""
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        # Add metadata comment
        comment = ET.Comment(f" Updated from Prometheus on {datetime.now().isoformat()} ")
        new_root.insert(0, comment)

        # Apply indentation
        self.base_updater._indent_xml(new_root)

        # Safety: if no hosts remain, fall back to original
        host_count = len(new_root.findall('.//host'))
        if host_count == 0:
            BlazeLogger.get_instance().warning("No hosts found after update; writing original platform to _update.xml to avoid empty file")
            orig_root = original_tree.getroot()
            self.base_updater._indent_xml(orig_root)
            self._write_with_doctype(orig_root, update_path)
            return

        # Write the update file with SimGrid-compatible prolog/doctype
        self._write_with_doctype(new_root, update_path)

        if has_changes:
            BlazeLogger.get_instance().info(f"✓ Created {update_path} with metrics for {updated_count} hosts (review before applying)")
        else:
            BlazeLogger.get_instance().debug(f"No changes detected, but {update_path} still created for review")

    def _add_missing_nodes_to_platform(self, root: ET.Element, missing_nodes: set,
                                      node_metrics: Dict[str, Dict[str, float]],
                                      allowed_layers: set) -> int:
        """
        Add nodes that were discovered but don't exist in platform.xml.
        Groups them by layer and adds them to the world zone.
        
        Args:
            root: Root element of the XML tree
            missing_nodes: Set of node IDs to add
            node_metrics: Metrics for all nodes
            allowed_layers: Set of allowed layer names
            
        Returns:
            Number of nodes added
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        added_count = 0
        world_zone = root.find(".//zone[@id='world']")
        
        if world_zone is None:
            BlazeLogger.get_instance().warning("Could not find 'world' zone to add missing nodes")
            return 0
        
        # Group missing nodes by layer (infer layer from name if missing)
        nodes_by_layer = {}
        for node_id in missing_nodes:
            metrics = node_metrics.get(node_id, {})
            layer = metrics.get('layer')

            # Infer layer from node prefix when not provided by metrics (e.g., cloud-* from Prometheus without layer labels)
            if not layer:
                layer = self.base_updater._infer_layer_prefix(node_id) or 'unknown'
            layer = layer.lower()
            
            # Skip if layer is not allowed
            if allowed_layers and layer not in allowed_layers:
                continue
                
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(node_id)
        
        # Add nodes to world zone
        for layer in sorted(nodes_by_layer.keys()):
            node_ids = nodes_by_layer[layer]
            for node_id in sorted(node_ids):
                metrics = node_metrics[node_id]
                
                # Create host element
                host = ET.Element('host')
                host.set('id', node_id)
                
                # Set compute properties from metrics
                if 'cpu_cores' in metrics:
                    host.set('core', str(int(metrics['cpu_cores'])))
                else:
                    host.set('core', '8')  # default
                
                # Set speed (GHz) - default to 50Gf if not available
                speed_gf = metrics.get('cpu_freq_ghz', 50.0)
                host.set('speed', f"{speed_gf:.0f}Gf")
                
                # Add properties
                props_to_add = {
                    'layer': layer,
                    'schedulable': 'true' if metrics.get('schedulable', True) else 'false',
                    'ram.GB': f"{metrics.get('ram_gb', 8.0):.1f}",
                    'debug.cpu_used': f"{metrics.get('cpu_used', 0.0):.3f}",
                    'debug.ram_used_gb': f"{metrics.get('ram_used_gb', 0.0):.3f}",
                }
                
                if metrics.get('gpu_count', 0) > 0:
                    props_to_add['gpu.count'] = str(int(metrics['gpu_count']))
                    props_to_add['gpu.memGB'] = str(int(metrics.get('gpu_mem_gb', 8)))
                    props_to_add['debug.gpu_used_gb'] = f"{metrics.get('gpu_used_gb', 0.0):.3f}"
                
                for prop_id, prop_value in props_to_add.items():
                    prop = ET.SubElement(host, 'prop')
                    prop.set('id', prop_id)
                    prop.set('value', prop_value)
                
                # Add to world zone
                world_zone.append(host)
                added_count += 1
                BlazeLogger.get_instance().debug(f"  Added node {node_id} (layer={layer}) to platform")
        
        return added_count

    def _write_with_doctype(self, root: ET.Element, path: str) -> None:
        """Write XML with XML declaration and SimGrid 4.1 DOCTYPE."""
        xml_body = ET.tostring(root, encoding='utf-8')
        prolog = b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        doctype = b"<!DOCTYPE platform SYSTEM \"https://simgrid.org/simgrid.dtd\">\n"
        with open(path, "wb") as f:
            f.write(prolog)
            f.write(doctype)
            f.write(xml_body)
