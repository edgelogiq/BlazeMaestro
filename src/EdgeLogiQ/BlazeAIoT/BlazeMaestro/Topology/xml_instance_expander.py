"""
from BlazeAIoT.Core.Utils import BlazeLogger
Expand services XML to explicit per-instance container entries.

Generates <instances><container ...> blocks from service definitions using:
- pod_node_mapping: service_id -> {pod_name: node_name}
- current_allocation: service_id -> node_name (fallback)
- service container templates for cpu/ram attributes

This keeps services.xml intact while adding runtime instance info into the
output XML (often same file) for Prometheus metric enrichment.
"""
import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional
from BlazeAIoT.Core.Utils import BlazeLogger


def _ensure_instances_element(service_elem: ET.Element) -> ET.Element:
    instances = service_elem.find('instances')
    if instances is None:
        instances = ET.SubElement(service_elem, 'instances')
    return instances


def _harvest_existing_metrics(instances: ET.Element) -> Dict[str, Dict[str, str]]:
    """Capture existing metrics by pod_name or node+container id to preserve history."""
    metrics_map: Dict[str, Dict[str, str]] = {}
    for c in instances.findall('container'):
        key = c.get('pod_name') or f"{c.get('node','')}::{c.get('id','')}"
        metrics_map[key] = {
            k: v for k, v in c.attrib.items()
            if k in (
                'cpu_usage_min', 'cpu_usage_max', 'cpu_usage_last',
                'ram_usage_min_gb', 'ram_usage_max_gb', 'ram_usage_last_gb'
            )
        }
    return metrics_map


def _copy_container_template(template: ET.Element) -> ET.Element:
    # shallow copy of container element with its attributes, excluding static capacity specs
    attrib = {}
    for k, v in template.attrib.items():
        # Skip cpu and ram_gb (template specs, not instance metrics)
        # Keep exec_time - it can vary per instance
        if k not in ('cpu', 'ram_gb'):
            attrib[k] = v
    new_elem = ET.Element('container', attrib=attrib)
    return new_elem


def expand_services_to_instances(
    input_xml_path: str,
    nodes_by_layer: Dict[str, Any],
    output_xml_path: str,
    metrics_provider: Optional[Any] = None,
    current_allocation: Optional[Dict[str, str]] = None,
    pod_node_mapping: Optional[Dict[str, Dict[str, str]]] = None,
):
    """Expand services into per-instance containers and write to output XML."""
    pod_node_mapping = pod_node_mapping or {}
    current_allocation = current_allocation or {}

    if not os.path.exists(input_xml_path):
        raise FileNotFoundError(f"services XML not found: {input_xml_path}")

    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    for service in root.findall('service'):
        service_id = service.get('id') or ''
        instances_elem = _ensure_instances_element(service)

        # capture existing instances (with metrics) to preserve historical data
        existing_containers = []
        for c in instances_elem.findall('container'):
            existing_containers.append({
                'elem': c,
                'node': c.get('node'),
                'pod_name': c.get('pod_name'),
                'container_id': c.get('id'),
                'metrics': {
                    k: v for k, v in c.attrib.items()
                    if k in (
                        'cpu_usage_min', 'cpu_usage_max', 'cpu_usage_last',
                        'ram_usage_min_gb', 'ram_usage_max_gb', 'ram_usage_last_gb',
                        'exec_time', 'execution_rate_hz'  # Preserve timing metrics per instance
                    )
                }
            })

        # build list of new instance targets from current K8s allocation
        targets = []
        pod_map = pod_node_mapping.get(service_id, {})
        if pod_map:
            for pod_name, pod_info in pod_map.items():
                # Handle both old format (string) and new format (dict with layer)
                if isinstance(pod_info, dict):
                    targets.append({
                        'node': pod_info.get('node'),
                        'pod_name': pod_name,
                        'layer': pod_info.get('layer')
                    })
                else:
                    # Legacy: pod_info is just the node name string
                    targets.append({'node': pod_info, 'pod_name': pod_name})
        elif service_id in current_allocation:
            targets.append({'node': current_allocation[service_id], 'pod_name': None})

        # CLEAR ALL OLD INSTANCES FIRST - rebuild fresh from K8s state
        for old_elem in list(instances_elem.findall('container')):
            instances_elem.remove(old_elem)

        # Add only current/live instances from K8s pod discovery
        container_templates = service.find('containers')
        if container_templates is not None:
            # Track pod names present now for this service
            present_pod_names = set(pod_map.keys()) if pod_map else set()
            
            # Track which old containers have been matched to current targets
            matched_old_containers = set()

            # 1) Add current/live instances
            for tmpl in container_templates.findall('container'):
                container_id = tmpl.get('id')
                for tgt in targets:
                    inst = _copy_container_template(tmpl)
                    if tgt['node']:
                        inst.set('node', tgt['node'])
                    if tgt['pod_name']:
                        inst.set('pod_name', tgt['pod_name'])
                    if tgt.get('layer'):
                        inst.set('layer', tgt['layer'])

                    # Try to find matching metrics from existing instance
                    # First try by pod_name (most reliable)
                    metrics_found = False
                    pod_key = tgt['pod_name'] if tgt['pod_name'] else None
                    if pod_key:
                        for idx, old in enumerate(existing_containers):
                            if old['pod_name'] == pod_key:
                                for k, v in old['metrics'].items():
                                    inst.set(k, v)
                                metrics_found = True
                                matched_old_containers.add(idx)
                                break
                    
                    # If no pod_name match, try node+container match (for renamed pods on same node)
                    if not metrics_found and tgt['node']:
                        for idx, old in enumerate(existing_containers):
                            if old['node'] == tgt['node'] and old['container_id'] == container_id and not old['pod_name']:
                                # Only match historical entries without pod_name
                                for k, v in old['metrics'].items():
                                    inst.set(k, v)
                                metrics_found = True
                                matched_old_containers.add(idx)
                                break

                    instances_elem.append(inst)

            # 2) Preserve historical instances for old containers that were not matched to current targets
            # This preserves pods that have disappeared
            for idx, old in enumerate(existing_containers):
                if idx in matched_old_containers:
                    continue  # Already handled above
                
                # Find matching template by container id
                tmpl = next((t for t in container_templates.findall('container') if t.get('id') == old['container_id']), None)
                if tmpl is None:
                    continue
                
                inst = _copy_container_template(tmpl)
                # Preserve node (historical), remove pod_name
                if old['node']:
                    inst.set('node', old['node'])
                # Do NOT set pod_name; intentionally left out to indicate historical entry
                # Preserve layer if it existed on old element
                old_layer = old['elem'].get('layer') if old['elem'] is not None else None
                if old_layer:
                    inst.set('layer', old_layer)
                # Copy historical metrics
                for k, v in old['metrics'].items():
                    inst.set(k, v)
                # Avoid duplicate of same node+container if already added above
                duplicate = False
                for existing in instances_elem.findall('container'):
                    # Match by pod_name (primary) or node+container_id (fallback for historical entries)
                    existing_pod = existing.get('pod_name')
                    inst_pod = inst.get('pod_name')
                    if existing_pod and inst_pod and existing_pod == inst_pod:
                        duplicate = True
                        break
                    # For historical entries without pod_name, check node match
                    if not existing_pod and not inst_pod and existing.get('node') == inst.get('node'):
                        duplicate = True
                        break
                if not duplicate:
                    instances_elem.append(inst)

    # Pretty-print the XML before writing
    _indent_xml(root)
    
    # write output
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)

    BlazeLogger.get_instance().debug("Expanded services into instances: %s", output_xml_path)


def _indent_xml(elem, level=0):
    """Add proper indentation to XML elements for readability."""
    indent = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for i, child in enumerate(elem):
            _indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent + "  " if i < len(elem) - 1 else indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
