"""
Service definition loader from XML configuration.

This module provides functionality to load service definitions from an XML file,
similar to how platform.xml defines the infrastructure topology.
"""

import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service, ContainerSpec, DependencySpec


class ServiceDefinitionLoader:
    """
    Load service definitions from XML configuration file.

    XML Format:
    <services>
        <service id="service-name" execution_rate_hz="10.0">
            <placement type="layer_lock" layer="cloud" />
            <!-- or -->
            <placement type="node_lock" node="node-1" />
            <!-- or -->
            <placement type="daemonset" layer="edge" />
            <!-- or -->
            <placement type="flexible" />

            <containers>
                <container id="container-1" cpu="2.0" ram_gb="4.0" gpu_gb="0.0" exec_time="0.01" />
            </containers>

            <dependencies>
                <dependency service_id="other-service" size_mb="1.0" />
            </dependencies>
        </service>
    </services>
    """

    def __init__(self, xml_path: str, simpy_env=None):
        """
        Initialize the service definition loader.

        Args:
            xml_path: Path to the services XML file
            simpy_env: Optional SimPy environment for ServiceSimulator instances
        """
        self.xml_path = xml_path
        self.simpy_env = simpy_env

        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Services XML file not found: {xml_path}")

    def load_services(self) -> List[Service]:
        """
        Load all service definitions from the XML file.

        Returns:
            List of Service objects
        """
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        if root.tag != 'services':
            raise ValueError(f"Root element must be 'services', got '{root.tag}'")

        services = []
        for service_elem in root.findall('service'):
            service = self._parse_service(service_elem)
            if service:
                services.append(service)

        return services

    def _parse_service(self, service_elem: ET.Element) -> Optional[Service]:
        """
        Parse a single service element.

        Args:
            service_elem: XML element for the service

        Returns:
            Service object or None if parsing fails
        """
        # Get basic attributes
        service_id = service_elem.get('id')
        if not service_id:
            print(f"Warning: Service element missing 'id' attribute, skipping")
            return None

        # Parse instance usage once (also includes timing hints if present)
        usage = self._parse_instance_usage(service_elem)
        usage_by_node = self._parse_instance_usage_by_node(service_elem)

        execution_rate_attr = service_elem.get('execution_rate_hz')
        if execution_rate_attr is not None:
            execution_rate_hz = float(execution_rate_attr)
        else:
            # Fall back to instance-derived execution_rate_hz if provided
                execution_rate_hz = float(usage.get('execution_rate_hz_avg') or usage.get('execution_rate_hz_max') or 1.0)

        # Parse containers
        containers = self._parse_containers(service_elem, usage)
        if not containers:
            print(f"Warning: Service '{service_id}' has no containers, skipping")
            return None

        # Parse dependencies
        dependencies = self._parse_dependencies(service_elem)

        # Parse placement
        placement = self._parse_placement(service_elem)

        control_loop = (service_elem.get('control_loop', 'false').lower() in ('true', '1', 'yes'))
        exclude_from_orchestration = (service_elem.get('exclude_from_orchestration', 'false').lower() in ('true', '1', 'yes'))

        # Create ServiceSimulator if simpy_env provided, otherwise plain Service
        if self.simpy_env is not None:
            from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.service_simulator import ServiceSimulator
            service = ServiceSimulator(
                env=self.simpy_env,
                service_id=service_id,
                containers=containers,
                dependencies=dependencies if dependencies else None
            )
            # Apply placement and rate settings
            service.execution_rate_hz = execution_rate_hz
            for k, v in placement.items():
                setattr(service, k, v)
            service.control_loop = control_loop
            service.exclude_from_orchestration = exclude_from_orchestration
        else:
            service = Service(
                service_id=service_id,
                containers=containers,
                dependencies=dependencies if dependencies else None,
                execution_rate_hz=execution_rate_hz,
                control_loop=control_loop,
                **placement
            )
            service.exclude_from_orchestration = exclude_from_orchestration

        # Attach per-node usage map for node-specific feasibility checks
        if usage_by_node:
            setattr(service, "instance_usage_by_node", usage_by_node)

        # A service excluded from orchestration is not moved, but its deadline still depends on
        # where its upstream services run: pin it to the node it was measured on so that V(x)
        # checks its period like any other node-locked service.
        if (exclude_from_orchestration and not getattr(service, "is_daemonset", False)
                and getattr(service, "locked_node_id", None) is None and usage_by_node):
            service.locked_node_id = next(iter(usage_by_node))

        return service

    def _parse_instance_usage_by_node(self, service_elem: ET.Element) -> Dict[str, Dict[str, Optional[float]]]:
        """Build per-node usage metrics from <instances> (exec_time, execution_rate_hz, cpu/ram max)."""
        instances_elem = service_elem.find('instances')
        if instances_elem is None:
            return {}

        per_node: Dict[str, Dict[str, Optional[float]]] = {}
        for inst in instances_elem.findall('container'):
            node = inst.get('node')
            if not node:
                continue
            entry = per_node.setdefault(node, {
                "exec_time_max": None,
                "execution_rate_hz_max": None,
                "cpu_usage_max": None,
                "ram_usage_max_gb": None,
            })

            exec_time = inst.get('exec_time')
            if exec_time:
                try:
                    val = float(exec_time)
                    if entry["exec_time_max"] is None or val > entry["exec_time_max"]:
                        entry["exec_time_max"] = val
                except Exception:
                    pass

            exec_rate = inst.get('execution_rate_hz')
            if exec_rate:
                try:
                    val = float(exec_rate)
                    if entry["execution_rate_hz_max"] is None or val > entry["execution_rate_hz_max"]:
                        entry["execution_rate_hz_max"] = val
                except Exception:
                    pass

            cpu_max = inst.get('cpu_usage_max')
            if cpu_max:
                try:
                    val = float(cpu_max)
                    if entry["cpu_usage_max"] is None or val > entry["cpu_usage_max"]:
                        entry["cpu_usage_max"] = val
                except Exception:
                    pass

            ram_max = inst.get('ram_usage_max_gb')
            if ram_max:
                try:
                    val = float(ram_max)
                    if entry["ram_usage_max_gb"] is None or val > entry["ram_usage_max_gb"]:
                        entry["ram_usage_max_gb"] = val
                except Exception:
                    pass

        return per_node

    def _parse_containers(self, service_elem: ET.Element, usage: Optional[Dict[str, Optional[float]]] = None) -> List[ContainerSpec]:
        """
        Parse container specifications.

        Args:
            service_elem: XML element for the service

        Returns:
            List of ContainerSpec objects
        """
        containers = []
        containers_elem = service_elem.find('containers')

        if containers_elem is None:
            return containers

        if usage is None:
            usage = self._parse_instance_usage(service_elem)

        for container_elem in containers_elem.findall('container'):
            container_id = container_elem.get('id')
            if not container_id:
                continue

            cpu = float(container_elem.get('cpu', '1.0'))
            ram_gb = float(container_elem.get('ram_gb', '1.0'))
            gpu_gb = float(container_elem.get('gpu_gb') or container_elem.get('gpu') or '0.0')  # GPU memory (GB)
            # Only parse exec_time if present - don't use default for non-time-constrained services
            exec_time = float(container_elem.get('exec_time')) if container_elem.get('exec_time') else None
            image_size_gb = float(container_elem.get('image_size_gb', '0.5'))
            exec_time_available = container_elem.get('exec_time_available', 'true').lower() == 'true'

            # Apply instance-derived usage metrics if available
            cpu_usage_min = usage.get("cpu_usage_min")
            cpu_usage_max = usage.get("cpu_usage_max")
            cpu_usage_last = usage.get("cpu_usage_last")
            ram_usage_min_gb = usage.get("ram_usage_min_gb")
            ram_usage_max_gb = usage.get("ram_usage_max_gb")
            ram_usage_last_gb = usage.get("ram_usage_last_gb")

            # Prefer recent observed usage over max to avoid over-constraining feasibility
            if cpu_usage_last is not None:
                cpu = float(cpu_usage_last)
            elif cpu_usage_max is not None:
                cpu = float(cpu_usage_max)

            if ram_usage_last_gb is not None:
                ram_gb = float(ram_usage_last_gb)
            elif ram_usage_max_gb is not None:
                ram_gb = float(ram_usage_max_gb)

            # If container exec_time missing, use instance-derived exec_time if available
            if exec_time is None:
                exec_time = usage.get("exec_time_max")

            container = ContainerSpec(
                id=container_id,
                cpu=cpu,
                ram_gb=ram_gb,
                gpu_gb=gpu_gb,
                exec_time=exec_time,
                image_size_gb=image_size_gb,
                exec_time_available=exec_time_available,
                cpu_usage_min=cpu_usage_min,
                cpu_usage_max=cpu_usage_max,
                cpu_usage_last=cpu_usage_last,
                ram_usage_min_gb=ram_usage_min_gb,
                ram_usage_max_gb=ram_usage_max_gb,
                ram_usage_last_gb=ram_usage_last_gb,
            )
            containers.append(container)

        return containers

    def _parse_instance_usage(self, service_elem: ET.Element) -> Dict[str, Optional[float]]:
        """Aggregate instance usage metrics from <instances> into min/max/last values."""
        instances_elem = service_elem.find('instances')
        if instances_elem is None:
            return {
                "cpu_usage_min": None,
                "cpu_usage_max": None,
                "cpu_usage_last": None,
                "ram_usage_min_gb": None,
                "ram_usage_max_gb": None,
                "ram_usage_last_gb": None,
                "exec_time_max": None,
                "exec_time_avg": None,
                "execution_rate_hz_max": None,
                "execution_rate_hz_avg": None,
            }

        cpu_mins: List[float] = []
        cpu_maxs: List[float] = []
        cpu_lasts: List[float] = []
        ram_mins: List[float] = []
        ram_maxs: List[float] = []
        ram_lasts: List[float] = []
        exec_times: List[float] = []
        exec_rates: List[float] = []

        for inst in instances_elem.findall('container'):
            cpu_min = inst.get('cpu_usage_min')
            cpu_max = inst.get('cpu_usage_max')
            cpu_last = inst.get('cpu_usage_last')
            ram_min = inst.get('ram_usage_min_gb')
            ram_max = inst.get('ram_usage_max_gb')
            ram_last = inst.get('ram_usage_last_gb')
            exec_time = inst.get('exec_time')
            exec_rate = inst.get('execution_rate_hz')

            if cpu_min:
                cpu_mins.append(float(cpu_min))
            if cpu_max:
                cpu_maxs.append(float(cpu_max))
            if cpu_last:
                cpu_lasts.append(float(cpu_last))
            if ram_min:
                ram_mins.append(float(ram_min))
            if ram_max:
                ram_maxs.append(float(ram_max))
            if ram_last:
                ram_lasts.append(float(ram_last))
            if exec_time:
                exec_times.append(float(exec_time))
            if exec_rate:
                exec_rates.append(float(exec_rate))

        def avg(vals: List[float]) -> Optional[float]:
            return float(sum(vals) / len(vals)) if vals else None

        return {
            "cpu_usage_min": min(cpu_mins) if cpu_mins else None,
            "cpu_usage_max": max(cpu_maxs) if cpu_maxs else None,
            "cpu_usage_last": avg(cpu_lasts),
            "ram_usage_min_gb": min(ram_mins) if ram_mins else None,
            "ram_usage_max_gb": max(ram_maxs) if ram_maxs else None,
            "ram_usage_last_gb": avg(ram_lasts),
            "exec_time_max": max(exec_times) if exec_times else None,
            "exec_time_avg": avg(exec_times),
            "execution_rate_hz_max": max(exec_rates) if exec_rates else None,
            "execution_rate_hz_avg": avg(exec_rates),
        }

    def _parse_dependencies(self, service_elem: ET.Element) -> List[DependencySpec]:
        """
        Parse service dependencies.

        Args:
            service_elem: XML element for the service

        Returns:
            List of DependencySpec objects
        """
        dependencies = []
        dependencies_elem = service_elem.find('dependencies')

        if dependencies_elem is None:
            return dependencies

        for dep_elem in dependencies_elem.findall('dependency'):
            dep_service_id = dep_elem.get('service_id') or dep_elem.get('service')
            if not dep_service_id:
                continue

            size_mb = float(dep_elem.get('size_mb', '0.0'))
            blocking_attr = dep_elem.get('blocking', 'true').lower()
            blocking = blocking_attr not in ('false', '0', 'no')
            feedback_attr = dep_elem.get('feedback', 'false').lower()
            feedback = feedback_attr in ('true', '1', 'yes')
            # Parse constraint_level: 'hard' = timing-critical (must fit in period),
            # 'soft' = latency-sensitive (optimize but not strict), 'none' = best-effort
            constraint_level = dep_elem.get('constraint_level', 'hard').lower()
            if constraint_level not in ('hard', 'soft', 'none'):
                constraint_level = 'hard'

            dependency = DependencySpec(
                service_id=dep_service_id,
                size_mb=size_mb,
                blocking=blocking,
                feedback=feedback,
                constraint_level=constraint_level
            )
            dependencies.append(dependency)

        return dependencies

    def _parse_placement(self, service_elem: ET.Element) -> Dict:
        """
        Parse placement configuration.

        Args:
            service_elem: XML element for the service

        Returns:
            Dictionary with placement parameters for Service constructor
        """
        placement_elem = service_elem.find('placement')

        if placement_elem is None:
            # Default: flexible placement
            return {}

        placement_type = placement_elem.get('type', 'flexible')
        
        # Normalize hyphenated types to underscored (e.g., 'layer-lock' -> 'layer_lock')
        placement_type = placement_type.replace('-', '_')

        if placement_type == 'layer_lock':
            layer_id = placement_elem.get('layer')
            if not layer_id:
                print(f"Warning: layer_lock placement missing 'layer' attribute")
                return {}
            return {'locked_layer_id': layer_id}

        elif placement_type == 'node_lock':
            node_id = placement_elem.get('node')
            if not node_id:
                print(f"Warning: node_lock placement missing 'node' attribute")
                return {}
            return {'locked_node_id': node_id}

        elif placement_type == 'daemonset':
            layer_id = placement_elem.get('layer')
            if not layer_id:
                print(f"Warning: daemonset placement missing 'layer' attribute")
                return {}
            return {
                'is_daemonset': True,
                'daemonset_layer_id': layer_id
            }

        elif placement_type == 'multi_layer':
            # Multi-layer service: spans multiple layers with per-layer rules
            layers = placement_elem.get('layers', '')
            layer_list = [l.strip() for l in layers.split(',') if l.strip()] if layers else []

            # Parse layer-rules to find constraints
            # - If a layer-lock is present, we lock to that layer (orchestrated)
            # - If a node_lock is present, we lock to that node (orchestrated)
            # - If ALL layers are daemonset (and no locks), treat service as daemonset (not orchestrated)
            layer_rules = placement_elem.find('layer-rules')
            locked_layer = None
            locked_node = None
            daemonset_layers = set()
            non_daemonset_found = False

            if layer_rules is not None:
                for layer_rule in layer_rules.findall('layer'):
                    layer_id = layer_rule.get('id')
                    rule_type = (layer_rule.get('type', 'flexible') or 'flexible').replace('-', '_')

                    if rule_type == 'layer_lock' and layer_id:
                        locked_layer = layer_id
                    elif rule_type == 'node_lock' and layer_id:
                        # node_lock in multi-layer: lock to specific node
                        node_id = layer_rule.get('node')
                        if node_id:
                            locked_node = node_id
                            non_daemonset_found = True
                    elif rule_type == 'daemonset' and layer_id:
                        daemonset_layers.add(layer_id)
                    else:
                        non_daemonset_found = True

            result = {
                'multi_layer': True,
                'layers': layer_list
            }

            # Apply locks if present (node_lock takes precedence over layer_lock)
            if locked_node:
                result['locked_node_id'] = locked_node
            elif locked_layer:
                result['locked_layer_id'] = locked_layer

            # Only mark daemonset if ALL rules are daemonset and no locks exist
            if not locked_node and not locked_layer and daemonset_layers and not non_daemonset_found:
                result['is_daemonset'] = True

            return result

        elif placement_type == 'flexible':
            return {}

        else:
            print(f"Warning: Unknown placement type '{placement_type}', using flexible")
            return {}


def load_services_from_xml(xml_path: str, simpy_env=None) -> List[Service]:
    """
    Convenience function to load services from XML file.

    Args:
        xml_path: Path to the services XML file
        simpy_env: Optional SimPy environment for ServiceSimulator instances

    Returns:
        List of Service objects
    """
    loader = ServiceDefinitionLoader(xml_path, simpy_env=simpy_env)
    return loader.load_services()


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python service_loader.py <services.xml>")
        sys.exit(1)

    xml_file = sys.argv[1]

    try:
        services = load_services_from_xml(xml_file)
        print(f"Loaded {len(services)} services from {xml_file}")

        for service in services:
            print(f"\n  {service.service_id}:")
            print(f"    Containers: {len(service.containers)}")
            print(f"    Dependencies: {len(service.dependencies)}")
            print(f"    Rate: {service.execution_rate_hz} Hz")

            if service.is_daemonset_service():
                print(f"    Placement: DaemonSet on {service.get_daemonset_layer()}")
            elif service.is_locked_to_layer():
                print(f"    Placement: Layer lock ({service.get_locked_layer()})")
            elif service.is_locked_to_node():
                print(f"    Placement: Node lock ({service.locked_node_id})")
            else:
                print(f"    Placement: Flexible")

    except Exception as e:
        print(f"Error loading services: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

