"""PlatformTopology: extract layers/nodes from platform.xml.

Parses SimGrid platform.xml to list hosts, groups them either by child zones
or by host-id prefixes (edge-/fog-/cloud-) when a single zone is used.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import simpy

from ..Topology.layer import Layer
from ..Topology.node import Node
from .node_simulator import NodeSimulator
from ..Metrics.base import BaseMetricsFacade

def parse_speed(speed_str: str) -> float:
    """
    Parse a speed string like '16Gf', '3200Mf', '2Tf', '500Kf', etc. to a float value in Gf (GigaFLOPS).
    Supported units: Kf (kilo), Mf (mega), Gf (giga), Tf (tera), no unit (assume Gf).
    """
    import re
    if not speed_str:
        return 1.0
    match = re.match(r"([0-9.]+)\s*([kKmMgGtT]?f)?", speed_str)
    if not match:
        return 1.0
    value = float(match.group(1))
    unit = match.group(2)
    if not unit or unit.lower() == 'gf':
        return value
    elif unit.lower() == 'kf':
        return value / 1e6
    elif unit.lower() == 'mf':
        return value / 1e3
    elif unit.lower() == 'tf':
        return value * 1e3
    else:
        return value  # fallback, treat as Gf

class PlatformTopology:
    """
    Build layers/nodes from platform.xml for the Gym/SimPy layer.
    Handles both multi-zone and single-zone (grouped by prefix) topologies.
    """
    def __init__(
        self,
        env: simpy.Environment,
        platform_xml_path: str,
        ram_defaults: Optional[Dict[str, float]] = None,
        metrics_facade: Optional[BaseMetricsFacade] = None,
    ) -> None:
        self.env = env
        self.platform_xml_path = platform_xml_path
        self.metrics_facade = metrics_facade
        self.ram_defaults = ram_defaults or {
            "edge": 8.0,
            "fog": 16.0,
            "cloud": 32.0,
            "mobile": 32.0,
            "default": 8.0,
        }
        self.default_gpu_mem_gb_per_card = 8.0

    def _infer_group(self, host_id: str) -> str:
        if host_id.startswith("edge-"):
            return "edge"
        if host_id.startswith("fog-"):
            return "fog"
        if host_id.startswith("cloud-"):
            return "cloud"
        if host_id.startswith("frank-"):
            return "edge"  # Frank nodes are edge devices
        return "default"

    def _infer_gpu_mem(self, host_id: str) -> float:
        return self.default_gpu_mem_gb_per_card if "-gpu" in host_id else 0.0

    def _infer_ram(self, group: str) -> float:
        return float(self.ram_defaults.get(group, self.ram_defaults["default"]))

    def _parse_host(self, h: ET.Element, group: str) -> NodeSimulator:
        hid = h.get("id")
        if not hid or hid == "cluster-1-master":
            return None
        cores = int(h.get("core", "1"))
        props = {p.get("id"): p.get("value") for p in h.findall("prop") if p.get("id")}
        ram_prop = props.get("ram.GB")
        ram = float(ram_prop) if ram_prop is not None else self._infer_ram(group)
        gpu_mem_prop = (
            props.get("gpu.memGB")
            or props.get("gpu.memoryGB")
            or props.get("gpu_mem_gb")
            or props.get("gpu.mem")
        )
        if gpu_mem_prop is not None:
            gpu_capacity_gb = float(gpu_mem_prop)
        else:
            gpu_count_str = props.get("gpu.count") or props.get("gpu") or props.get("gpu_count")
            if gpu_count_str is not None:
                try:
                    gpu_capacity_gb = float(gpu_count_str) * self.default_gpu_mem_gb_per_card
                except ValueError:
                    gpu_capacity_gb = self._infer_gpu_mem(hid)
            else:
                gpu_capacity_gb = self._infer_gpu_mem(hid)
        gpu_shared_str = props.get("gpu.shared") or props.get("gpu_shared") or "false"
        gpu_shared = str(gpu_shared_str).lower() in ("1", "true", "yes")
        speed = parse_speed(h.get("speed", "1.0"))
        battery_powered = str(props.get("battery.powered", "false")).lower() in ("1", "true", "yes")
        battery_voltage = float(props.get("battery.voltage") or "0.0")
        battery_capacity_ah = float(props.get("battery.capacity.ah") or "0.0")
        jetson_base_prop = props.get("power.jetson.base")
        jetson_active_prop = props.get("power.jetson.active")
        motor_max_prop = props.get("power.motor.max")

        power_base = float(props.get("power.base") or jetson_base_prop or "5.0")
        power_active = float(props.get("power.active") or jetson_active_prop or "30.0")
        power_jetson_base = float(jetson_base_prop or power_base)
        power_jetson_active = float(jetson_active_prop or power_active)
        power_motor_max = float(motor_max_prop or "0.0")
        # Check if node is schedulable (defaults to true unless explicitly set to false)
        schedulable_str = props.get("schedulable") or props.get("node.schedulable") or "true"
        schedulable = str(schedulable_str).lower() not in ("0", "false", "no")
        node = NodeSimulator(self.env, hid, cpu=cores, ram=ram, gpu=gpu_capacity_gb, gpu_shared=gpu_shared, metrics_facade=self.metrics_facade, schedulable=schedulable)
        node.battery_powered = battery_powered  # type: ignore
        node.battery_voltage = battery_voltage  # type: ignore
        node.battery_capacity_ah = battery_capacity_ah  # type: ignore
        node.power_base = power_base  # type: ignore
        node.power_active = power_active  # type: ignore
        node.power_jetson_base = power_jetson_base  # type: ignore
        node.power_jetson_active = power_jetson_active  # type: ignore
        node.power_motor_max = power_motor_max  # type: ignore
        return node

    def _parse_zone(self, zone: ET.Element) -> Tuple[Layer, List[Node]]:
        cid = zone.get("id", "zone")
        layer = Layer(cid)
        nodes = []
        for h in zone.findall("host"):
            group = self._infer_group(h.get("id", ""))
            node = self._parse_host(h, group)
            if node:
                node.layer_id = cid  # Set layer_id for allocation emitter
                layer.add_node(node)
                nodes.append(node)
        return layer, nodes

    def _parse_single_zone(self, zone: ET.Element) -> Tuple[List[Layer], List[Node]]:
        groups: Dict[str, Layer] = {}
        all_nodes: List[Node] = []
        for h in zone.findall("host"):
            hid = h.get("id")
            if not hid or hid == "cluster-1-master":
                continue
            group = self._infer_group(hid)
            if group not in groups:
                groups[group] = Layer(group)
            node = self._parse_host(h, group)
            if node:
                node.layer_id = group  # Set layer_id for allocation emitter
                groups[group].add_node(node)
                all_nodes.append(node)
        return list(groups.values()), all_nodes

    def build(self) -> Tuple[List[Layer], List[Node]]:
        """
        Parse the platform.xml and return a list of Layer objects and all Node objects.
        Handles both multi-zone and single-zone topologies.
        """
        if not os.path.exists(self.platform_xml_path):
            raise FileNotFoundError(self.platform_xml_path)
        tree = ET.parse(self.platform_xml_path)
        root = tree.getroot()
        zones = root.findall(".//zone")
        top_zone = zones[0] if zones else None
        child_zones = top_zone.findall("zone") if top_zone is not None else []
        layers: List[Layer] = []
        all_nodes: List[Node] = []
        if child_zones:
            for z in child_zones:
                layer, nodes = self._parse_zone(z)
                layers.append(layer)
                all_nodes.extend(nodes)
        elif top_zone is not None:
            layers, all_nodes = self._parse_single_zone(top_zone)
        else:
            raise ValueError("Invalid platform.xml: no <zone> found")
        # Attach metrics facade if needed
        if self.metrics_facade is None and all_nodes:
            from ..Metrics.simulator_facade import SimulatorMetricsFacade
            from .util.network_util import NetworkUtil
            
            # Build latency matrix from platform.xml
            try:
                latency_matrix = NetworkUtil.build_latency_from_platform_xml(self.platform_xml_path, all_nodes)
            except Exception:
                # Fall back to None if XML parsing fails
                latency_matrix = None
            
            self.metrics_facade = SimulatorMetricsFacade(all_nodes, latency_matrix=latency_matrix)
            for node in all_nodes:
                node.metrics.set_facade(self.metrics_facade)
        return layers, all_nodes
