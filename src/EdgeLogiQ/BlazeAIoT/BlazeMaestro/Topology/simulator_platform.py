"""
Simulator Platform implementation.

Wraps the existing PlatformTopology to conform to the Platform interface.
"""

from typing import List, Tuple
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.platform import Platform
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.layer import Layer
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node


class SimulatorPlatform(Platform):
    """
    Platform implementation for SimGrid/SimPy simulator.

    Loads topology from platform.xml using the existing PlatformTopology.
    """

    def __init__(self, env, platform_xml_path: str, ram_defaults: dict = None, metrics_facade=None):
        """
        Initialize simulator platform.

        Args:
            env: SimPy environment
            platform_xml_path: Path to platform.xml file
            ram_defaults: Optional RAM defaults for different node types
            metrics_facade: Optional metrics facade
        """
        self.env = env
        self.platform_xml_path = platform_xml_path
        self.ram_defaults = ram_defaults
        self.metrics_facade = metrics_facade

    def build(self) -> Tuple[List[Layer], List[Node]]:
        """
        Build topology from platform.xml.

        Returns:
            Tuple of (layers, nodes)
        """
        from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.platform_topology import PlatformTopology

        topology = PlatformTopology(
            self.env,
            self.platform_xml_path,
            self.ram_defaults,
            self.metrics_facade
        )

        return topology.build()

    def get_name(self) -> str:
        """Get platform name."""
        import os
        return f"Simulator ({os.path.basename(self.platform_xml_path)})"

