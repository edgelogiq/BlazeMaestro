"""
Platform abstraction layer for topology management.

This module provides a base Platform interface and implementations for both
simulator (SimGrid/SimPy) and real Kubernetes cluster environments.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.layer import Layer
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node


class Platform(ABC):
    """
    Abstract base class for platform topology providers.

    A Platform is responsible for discovering and providing the infrastructure
    topology (layers and nodes). Implementations can:
    - Load from simulator (platform.xml)
    - Query from Kubernetes clusters
    - Read from configuration files
    - Discover from cloud providers
    """

    @abstractmethod
    def build(self) -> Tuple[List[Layer], List[Node]]:
        """
        Build and return the platform topology.

        Returns:
            Tuple of (layers, nodes):
                - layers: List of Layer objects
                - nodes: List of all Node objects across all layers
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get a human-readable name for this platform.

        Returns:
            Platform name (e.g., "Simulator", "Kubernetes Production Cluster")
        """
        pass


class PlatformTopologyManager:
    """
    Manages platform topology through dependency injection.

    This class acts as a facade that accepts any Platform implementation
    and provides a unified interface for topology management.
    """

    def __init__(self, platform: Platform):
        """
        Initialize the topology manager with a platform provider.

        Args:
            platform: Platform implementation (simulator, Kubernetes, etc.)
        """
        self.platform = platform
        self.layers = None
        self.nodes = None

    def load_topology(self) -> Tuple[List[Layer], List[Node]]:
        """
        Load the topology from the injected platform.

        Returns:
            Tuple of (layers, nodes)
        """
        self.layers, self.nodes = self.platform.build()
        return self.layers, self.nodes

    def get_layers(self) -> List[Layer]:
        """
        Get all layers.

        Returns:
            List of Layer objects
        """
        if self.layers is None:
            self.load_topology()
        return self.layers

    def get_nodes(self) -> List[Node]:
        """
        Get all nodes across all layers.

        Returns:
            List of Node objects
        """
        if self.nodes is None:
            self.load_topology()
        return self.nodes

    def get_node_to_layer_map(self) -> dict:
        """
        Create a mapping from node_id to layer_id.

        Returns:
            Dictionary mapping node_id -> layer_id
        """
        if self.layers is None:
            self.load_topology()

        node_to_layer = {}
        for layer in self.layers:
            for node_id in layer.get_node_ids():
                node_to_layer[node_id] = layer.layer_id

        return node_to_layer

    def get_platform_name(self) -> str:
        """
        Get the name of the platform.

        Returns:
            Platform name
        """
        return self.platform.get_name()

    def __str__(self):
        """String representation."""
        layers_count = len(self.layers) if self.layers else 0
        nodes_count = len(self.nodes) if self.nodes else 0
        return f"PlatformTopologyManager({self.get_platform_name()}: {layers_count} layers, {nodes_count} nodes)"

