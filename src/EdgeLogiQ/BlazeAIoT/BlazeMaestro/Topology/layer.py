"""
Layer class representing a group of nodes.
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .node import Node

class Layer:
    """
    Represents a layer of nodes.
    """
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.nodes = []

    def add_node(self, node: 'Node'):
        self.nodes.append(node)

    def get_nodes(self):
        return self.nodes

    def __str__(self):
        return f"Layer(id={self.layer_id}, nodes={len(self.nodes)})"
