"""
This file makes the Simulator directory a Python package.

Keep imports lightweight to avoid requiring optional runtime deps (e.g., simgrid)
on mere package import. Import heavy modules explicitly where needed.
"""

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.layer import Layer
