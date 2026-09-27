"""
Simulator class for running the orchestration simulation using SimPy and Gymnasium.
"""
import simpy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.layer import Layer
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.node import Node
from simgrid import Host


class OrchestrationEnv(gym.Env):
    """
    A Gymnasium environment for the orchestration problem.
    """
    def __init__(self, clusters, processes):
        super().__init__()
        self.clusters = clusters
        self.processes = processes
        self.all_nodes = [node for cluster in self.clusters for node in cluster.get_nodes()]
        
        # SimGrid setup is now handled in main.py
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.all_nodes))
        
        # Observation space: [process_req_cpu, process_req_ram, process_req_gpu, node_avail_cpu, node_avail_ram, node_avail_gpu, ...]
        num_nodes = len(self.all_nodes)
        obs_dim = 3 + num_nodes * 3  # 3 for process reqs, 3 per node for available resources
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.current_process_index = 0

    def get_network_latency(self, from_node, to_node):
        """
        Calculates the network latency between two nodes using SimGrid.
        """
        from_host = Host.by_name(from_node.node_id)
        to_host = Host.by_name(to_node.node_id)
        # Prefer Host.route_to if available. It may return (links, latency) or just a links list.
        try:
            route = from_host.route_to(to_host)
            # If route is a tuple (links, latency)
            if isinstance(route, tuple) and len(route) == 2:
                return float(route[1])
            # Else try to sum link latencies
            total = 0.0
            for link in (route or []):
                try:
                    total += float(link.latency())
                except Exception:
                    pass
            return total
        except Exception:
            # Fallback: no routing info
            return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_process_index = 0
        # Reset node resources if necessary (SimPy handles this, but good for consistency)
        return self._get_obs(), {}

    def step(self, action):
        process = self.processes[self.current_process_index]
        node = self.all_nodes[action]

        # Check for GPU availability (dedicated or shared)
        if process.required_gpu > 0 and not node.gpu:
            reward = -1.5 # Heavier penalty for trying to schedule a GPU task on a non-GPU node
        # Simple reward: 1 if allocation is possible, -1 otherwise
        elif (
            node.cpu.capacity - node.cpu.count >= process.required_cpu and
            (
                # Dedicated GPU memory: need RAM for RAM only and GPU container for GPU
                (
                    not getattr(node, "gpu_shared", False)
                    and node.ram.level >= process.required_ram
                    and (not node.gpu or node.gpu.level >= process.required_gpu)
                )
                or
                # Shared GPU memory: GPU memory comes from RAM, ensure combined RAM available
                (
                    getattr(node, "gpu_shared", False)
                    and node.ram.level >= (process.required_ram + process.required_gpu)
                    and (not node.gpu or node.gpu.level >= process.required_gpu)
                )
            )
        ):
            reward = 1
            # In a real scenario, you would start the SimPy process here
        else:
            reward = -1

        self.current_process_index += 1
        terminated = self.current_process_index >= len(self.processes)
        truncated = False  # Not used here
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        if self.current_process_index >= len(self.processes):
            shape = self.observation_space.shape
            shape = tuple(shape) if shape is not None else (0,)
            return np.zeros(shape, dtype=np.float32)
            
        process = self.processes[self.current_process_index]
        process_reqs = [process.required_cpu, process.required_ram, process.required_gpu]
        
        node_resources = []
        for node in self.all_nodes:
            gpu_level = node.gpu.level if node.gpu else 0
            node_resources.extend([
                node.cpu.capacity - node.cpu.count,
                node.ram.level,
                gpu_level
            ])
            
        return np.array(process_reqs + node_resources, dtype=np.float32)

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
