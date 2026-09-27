"""
SimPy-only Service (Kubernetes-like) and Container models for the simulator.
"""
from __future__ import annotations
import simpy
from simpy.events import AllOf

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service, ContainerSpec, DependencySpec


class ServiceSimulator(Service):
    def __init__(self, env: simpy.Environment, service_id: str, containers: list[ContainerSpec], dependencies: list[DependencySpec] | None = None):
        super().__init__(service_id, containers, dependencies)
        self.env = env
        self.completed_event = None
        self.node = None
        # Track measured usage per-container during simulation (for debugging)
        self._measured_cpu_last: dict[str, float] = {}
        self._measured_ram_last_gb: dict[str, float] = {}
        self._measured_gpu_last_gb: dict[str, float] = {}  # GPU usage for debugging
        # Track variation per iteration (stochastic fluctuation)
        self._cpu_variance: dict[str, float] = {}
        self._ram_variance: dict[str, float] = {}
        self._active: bool = False  # Service currently running

    def _run_container(self, node, cs: ContainerSpec, registry_pull_mbps: float):
        # Image pull time approximation
        bits = cs.image_size_gb * 1024.0 * 1024.0 * 8.0  # GB -> Mb
        pull_time = bits / max(1.0, registry_pull_mbps)

        # Reserve RAM; include shared GPU memory if applicable
        ram_to_get = cs.ram_gb
        if cs.gpu_gb > 0 and getattr(node, "gpu_shared", False):
            ram_to_get += cs.gpu_gb

        # CPU reservation (coarse-grained), RAM reservation, GPU mem reservation if needed
        with node.cpu.request() as cpu_req:
            yield cpu_req
            yield node.ram.get(ram_to_get)
            gpu_req = None
            if cs.gpu_gb > 0:
                if not node.gpu:
                    print(f"{self.env.now:.2f}: {self.service_id}/{cs.id} failed. GPU required but not available on {node.node_id}")
                    yield node.ram.put(ram_to_get)
                    return
                gpu_req = node.gpu.get(cs.gpu_gb)
                yield gpu_req

            # Simulate pull + start + run
            if pull_time > 0:
                print(f"{self.env.now:.2f}: {self.service_id}/{cs.id} pulling image ({cs.image_size_gb} GB) on {node.node_id}")
                yield self.env.timeout(pull_time)
            print(f"{self.env.now:.2f}: {self.service_id}/{cs.id} starting on {node.node_id}")

            # Record measured usage within historical min/max bounds, influenced by node capacity
            import random
            
            # Get historical metrics (or fall back to template specs)
            cpu_min = getattr(cs, 'cpu_usage_min', None) or float(getattr(cs, 'cpu', 0.5))
            cpu_max = getattr(cs, 'cpu_usage_max', None) or float(getattr(cs, 'cpu', 0.5)) * 1.5
            ram_min = getattr(cs, 'ram_usage_min_gb', None) or float(ram_to_get) * 0.8
            ram_max = getattr(cs, 'ram_usage_max_gb', None) or float(ram_to_get) * 1.2

            if cpu_max < cpu_min:
                cpu_min, cpu_max = cpu_max, cpu_min
            if ram_max < ram_min:
                ram_min, ram_max = ram_max, ram_min
            
            # Get node capacity to influence usage
            node_cpu_capacity = float(getattr(node, 'cpu', 1.0))
            node_ram_capacity = float(getattr(node, 'ram', 8.0))
            node_performance = float(getattr(node, 'performance', 1.0))
            
            # Scale ranges based on node capacity (more capacity allows higher usage)
            # Cap metrics at node capacity
            cpu_max = min(cpu_max, node_cpu_capacity * 0.9)  # Leave 10% headroom
            ram_max = min(ram_max, node_ram_capacity * 0.9)
            
            # Generate realistic value within min/max range, influenced by performance
            def _bounded_variance(min_val: float, max_val: float, cap: float, perf: float, last: float | None) -> float:
                if max_val <= 0:
                    return 0.0
                span = max_val - min_val
                # Try a few times to avoid repeating the same value when range allows
                for _ in range(3):
                    base = random.uniform(min_val, max_val) if span > 0 else min_val
                    scaled = base * (0.8 + 0.4 * perf)
                    scaled = max(min_val, min(max_val, scaled))
                    scaled = min(scaled, cap)
                    if span == 0 or last is None or scaled != last:
                        return scaled
                return max(min_val, min(max_val, (last if last is not None else min_val)))

            cpu_variance = _bounded_variance(
                cpu_min,
                cpu_max,
                node_cpu_capacity,
                node_performance,
                self._cpu_variance.get(cs.id),
            )
            ram_variance = _bounded_variance(
                ram_min,
                ram_max,
                node_ram_capacity,
                node_performance,
                self._ram_variance.get(cs.id),
            )
            
            try:
                self._measured_cpu_last[cs.id] = float(cpu_variance)
                self._measured_ram_last_gb[cs.id] = float(ram_variance)
                self._measured_gpu_last_gb[cs.id] = float(getattr(cs, 'gpu_gb', 0.0))
                # Store variance for future iterations
                self._cpu_variance[cs.id] = cpu_variance
                self._ram_variance[cs.id] = ram_variance
            except Exception:
                pass

            # Simulate execution time (scaled by node performance)
            # Skip execution timeout for non-time-constrained services (exec_time is None)
            if cs.exec_time is not None:
                perf = getattr(node, "performance", 1.0)
                actual_exec_time = cs.exec_time / perf if perf > 0 else cs.exec_time
                
                # Mark as active during execution
                self._active = True
                yield self.env.timeout(actual_exec_time)
                self._active = False
                print(f"{self.env.now:.2f}: {self.service_id}/{cs.id} finished on {node.node_id} (perf={perf}, exec_time={actual_exec_time:.2f}s)")
            else:
                # Non-time-constrained service - no execution delay
                print(f"{self.env.now:.2f}: {self.service_id}/{cs.id} executed on {node.node_id} (no time constraint)")

            # Release
            yield node.ram.put(ram_to_get)
            if gpu_req:
                yield node.gpu.put(cs.gpu_gb)

    def run(self, node, registry_pull_mbps: float = 200.0, services_map: dict[str, "ServiceSimulator"] | None = None):
        self.node = node
        self.completed_event = self.env.event()
        # Wait for upstream dependencies and simulate network delay
        if self.dependencies and services_map:
            for dep in self.dependencies:
                up = services_map.get(dep.service_id)
                if not up:
                    continue
                if hasattr(up, 'completed_event') and up.completed_event is not None:
                    yield up.completed_event
                throughput_mbps = 200.0
                transfer_time = (dep.size_mb / max(1.0, throughput_mbps)) if dep.size_mb > 0 else 0.0
                if transfer_time > 0:
                    yield self.env.timeout(transfer_time)
        # Run all containers concurrently (service semantics similar to pods)
        procs = [self.env.process(self._run_container(node, cs, registry_pull_mbps)) for cs in self.containers]
        yield AllOf(self.env, procs)
        if self.completed_event is not None and not self.completed_event.triggered:
            self.completed_event.succeed()
        self.node = None

    def get_measured_usage(self) -> dict:
        """Return measured per-container CPU/RAM/GPU usage for this service with realistic variation.

        Format: {container_id: {"cpu_usage_last": float, "ram_usage_last_gb": float, "gpu_usage_last_gb": float}}
        
        Returns varied measurements if active or recently completed to simulate fluctuation across iterations.
        """
        import random
        result = {}
        for cid, cpu_last in self._measured_cpu_last.items():
            # Apply small random variation (±10% jitter) each time called
            # to simulate per-iteration measurement variance
            cpu_jitter = random.uniform(0.9, 1.1)
            ram_jitter = random.uniform(0.9, 1.1)
            
            result[cid] = {
                "cpu_usage_last": float(cpu_last * cpu_jitter),
                "ram_usage_last_gb": float(self._measured_ram_last_gb.get(cid, 0.0) * ram_jitter),
                "gpu_usage_last_gb": float(self._measured_gpu_last_gb.get(cid, 0.0))
            }
        return result
