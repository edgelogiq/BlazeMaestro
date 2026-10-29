"""
Base NSGA-II orchestrator with common functionality for all NSGA-II implementations.

This module provides shared logic for NSGA-II based orchestrators to avoid code duplication.
Different implementations can inherit from this and override specific methods or parameters.
"""

from typing import Any, Sequence, List, Tuple, Optional, Dict
from ..Metrics.base import BaseMetricsFacade
from ..Metrics.simulator_facade import SimulatorMetricsFacade
from .orchestrator_utils import OrchestratorUtils
from .base_orchestrator import BaseOrchestrator


class BaseNSGA2Orchestrator(BaseOrchestrator):
    """Base class for NSGA-II orchestration implementations.

    Provides common methods for:
    - Service resource demand calculation
    - Node capacity constraint enforcement
    - Service dependency and latency calculation
    - Energy and period constraint evaluation

    Subclasses should override _propose_assignment() to use different NSGA-II implementations.
    """

    def __init__(self, services: Sequence[Any], nodes: Sequence[Any],
                 energy_estimator=None, population_size: int = 40, generations: int = 80,
                 metrics: Optional[BaseMetricsFacade] = None):
        """Initialize Base NSGA-II Orchestrator.

        Args:
            services: List of services to orchestrate
            nodes: List of available nodes
            energy_estimator: Optional energy estimator for battery-aware scheduling
            population_size: NSGA-II population size
            generations: NSGA-II number of generations
            metrics: Optional metrics facade (defaults to SimulatorMetricsFacade)
        """
        # Use metrics facade (provides latency via get_latency_matrix)
        metrics_facade = metrics or SimulatorMetricsFacade(nodes)
        
        # Store directly instead of calling super()
        self.services = services
        self.nodes = nodes
        self.metrics = metrics_facade
        self.energy_estimator = energy_estimator
        self.population_size = population_size
        self.generations = generations
        self.num_services = len(services)
        self.num_nodes = len(nodes)

    def _service_demands(self, svc: Any, node_id: Optional[str] = None) -> Tuple[float, float, float, float, float]:
        """Calculate service resource demands: (cpu, ram, gpu, exec_time, period).

        Uses unified implementation from OrchestratorUtils.

        Args:
            svc: Service object
            node_id: Optional node ID for per-node demands

        Returns:
            Tuple of (cpu_cores, ram_gb, gpu_gb, exec_time_s, period_s)
        """
        return OrchestratorUtils.calculate_service_demands(svc, node_id)

    def _dependency_wait(self, s_idx: int, n_idx: int, assignment: List[int], hard_only: bool = False) -> float:
        """Calculate latency waiting time due to service dependencies.

        Accounts for network latency and data transfer time for dependent services.

        Args:
            s_idx: Service index
            n_idx: Node index being assigned
            assignment: Current assignment
            hard_only: If True, only include dependencies with constraint_level='hard'

        Returns:
            Wait time in seconds
        """
        svc = self.services[s_idx]
        deps = getattr(svc, "dependencies", []) or []
        if not deps:
            return 0.0

        throughput_mbps = 200.0  # Assumed network throughput
        latency_matrix = OrchestratorUtils.get_latency_matrix(self.metrics)

        # If latency matrix not available, try to generate it
        if latency_matrix is None and self.metrics and hasattr(self.metrics, 'get_latency_matrix'):
            try:
                latency_matrix = self.metrics.get_latency_matrix(self.nodes)
            except Exception:
                latency_matrix = None

        svc_index = {getattr(s, "service_id", str(i)): i for i, s in enumerate(self.services)}
        visiting: set[str] = set()

        def dfs(cur_svc: Any, cur_node_idx: int) -> float:
            total = 0.0
            for dep in getattr(cur_svc, "dependencies", []) or []:
                if getattr(dep, "blocking", True) is False:
                    continue
                # Skip if hard_only=True and constraint_level is not 'hard'
                if hard_only:
                    constraint_level = getattr(dep, "constraint_level", "hard")
                    if constraint_level != "hard":
                        continue
                dep_id = getattr(dep, "service_id", None)
                if not isinstance(dep_id, str):
                    continue
                up_idx = svc_index.get(dep_id)
                if up_idx is None:
                    continue
                try:
                    up_node_idx = int(assignment[up_idx])
                except Exception:
                    continue

                try:
                    net_lat = float(latency_matrix[up_node_idx][cur_node_idx]) if latency_matrix is not None else 0.0
                except Exception:
                    net_lat = 0.0
                size_mb = float(getattr(dep, "size_mb", 0.0))
                transfer = (size_mb / max(1.0, throughput_mbps)) if size_mb > 0 else 0.0
                edge_cost = max(0.0, net_lat + transfer)

                if dep_id in visiting:
                    total += edge_cost
                    continue

                visiting.add(dep_id)
                dep_svc = self.services[up_idx]
                total += edge_cost + dfs(dep_svc, up_node_idx)
                visiting.remove(dep_id)
            return total

        total = dfs(svc, int(n_idx))

        if getattr(svc, "control_loop", False):
            for down_idx, down_svc in enumerate(self.services):
                for dep in getattr(down_svc, "dependencies", []) or []:
                    if getattr(dep, "feedback", False) is not True:
                        continue
                    if getattr(dep, "service_id", None) != getattr(svc, "service_id", None):
                        continue
                    try:
                        down_node_idx = int(assignment[down_idx])
                    except Exception:
                        continue
                    try:
                        net_lat = float(latency_matrix[ int(n_idx) ][down_node_idx]) if latency_matrix is not None else 0.0
                    except Exception:
                        net_lat = 0.0
                    transfer = (float(getattr(dep, "size_mb", 0.0)) / max(1.0, throughput_mbps)) if getattr(dep, "size_mb", 0.0) > 0 else 0.0
                    total += max(0.0, net_lat + transfer)

        return total

    def _evaluate_assignment(self, assignment: List[int]) -> Tuple[float, float, float, bool]:
        """Evaluate an assignment for feasibility and objective values.

        Checks:
        - Node capacity constraints (CPU, RAM, GPU)
        - Locked-node constraints
        - Period/deadline constraints

        Objectives:
        - Total energy consumption
        - Total latency (network + transfer)
        - Period violations count

        Args:
            assignment: Service -> node assignment

        Returns:
            Tuple of (total_energy, total_latency, period_violations, is_valid)
        """
        node_cpu = [0.0] * self.num_nodes
        node_ram = [0.0] * self.num_nodes
        node_gpu = [0.0] * self.num_nodes
        total_energy = 0.0
        total_latency = 0.0
        period_violations = 0
        valid = True

        # Seed with current baseline usage from non-optimizable services
        current_assignment = getattr(self, '_current_assignment', None)
        baseline_usage = self._current_load_usage(current_assignment, exclude_services=self.services)
        node_index_by_id = {str(getattr(n, 'node_id', i)): i for i, n in enumerate(self.nodes)}
        for nid, usage in baseline_usage.items():
            node_idx = node_index_by_id.get(str(nid))
            if node_idx is None:
                continue
            node_cpu[node_idx] += float(usage.get('cpu', 0.0))
            node_ram[node_idx] += float(usage.get('ram', 0.0))
            node_gpu[node_idx] += float(usage.get('gpu', 0.0))

        # Shared period test of V(x) (same rule as every other orchestrator)
        period_utils = OrchestratorUtils(self.services, self.nodes, OrchestratorUtils.get_latency_matrix(self.metrics))
        placement_map = {str(getattr(sv, "service_id", k)): str(getattr(self.nodes[int(a)], "node_id", ""))
                         for k, (sv, a) in enumerate(zip(self.services, assignment))}
        for s_idx, n_idx in enumerate(assignment):
            svc = self.services[s_idx]
            node = self.nodes[n_idx]

            # Check locked-node constraint
            locked_nid = getattr(svc, 'locked_node_id', None)
            if locked_nid is not None and str(getattr(node, 'node_id', '')) != str(locked_nid):
                valid = False
            
            # Check locked-layer constraint
            locked_layer = getattr(svc, 'locked_layer_id', None)
            if locked_layer is not None:
                node_layer = getattr(node, 'layer_id', None)
                if node_layer != locked_layer:
                    valid = False

            # Get service demands
            nid = str(getattr(node, 'node_id', ''))
            cpu, ram, gpu, exec_time, period = self._service_demands(svc, nid)

            # Aggregate resource usage
            node_cpu[n_idx] += cpu

            # RAM + GPU memory if shared
            gpu_shared = bool(node.metrics.get_property('gpu_shared') or getattr(node, "gpu_shared", False))
            ram_needed = ram + (gpu if gpu_shared else 0.0)
            node_ram[n_idx] += ram_needed

            # GPU capacity check
            if gpu > 0:
                caps = node.metrics.get_capacity()
                if float(caps.get("gpu_gb", 0.0)) <= 0.0:
                    valid = False
                else:
                    node_gpu[n_idx] += gpu
                    if float(caps.get("gpu_gb", 0.0)) < node_gpu[n_idx]:
                        valid = False

            # Check node capacity constraints
            caps = node.metrics.get_capacity()
            if node_cpu[n_idx] > float(caps.get("cpu", 0.0)):
                valid = False
            if node_ram[n_idx] > float(caps.get("ram_gb", 0.0)):
                valid = False

            # Energy calculation (0 for grid-powered nodes)
            total_energy += OrchestratorUtils.estimate_energy(svc, node)

            # L_tot(x) = sum_s L_s(x): transitive latency over blocking dependencies only (paper Eq. ltot_def)
            # Get network latency for this service's dependencies only
            service_network_latency = self._dependency_wait(s_idx, n_idx, assignment)
            total_latency += service_network_latency

            if period < float('inf') and not period_utils.service_period_ok(svc, nid, placement_map):
                period_violations += 1

        return total_energy, total_latency, period_violations, valid

    def _is_assignment_valid(self, assignment: List[int]) -> bool:
        """Quick validation check for an assignment.

        Args:
            assignment: Service -> node assignment

        Returns:
            True if assignment is valid (meets all constraints)
        """
        node_cpu = [0.0] * self.num_nodes
        node_mem = [0.0] * self.num_nodes
        node_gpu = [0.0] * self.num_nodes

        for s_idx, n_idx in enumerate(assignment):
            svc = self.services[s_idx]
            node = self.nodes[n_idx]

            # Check locked-node constraint
            locked_nid = getattr(svc, 'locked_node_id', None)
            if locked_nid is not None and str(getattr(node, 'node_id', '')) != str(locked_nid):
                return False
            
            # Check locked-layer constraint
            locked_layer = getattr(svc, 'locked_layer_id', None)
            if locked_layer is not None:
                node_layer = getattr(node, 'layer_id', None)
                if node_layer != locked_layer:
                    return False

            # Aggregate container needs
            cpu, ram, gpu, _, _ = self._service_demands(svc)
            node_cpu[n_idx] += cpu

            # RAM + GPU if shared
            gpu_shared = getattr(node, "gpu_shared", False)
            ram_needed = ram + (gpu if gpu_shared else 0.0)
            node_mem[n_idx] += ram_needed

            # GPU capacity check
            if gpu > 0:
                caps = node.metrics.get_capacity()
                if float(caps.get('gpu_gb', 0.0)) <= 0.0:
                    return False
                node_gpu[n_idx] += gpu
                if node_gpu[n_idx] > float(caps.get('gpu_gb', 0.0)):
                    return False

        # Check all node capacities
        for i, node in enumerate(self.nodes):
            caps = node.metrics.get_capacity()
            if node_cpu[i] > float(caps.get('cpu', 0.0)):
                return False
            if node_mem[i] > float(caps.get('ram_gb', 0.0)):
                return False

        return True

    def _propose_assignment(self, services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Propose an assignment using NSGA-II optimization.

        This method bridges BaseOrchestrator interface with NSGA2 optimize() method.

        Args:
            services: Services to assign (defaults to self.services)

        Returns:
            Assignment dictionary {service_id: node_id}
        """
        # Call the optimize() method that subclasses implement
        if hasattr(self, 'optimize'):
            return self.optimize(services=services)
        else:
            raise NotImplementedError("Subclasses must implement optimize() method")
