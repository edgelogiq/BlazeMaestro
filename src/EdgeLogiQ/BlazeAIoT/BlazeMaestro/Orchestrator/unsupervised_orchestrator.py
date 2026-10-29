from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import random

from ..Metrics.base import BaseMetricsFacade
from ..Metrics.simulator_facade import SimulatorMetricsFacade
from .orchestrator_utils import OrchestratorUtils
from .base_orchestrator import BaseOrchestrator


@dataclass
class OrchestratorWeights:
    """Weights for scoring each candidate node. Higher weight = more important to minimize."""
    energy: float = 10.0  # Increased: prioritize energy minimization
    latency: float = 1.0  # Reduced: secondary importance
    balance: float = 0.5  # Reduced: tertiary importance


class GreedyWeightedOrchestrator(BaseOrchestrator):
    """Energy-aware greedy orchestrator using weighted multi-objective scoring.

    Scoring per candidate node combines:
      - energy: estimated energy consumption for (service,node) placement
      - latency: sum of link latencies to already-placed upstream dependencies
      - balance: post-placement residual capacity score (prefer balanced loads)
    
    Uses a greedy selection algorithm that places services in topological order,
    selecting the node with the best weighted score for each service.
    """

    def __init__(
        self,
        services: Sequence[Any],
        nodes: Sequence[Any],
        latency_matrix: Optional[Any] = None,
        *,
        weights: Optional[OrchestratorWeights] = None,
        metrics: Optional[BaseMetricsFacade] = None,
        random_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        metrics_facade = metrics or SimulatorMetricsFacade(nodes)
        
        # Initialize base class
        super().__init__(services, nodes, metrics_facade)
        
        if random_seed is not None:
            random.seed(random_seed)
        self.weights = weights or OrchestratorWeights()
        self.verbose = verbose
        self.backtrack_budget = 5000  # max placement trials per decision (bounds decision time)
        
        # Map node_id -> index for quick lookups
        self._node_index: Dict[str, int] = {str(getattr(n, "node_id", i)): i for i, n in enumerate(self.nodes)}
        
        # Latency matrix (use provided or get from metrics)
        self.latency_matrix = latency_matrix if latency_matrix is not None else self.metrics.get_latency_matrix(self.nodes)
        


    # ---------- Constraints and helpers ----------
    def _is_feasible(self, svc: Any, node_id: str, avail: Dict[str, float], caps: Dict[str, float], gpu_shared: bool) -> bool:
        """Check feasibility with custom tolerance for unreliable execution times."""
        cpu, ram, gpu, exec_time, period = self._service_demands(svc, node_id)
        needed_ram = ram + (gpu if gpu_shared else 0.0)
        
        # Period feasibility is checked with the shared V(x) test (OrchestratorUtils.service_period_ok)
        if avail.get("cpu", 0.0) < cpu:
            return False
        if avail.get("ram_gb", 0.0) < needed_ram:
            return False
        if gpu > 0.0 and caps.get("gpu_gb", 0.0) < gpu:
            return False
        return True

    def _apply_allocation(self, svc: Any, avail: Dict[str, float], gpu_shared: bool) -> None:
        cpu, ram, gpu, exec_time, period = self._service_demands(svc)
        avail["cpu"] = max(0.0, avail.get("cpu", 0.0) - cpu)
        used_ram = ram + (gpu if gpu_shared else 0.0)
        avail["ram_gb"] = max(0.0, avail.get("ram_gb", 0.0) - used_ram)
        # GPU accounting uses capacity feasibility only; leave 'available' as-is (simulator tracks runtime levels)

    def _latency_cost(self, svc: Any, candidate_idx: int, placement: Dict[str, str], hard_only: bool = False) -> float:
        """Calculate latency cost using base class helper.
        
        Args:
            hard_only: If True, only include dependencies with constraint_level='hard'
        """
        if hard_only:
            return self._latency_cost_hard_only(svc, candidate_idx, placement, self.latency_matrix, self._node_index, self.services)
        return self._latency_cost_full(svc, candidate_idx, placement, self.latency_matrix, self._node_index, self.services)

    def _initialize_node_resources(self, exclude_services: Optional[Sequence[Any]] = None) -> tuple:
        """Initialize node resource tracking.
        
        Returns:
            Tuple of (capacities, available, gpu_shared) dicts
        """
        node_caps = {}
        node_avail = {}
        node_gpu_shared = {}
        
        for n in self.nodes:
            nid = str(getattr(n, "node_id", ""))
            node_caps[nid] = self.metrics.get_node_capacity(nid)
            node_avail[nid] = self.metrics.get_node_available(nid)
            node_gpu_shared[nid] = bool(self.metrics.get_node_property(nid, "gpu_shared") or getattr(n, "gpu_shared", False))

        # Apply current load baseline so feasibility considers actual usage + new demand
        current_assignment = getattr(self, '_current_assignment', None)
        baseline_usage = self._current_load_usage(current_assignment, exclude_services)
        for nid, usage in baseline_usage.items():
            if nid not in node_avail:
                continue
            node_avail[nid]["cpu"] = max(0.0, float(node_avail[nid].get("cpu", 0.0)) - float(usage.get("cpu", 0.0)))
            node_avail[nid]["ram_gb"] = max(0.0, float(node_avail[nid].get("ram_gb", 0.0)) - float(usage.get("ram", 0.0)))
            node_avail[nid]["gpu_gb"] = max(0.0, float(node_avail[nid].get("gpu_gb", 0.0)) - float(usage.get("gpu", 0.0)))
        
        if self.verbose:
            total_cpu_cap = sum(c.get('cpu', 0) for c in node_caps.values())
            if total_cpu_cap == 0:
                print(f"[GreedyWeightedOrchestrator] WARNING: All nodes have zero CPU capacity!")
                print(f"  Node caps sample: {dict(list(node_caps.items())[:3])}")
        
        return node_caps, node_avail, node_gpu_shared

    def _filter_candidate_nodes(self, svc: Any, placement: Optional[Dict[str, str]] = None) -> List[Any]:
        """Filter nodes based on lock constraints and the service's allowed layers.
        
        Args:
            svc: Service to filter nodes for
            placement: Current placement map (unused)
        
        Returns:
            List of candidate nodes
        """
        locked_nid = getattr(svc, 'locked_node_id', None)
        locked_layer = getattr(svc, 'locked_layer_id', None)
        
        candidate_nodes = []
        for n in self.nodes:
            node_id = str(getattr(n, 'node_id', ''))
            node_layer = getattr(n, 'layer_id', None)
            
            # Check node lock
            if locked_nid is not None and node_id != str(locked_nid):
                continue
            
            # Check layer lock
            if locked_layer is not None and node_layer != locked_layer:
                continue
            
            # Multi-layer services may use any of their allowed layers (no layer preservation,
            # which is not part of V(x))
            if getattr(svc, 'multi_layer', False):
                allowed_layers = getattr(svc, 'layers', [])
                if allowed_layers and node_layer not in allowed_layers:
                    continue
            
            candidate_nodes.append(n)
        
        return candidate_nodes

    def _evaluate_candidate_node(self, svc: Any, n: Any, placement: Dict[str, str],
                                 node_avail: Dict[str, Dict[str, float]],
                                 node_caps: Dict[str, Dict[str, float]],
                                 node_gpu_shared: Dict[str, bool]) -> Optional[Dict[str, Any]]:
        """Evaluate a node as candidate for service placement."""
        nid = str(getattr(n, 'node_id', ''))
        avail = node_avail[nid]
        caps = node_caps[nid]
        gshared = node_gpu_shared[nid]
        
        if not self._is_feasible(svc, nid, avail, caps, gshared):
            return None
        
        # Estimate energy
        energy = OrchestratorUtils.estimate_energy(svc, n)
        
        # Calculate latency (full, for optimization objective)
        candidate_idx = self._node_index.get(nid)
        lat = self._latency_cost(svc, candidate_idx, placement) if candidate_idx is not None else 0.0
        
        # Period part of V(x), shared with every other orchestrator
        if not OrchestratorUtils(self.services, self.nodes, self._get_latency_matrix()).service_period_ok(svc, nid, placement):
            return None
        
        # Calculate balance cost
        sim_avail = dict(avail)
        self._apply_allocation(svc, sim_avail, gshared)
        balance = self._balance_cost(sim_avail, caps)
        
        return {
            "nid": nid,
            "energy": energy,
            "latency": lat,
            "balance": balance,
            "caps": caps,
            "avail": avail,
            "gshared": gshared,
        }




    def _propose_assignment(self, services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Generate a service-to-node assignment (inference only, no training).
        
        Args:
            services: Optional services list. If provided, uses these instead of self.services.
        
        Returns:
            Dict mapping service_id (str) to node_id (str)
        """
        svcs: List[Any] = list(services) if services is not None else list(self.services)
        
        # Initialize node resource tracking
        node_caps, node_avail, node_gpu_shared = self._initialize_node_resources(exclude_services=svcs)
        
        # Seed with the current assignment so services outside svcs keep their node
        placement: Dict[str, str] = {}
        if hasattr(self, '_current_assignment') and self._current_assignment:
            placement = dict(self._current_assignment)
        
        # Place services in topological order, depth-first in (energy, latency, balance) order
        # with backtracking: when a service has no feasible candidate, revisit earlier choices
        # instead of failing the whole decision.
        import copy as _copy
        order = self._topological_sort(svcs)
        budget = [self.backtrack_budget]

        def place(i: int) -> bool:
            if i == len(order):
                return True
            svc = order[i]
            sid = str(getattr(svc, "service_id", ""))
            candidates = []
            for n in self._filter_candidate_nodes(svc, placement):
                c = self._evaluate_candidate_node(svc, n, placement, node_avail, node_caps, node_gpu_shared)
                if c:
                    candidates.append(c)
            candidates.sort(key=lambda c: (c["energy"], c["latency"], c["balance"]))
            for c in candidates:
                if budget[0] <= 0:
                    return False
                budget[0] -= 1
                saved_avail = _copy.deepcopy(node_avail[c["nid"]])
                prev = placement.get(sid)
                placement[sid] = c["nid"]
                self._apply_allocation(svc, node_avail[c["nid"]], c["gshared"])
                if place(i + 1):
                    return True
                node_avail[c["nid"]] = saved_avail
                if prev is None:
                    placement.pop(sid, None)
                else:
                    placement[sid] = prev
            return False

        if not place(0):
            raise RuntimeError("No feasible placement found by Greedy Weighted (backtracking exhausted)")
        # Every service passed the shared V(x) checks during the search: no repair pass needed.
        return placement

    # ---------- Optimize ----------
    def optimize(self, services: Optional[Sequence[Any]] = None,
                 current_assignment: Optional[Dict[str, str]] = None,
                 hysteresis_threshold: float = 0.0) -> Dict[str, str]:
        """Generate optimized placement with hysteresis support.
        
        Delegates to base class which handles hysteresis logic.
        """
        # Read by the placement search (seed placement and baseline load)
        self._current_assignment = current_assignment if current_assignment else {}
        return super().optimize(services, current_assignment, hysteresis_threshold)




    def run_single_episode_training(self, all_nodes, services, latency):
        """Perform a single inference step and return (loss, violation_count)."""
        # Update orchestrator with new environment
        self.services = services
        self.num_services = len(services)
        self.nodes = all_nodes
        self.num_nodes = len(all_nodes)
        self.latency_matrix = latency

        unsup_assignment = self.optimize(services)
        unsup_violations = []
        for svc in services:
            if svc.service_id in unsup_assignment:
                node_id = unsup_assignment[svc.service_id]
                node = next((n for n in all_nodes if n.node_id == node_id), None)
                if node:
                    cpu_req, ram_req, gpu_req, exec_time, period = self._service_demands(svc)
                    caps = node.metrics.get_capacity()
                    if cpu_req > caps.get("cpu", 0):
                        unsup_violations.append(f"CPU: {cpu_req} > {caps.get('cpu', 0)}")
                    if ram_req > caps.get("ram_gb", 0):
                        unsup_violations.append(f"RAM: {ram_req} > {caps.get('ram_gb', 0)}")
                    if gpu_req > caps.get("gpu_gb", 0):
                        unsup_violations.append(f"GPU: {gpu_req} > {caps.get('gpu_gb', 0)}")

        return 0.0, len(unsup_violations)
