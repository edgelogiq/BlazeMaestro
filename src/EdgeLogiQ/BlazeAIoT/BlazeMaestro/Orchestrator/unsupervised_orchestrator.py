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
        """Filter nodes based on lock constraints and multi-layer preservation.
        
        Args:
            svc: Service to filter nodes for
            placement: Current placement map (optional, for multi-layer layer preservation)
        
        Returns:
            List of candidate nodes
        """
        locked_nid = getattr(svc, 'locked_node_id', None)
        locked_layer = getattr(svc, 'locked_layer_id', None)
        
        # Get current placement to preserve layer for multi-layer services
        current_layer = None
        if placement and getattr(svc, 'multi_layer', False):
            current_node_id = placement.get(str(getattr(svc, 'service_id', '')))
            if current_node_id:
                for node in self.nodes:
                    if str(getattr(node, 'node_id', '')) == current_node_id:
                        current_layer = getattr(node, 'layer_id', None)
                        break
        
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
            
            # Check multi-layer layer preservation
            if getattr(svc, 'multi_layer', False):
                allowed_layers = getattr(svc, 'layers', [])
                if allowed_layers:
                    # If currently placed and in an allowed layer, restrict to that layer
                    if current_layer and current_layer in allowed_layers:
                        if node_layer != current_layer:
                            continue
                    # If not currently placed, allow any allowed layer
                    elif node_layer not in allowed_layers:
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
        
        # Initialize placement with current assignment for multi-layer layer preservation
        placement: Dict[str, str] = {}
        if hasattr(self, '_current_assignment') and self._current_assignment:
            placement = dict(self._current_assignment)
        
        # Place services in topological order
        for svc in self._topological_sort(svcs):
            # Get candidate nodes (respecting locks and multi-layer preservation)
            candidate_nodes = self._filter_candidate_nodes(svc, placement)
            
            # Evaluate all candidates
            candidates = []
            for n in candidate_nodes:
                candidate = self._evaluate_candidate_node(svc, n, placement, node_avail, node_caps, node_gpu_shared)
                if candidate:
                    candidates.append(candidate)
            
            # Select best candidate (no fallback)
            if not candidates:
                svc_id = getattr(svc, 'service_id', svc)
                # Log detailed diagnostic info for debugging
                if self.verbose:
                    print(f"\n[GreedyWeightedOrchestrator] DIAGNOSTICS for {svc_id}:")
                    cpu, ram, gpu, _, _ = self._service_demands(svc)
                    print(f"  Service demands: cpu={cpu:.3f}, ram={ram:.3f}GB, gpu={gpu:.3f}GB")
                    
                    print(f"  Candidate nodes evaluated:")
                    for n in candidate_nodes:
                        nid = str(getattr(n, 'node_id', ''))
                        caps = node_caps[nid]
                        avail = node_avail[nid]
                        gshared = node_gpu_shared[nid]
                        print(f"    {nid}:")
                        print(f"      Capacity: cpu={caps.get('cpu', 0):.1f}, ram={caps.get('ram_gb', 0):.1f}GB, gpu={caps.get('gpu_gb', 0):.1f}GB")
                        print(f"      Available: cpu={avail.get('cpu', 0):.3f}, ram={avail.get('ram_gb', 0):.3f}GB, gpu={avail.get('gpu_gb', 0):.3f}GB")
                        needed_ram = ram + (gpu if gshared else 0.0)
                        cpu_ok = avail.get("cpu", 0.0) >= cpu
                        ram_ok = avail.get("ram_gb", 0.0) >= needed_ram
                        gpu_ok = gpu <= 0.0 or caps.get("gpu_gb", 0.0) >= gpu
                        print(f"      Feasibility: cpu_ok={cpu_ok}, ram_ok={ram_ok}, gpu_ok={gpu_ok}")
                        _, _, _, exec_time_n, period_n = self._service_demands(svc, nid)
                        if exec_time_n is not None and period_n < float('inf'):
                            time_ok = exec_time_n <= period_n
                            print(f"      Timing: exec_time={exec_time_n:.3f}s <= period={period_n:.3f}s? {time_ok}")
                
                raise RuntimeError(
                    f"No feasible candidates for service {svc_id}"
                )
            
            # Select best candidate by (energy, latency, balance)
            candidates.sort(key=lambda c: (c["energy"], c["latency"], c["balance"]))
            best = candidates[0]
            placement[str(getattr(svc, "service_id", ""))] = best["nid"]
            self._apply_allocation(svc, node_avail[best["nid"]], best["gshared"])
        
        if not placement:
            if self.verbose:
                print("[GreedyWeighted] No valid placement generated. Check assignment logic.")
                print(f"Services: {self.services}")
                print(f"Nodes: {self.nodes}")
            return {}

        # Post-placement validation uses full placement (captures cyclic/feedback latency)
        # Re-initialize baseline resources to avoid double-counting allocations
        v_caps, v_avail, v_gpu_shared = self._initialize_node_resources(exclude_services=svcs)
        placement = self._validate_and_fix_placement(placement, svcs, v_caps, v_avail, v_gpu_shared)

        return placement

    def _validate_and_fix_placement(self, placement: Dict[str, str], svcs: List[Any],
                                   node_caps: Dict[str, Dict[str, float]],
                                   node_avail: Dict[str, Dict[str, float]],
                                   node_gpu_shared: Dict[str, bool]) -> Dict[str, str]:
        """Validate placement and fix violations through reassignment."""
        # Calculate baseline usage
        baseline_usage = {}
        for nid, caps in node_caps.items():
            baseline_usage[nid] = {
                "cpu": max(0.0, caps["cpu"] - node_avail[nid]["cpu"]),
                "ram": max(0.0, caps["ram_gb"] - node_avail[nid]["ram_gb"]),
                "gpu": 0.0
            }
        
        modified = True
        max_iterations = 50
        iteration_count = 0
        
        while modified and iteration_count < max_iterations:
            modified = False
            iteration_count += 1
            
            # Recalculate current usage
            current_usage = {nid: dict(u) for nid, u in baseline_usage.items()}
            for svc in svcs:
                sid = str(getattr(svc, "service_id", ""))
                if sid in placement:
                    nid = placement[sid]
                    if nid in current_usage:
                        cpu, ram, gpu, exec_time, period = self._service_demands(svc)
                        if period > 0 and period < float('inf'):
                            cpu = cpu * min(1.0, exec_time / period)
                        gshared = node_gpu_shared[nid]
                        current_usage[nid]["cpu"] += cpu
                        current_usage[nid]["ram"] += ram + (gpu if gshared else 0.0)
                        current_usage[nid]["gpu"] += gpu
            
            # Check each service for violations
            for svc in self._topological_sort(svcs):
                sid = str(getattr(svc, "service_id", ""))
                if sid not in placement:
                    continue
                
                if self._check_and_fix_service_violation(svc, sid, placement, current_usage, 
                                                         node_caps, node_gpu_shared):
                    modified = True
                    break
        
        return placement

    def _check_and_fix_service_violation(self, svc: Any, sid: str, placement: Dict[str, str],
                                        current_usage: Dict[str, Dict[str, float]],
                                        node_caps: Dict[str, Dict[str, float]],
                                        node_gpu_shared: Dict[str, bool]) -> bool:
        """Check if service has violations and fix if needed.
        
        Returns:
            True if placement was modified
        """
        assigned_nid = placement[sid]
        caps = node_caps[assigned_nid]
        cpu, ram, gpu, exec_time, period = self._service_demands(svc)
        
        # Check utilization
        u = current_usage[assigned_nid]
        cpu_util = u["cpu"] / caps["cpu"] if caps["cpu"] > 0 else (1.0 if u["cpu"] > 0 else 0.0)
        ram_util = u["ram"] / caps["ram_gb"] if caps["ram_gb"] > 0 else (1.0 if u["ram"] > 0 else 0.0)
        gpu_util = u["gpu"] / caps["gpu_gb"] if caps.get("gpu_gb", 0) > 0 else (1.0 if u["gpu"] > 0 else 0.0)
        
        # Check period violation
        period_violation = False
        if period < float('inf'):
            assigned_idx = self._node_index.get(assigned_nid)
            if assigned_idx is not None:
                lat = self._latency_cost(svc, assigned_idx, placement)
                if exec_time + lat > period:
                    period_violation = True
        
        # Check lock violations
        lock_violation = False
        locked_nid = getattr(svc, 'locked_node_id', None)
        if locked_nid is not None and str(locked_nid) != assigned_nid:
            lock_violation = True
        locked_layer = getattr(svc, 'locked_layer_id', None)
        if locked_layer is not None:
            assigned_node = next((n for n in self.nodes if str(getattr(n, 'node_id', '')) == assigned_nid), None)
            if assigned_node and getattr(assigned_node, 'layer_id', None) != locked_layer:
                lock_violation = True
        
        # If violations, find alternative
        if cpu_util > 1.0 or ram_util > 1.0 or gpu_util > 1.0 or period_violation or lock_violation:
            best_alt_nid = self._find_alternative_node(svc, assigned_nid, placement, current_usage, 
                                                       node_caps, node_gpu_shared)
            if best_alt_nid is not None:
                placement[sid] = best_alt_nid
                if self.verbose:
                    alt_u = current_usage[best_alt_nid]
                    alt_caps = node_caps[best_alt_nid]
                    alt_util = max((alt_u["cpu"] + cpu) / alt_caps["cpu"] if alt_caps["cpu"] > 0 else 1.0,
                                 (alt_u["ram"] + ram) / alt_caps["ram_gb"] if alt_caps["ram_gb"] > 0 else 1.0)
                    print(f"[GreedyWeighted] Reassigned {sid} from {assigned_nid} to {best_alt_nid} (util={alt_util:.0%})")
                return True
        
        return False

    def _find_alternative_node(self, svc: Any, current_nid: str, placement: Dict[str, str],
                              current_usage: Dict[str, Dict[str, float]],
                              node_caps: Dict[str, Dict[str, float]],
                              node_gpu_shared: Dict[str, bool]) -> Optional[str]:
        """Find alternative node with lower utilization, preserving layer for multi-layer services."""
        locked_nid = getattr(svc, 'locked_node_id', None)
        locked_layer = getattr(svc, 'locked_layer_id', None)
        cpu, ram, gpu, exec_time, period = self._service_demands(svc)
        
        # Get current layer for multi-layer services
        current_layer = None
        if getattr(svc, 'multi_layer', False):
            current_node = next((n for n in self.nodes if str(getattr(n, 'node_id', '')) == current_nid), None)
            if current_node:
                current_layer = getattr(current_node, 'layer_id', None)
        
        best_alt_nid = None
        best_alt_util = 1.0
        
        for alt_nid in node_caps.keys():
            if alt_nid == current_nid:
                continue
            
            alt_node = next((n for n in self.nodes if str(getattr(n, 'node_id', '')) == alt_nid), None)
            if not alt_node:
                continue
            
            alt_node_layer = getattr(alt_node, 'layer_id', None)
            
            # Check locks
            if locked_nid is not None and str(locked_nid) != alt_nid:
                continue
            if locked_layer is not None and alt_node_layer != locked_layer:
                continue
            
            # Check multi-layer layer preservation
            if getattr(svc, 'multi_layer', False):
                allowed_layers = getattr(svc, 'layers', [])
                if allowed_layers and current_layer and current_layer in allowed_layers:
                    # Must stay in current layer
                    if alt_node_layer != current_layer:
                        continue
            
            # Check capacity
            alt_caps = node_caps[alt_nid]
            if gpu > 0 and alt_caps.get("gpu_gb", 0.0) < gpu:
                continue
            
            # Check period constraint - simulate placement on alternative node
            alt_idx = self._node_index.get(alt_nid)
            if alt_idx is not None and period < float('inf'):
                # Create temporary placement with service on alternative node
                temp_placement = dict(placement)
                sid_str = str(getattr(svc, 'service_id', ''))
                temp_placement[sid_str] = alt_nid
                lat = self._latency_cost(svc, alt_idx, temp_placement)
                if exec_time + lat > period:
                    continue
            
            # Calculate utilization
            alt_u = current_usage[alt_nid]
            alt_cpu_util = (alt_u["cpu"] + cpu) / alt_caps["cpu"] if alt_caps["cpu"] > 0 else 1.0
            alt_ram_util = (alt_u["ram"] + ram) / alt_caps["ram_gb"] if alt_caps["ram_gb"] > 0 else 1.0
            alt_gpu_util = (alt_u["gpu"] + gpu) / alt_caps["gpu_gb"] if alt_caps.get("gpu_gb", 0) > 0 else (1.0 if (alt_u["gpu"] + gpu) > 0 else 0.0)
            
            if alt_cpu_util > 1.0 or alt_ram_util > 1.0 or alt_gpu_util > 1.0:
                continue
            
            alt_max_util = max(alt_cpu_util, alt_ram_util, alt_gpu_util)
            if alt_max_util < best_alt_util:
                best_alt_nid = alt_nid
                best_alt_util = alt_max_util
        
        return best_alt_nid

    # ---------- Optimize ----------
    def optimize(self, services: Optional[Sequence[Any]] = None,
                 current_assignment: Optional[Dict[str, str]] = None,
                 hysteresis_threshold: float = 0.0) -> Dict[str, str]:
        """Generate optimized placement with hysteresis support.
        
        Delegates to base class which handles hysteresis logic.
        """
        # Store current assignment for use in layer preservation
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
