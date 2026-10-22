"""
Orchestration Strategy Pattern Implementation

This module provides the Strategy pattern interfaces for service placement algorithms.
Strategies are interchangeable algorithms that can be selected at runtime.

This also consolidates common orchestration logic (hysteresis, scoring, energy estimation)
that was previously in BaseOrchestrator.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence, Any, Optional
import os
from BlazeAIoT.Core.Utils import BlazeLogger
from .orchestrator_utils import OrchestratorUtils


class OrchestrationStrategy(ABC):
    """
    Abstract base for orchestration strategies.
    
    Each strategy encapsulates a different service placement algorithm
    (NSGA-II, DQN, Unsupervised, etc).
    """
    
    @abstractmethod
    def propose_assignment(self, context: 'OrchestrationContext', 
                          services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """
        Generate a service placement assignment.
        
        Args:
            context: OrchestrationContext with services, nodes, metrics, config
            services: Optional subset of services to optimize (if None, use all)
        
        Returns:
            Dict mapping service_id -> node_id
        """
        pass
    
    @abstractmethod
    def run_single_episode_training(self, all_nodes: Sequence[Any], 
                                   services: Sequence[Any], 
                                   latency: List[List[float]]) -> int:
        """
        Run a single training episode (if applicable).
        Used for RL-based orchestrators.
        
        Args:
            all_nodes: List of all available nodes
            services: List of services to orchestrate
            latency: 2D list/array of network latencies between nodes
        
        Returns:
            int: Number of violations after training episode
        """
        pass


class OrchestrationContext:
    """
    Context for orchestration strategies.
    
    Encapsulates shared data and common operations (hysteresis, scoring, etc)
    that strategies use during optimization.
    """
    
    def __init__(self, services: Sequence[Any], nodes: Sequence[Any], 
                 metrics: Any, config: Dict[str, Any] = None):
        """
        Initialize orchestration context.
        
        Args:
            services: List of services to orchestrate
            nodes: List of available nodes
            metrics: Metrics provider for latency lookups
            config: Configuration dict
        """
        self.services = services
        self.nodes = nodes
        self.metrics = metrics
        self.config = config or {}
        
        self.num_services = len(self.services)
        self.num_nodes = len(self.nodes)
        
        # Track placement history
        self._placement_history = {}
        self._history_window = 3
    
    def optimize(self, strategy: OrchestrationStrategy,
                current_assignment: Optional[Dict[str, str]] = None,
                hysteresis_threshold: float = 0.0,
                services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """
        Run orchestration using the provided strategy.
        
        Args:
            strategy: OrchestrationStrategy to use
            current_assignment: Current placement (for hysteresis)
            hysteresis_threshold: Threshold for avoiding unnecessary migrations
            services: Optional subset of services (if None, optimize all)
        
        Returns:
            Dict mapping service_id -> node_id
        """
        # Get proposal from strategy
        proposal = strategy.propose_assignment(self, services)
        
        # Apply hysteresis to avoid flip-flops
        result = self._apply_hysteresis(proposal, current_assignment, hysteresis_threshold)
        
        # Update placement history
        self._update_placement_history(result)
        
        return result
    
    def _apply_hysteresis(self, new_assignment: Dict[str, str], 
                         current_assignment: Optional[Dict[str, str]], 
                         hysteresis_threshold: float) -> Dict[str, str]:
        """
        Apply hysteresis to reduce unnecessary migrations.
        
        Smart reassignment policy with flip-flop prevention:
        1. If current assignment has violations → ALWAYS reassign (fix violations immediately)
        2. If current assignment is violation-free → Only reassign if improvement > threshold
        3. ANTI-FLIP-FLOP: Penalize returning to recently-used nodes (within last 2 steps)
        
        This balances stability (avoid unnecessary reassignments) with correctness (fix violations ASAP)
        while preventing oscillation between nodes.
        """
        if current_assignment is not None and len(current_assignment) == len(new_assignment):
            # Score BOTH assignments with current service parameters (handles dynamic scenarios)
            new_violations, new_energy = self._assignment_score(new_assignment)
            cur_violations, cur_energy = self._assignment_score(current_assignment)
            
            if os.getenv('DEBUG_HYSTERESIS'):
                print(f"[HYSTERESIS] new=(violations={new_violations}, energy={new_energy:.2f}), "
                      f"cur=(violations={cur_violations}, energy={cur_energy:.2f}), threshold={hysteresis_threshold}")
            
            # Case 1: Current has violations → Always reassign if new is better
            if cur_violations > 0:
                if new_violations <= cur_violations:
                    # Never keep an infeasible placement: any candidate that does not increase the
                    # violation count is adopted immediately (anti-flip-flop only applies when feasible).
                    if os.getenv('DEBUG_HYSTERESIS'):
                        print(f"[HYSTERESIS] Reassigning to fix violations: {cur_violations} → {new_violations}")
                    self._update_placement_history(new_assignment)
                    return new_assignment
                else:
                    if os.getenv('DEBUG_HYSTERESIS'):
                        print(f"[HYSTERESIS] Keeping current (new has more violations)")
                    return current_assignment.copy()
            
            # Case 2: Current is violation-free → Only reassign if significant improvement
            if new_violations > 0:
                if os.getenv('DEBUG_HYSTERESIS'):
                    print(f"[HYSTERESIS] Keeping current (new would introduce violations)")
                return current_assignment.copy()
            
            # Case 3: Both violation-free → Check for flip-flops first
            flip_flops = self._count_flip_flops(new_assignment, current_assignment)
            if flip_flops > 0:
                effective_threshold = hysteresis_threshold * 2.0
                if os.getenv('DEBUG_HYSTERESIS'):
                    print(f"[HYSTERESIS] Detected {flip_flops} flip-flop(s), using 2x threshold: {effective_threshold:.2%}")
            else:
                effective_threshold = hysteresis_threshold
            
            # Threshold on both objective values (energy and latency), not energy alone
            utils = OrchestratorUtils(self.services, self.nodes, self.get_latency_matrix())
            accept, e_gain, l_gain = OrchestratorUtils.hysteresis_accepts(
                cur_energy, new_energy,
                utils.total_dependency_latency(current_assignment),
                utils.total_dependency_latency(new_assignment),
                effective_threshold)
            improvement_ratio = max(e_gain, l_gain)
            if accept:
                if os.getenv('DEBUG_HYSTERESIS'):
                    print(f"[HYSTERESIS] Reassigning for optimization: improvement={improvement_ratio:.2%} >= threshold={effective_threshold:.2%}")
                self._update_placement_history(new_assignment)
                return new_assignment
            else:
                if os.getenv('DEBUG_HYSTERESIS'):
                    print(f"[HYSTERESIS] Keeping current: improvement={improvement_ratio:.2%} < threshold={effective_threshold:.2%}")
                return current_assignment.copy()
        
        # Handle size mismatch or first assignment
        if not new_assignment and current_assignment:
            if os.getenv('DEBUG_HYSTERESIS'):
                print(f"[HYSTERESIS] Keeping current: new assignment is empty (orchestrator failed)")
            return current_assignment.copy()
        
        # First assignment or valid size mismatch - accept new
        self._update_placement_history(new_assignment)
        return new_assignment
    
    def _assignment_score(self, assignment: Dict[str, str], orchestrator_name: str = None) -> tuple:
        """
        Score an assignment, returning (violations, energy).
        
        Args:
            assignment: Service to node assignment mapping
            orchestrator_name: Optional name of orchestrator for logging context
        
        Violations include:
        - Missing node assignments
        - Period constraint violations (exec_time + network_latency > period)
        """
        violations = 0
        total_energy = 0.0
        logger = BlazeLogger.get_instance()

        # Build indices for dependency lookups
        service_by_id = {str(getattr(svc, 'service_id', '')): svc for svc in self.services}
        node_by_id = {str(getattr(n, 'node_id', '')): n for n in self.nodes}
        node_index_map = {str(getattr(n, 'node_id', '')): i for i, n in enumerate(self.nodes)}
        latency_matrix = self.get_latency_matrix()
        has_latency_matrix = latency_matrix is not None and len(latency_matrix) > 0

        period_utils = OrchestratorUtils(self.services, self.nodes, latency_matrix)
        for svc in self.services:
            sid = getattr(svc, 'service_id', None)
            if sid is None:
                violations += 1
                logger.warning("[ASSIGNMENT_VIOLATION] service has no service_id; cannot validate placement")
                continue
            
            # Skip services excluded from orchestration
            if getattr(svc, 'exclude_from_orchestration', False):
                continue
            
            sid_str = str(sid)
            nid = assignment.get(sid_str, None)
            if nid is None:
                violations += 1
                logger.warning(f"[ASSIGNMENT_VIOLATION] missing placement for service={sid_str}")
                continue
            
            node = node_by_id.get(nid)
            if node is None:
                violations += 1
                logger.warning(f"[ASSIGNMENT_VIOLATION] unknown node '{nid}' for service={sid_str}")
                continue
            
            # Period part of V(x), shared with every orchestrator
            if not period_utils.service_period_ok(svc, nid, assignment):
                violations += 1
            
            # Add energy cost
            total_energy += self._estimate_energy(svc, node)
        
        return violations, total_energy
    
    def _service_demands(self, svc: Any) -> tuple:
        """Extract service resource demands."""
        return (
            float(getattr(svc, 'cpu', 0.0)),
            float(getattr(svc, 'ram_gb', 0.0)),
            float(getattr(svc, 'gpu', 0.0)),
            float(getattr(svc, 'period_constraint', 0.0)),
            float(getattr(svc, 'ram_constraint', 0.0))
        )
    
    def _estimate_energy(self, service: Any, node: Any) -> float:
        """Estimate energy consumption for service on node."""
        cpu, _, _, _, _ = self._service_demands(service)
        
        # Base energy from node
        node_energy = float(getattr(node, 'idle_power', 10.0))
        
        # CPU scaling (rough estimate)
        if hasattr(node, 'max_power'):
            max_power = float(node.max_power)
            node_energy += cpu * (max_power - node_energy)
        else:
            node_energy += cpu * 50
        
        return node_energy
    
    def _update_placement_history(self, assignment: Dict[str, str]) -> None:
        """Update placement history for stability tracking."""
        for service_id, node_id in assignment.items():
            if service_id not in self._placement_history:
                self._placement_history[service_id] = []
            
            self._placement_history[service_id].append(node_id)
            
            # Keep only recent history
            if len(self._placement_history[service_id]) > self._history_window:
                self._placement_history[service_id].pop(0)
    
    def get_latency_matrix(self) -> List[List[float]]:
        """Get fresh latency matrix from metrics provider."""
        if self.metrics and hasattr(self.metrics, 'get_latency_matrix'):
            try:
                return self.metrics.get_latency_matrix()
            except:
                pass
        return [[0.0] * self.num_nodes for _ in range(self.num_nodes)]
    
    def _count_flip_flops(self, new_assignment: Dict[str, str], 
                         current_assignment: Dict[str, str]) -> int:
        """
        Count how many services would flip-flop (return to a recently-used node).
        
        A flip-flop is detected when:
        - Service moves from current node A to new node B
        - Node B was used within the last 2 placements (excluding current)
        
        Returns: Number of services that would flip-flop
        """
        flip_flop_count = 0
        
        for service_id, new_node in new_assignment.items():
            current_node = current_assignment.get(service_id)
            
            # If not changing nodes, no flip-flop
            if new_node == current_node:
                continue
            
            # Check placement history for this service
            history = self._placement_history.get(service_id, [])
            
            # Look at last 2 placements (excluding current)
            if len(history) >= 2:
                recent_nodes = history[-2:-1]
                if new_node in recent_nodes:
                    flip_flop_count += 1
                    if os.getenv('DEBUG_HYSTERESIS'):
                        print(f"[FLIP-FLOP] Service {service_id}: {current_node} → {new_node} "
                              f"(was on {new_node} recently, history: {history[-3:]})")
        
        return flip_flop_count
    
    def _estimate_energy(self, service: Any, node: Any) -> float:
        """
        Estimate energy consumption for a service on a node.
        Returns energy cost ONLY for battery-powered nodes.
        Grid-powered nodes return 0 (unlimited power).
        For battery nodes, uses actual power model:
        Power = base_power + (cpu_ratio + gpu_ratio) × active_power
        Energy = Power × execution_time
        """
        # Check if node is battery-powered
        battery_powered = getattr(node, 'battery_powered', False)
        if not battery_powered:
            return 0.0

        # Battery-powered node: calculate realistic energy consumption
        base_power = getattr(node, 'power_base', 5.0)  # watts
        active_power = getattr(node, 'power_active', 30.0)  # watts

        cpu_needed = sum(getattr(c, 'cpu', 0.0) for c in getattr(service, 'containers', []))
        gpu_needed = sum(getattr(c, 'gpu_gb', 0.0) for c in getattr(service, 'containers', []))
        
        # Use new execution time extraction method if available
        exec_time, is_reliable = (service.get_execution_time() 
                                 if hasattr(service, 'get_execution_time') 
                                 else (None, False))
        
        # Handle non-time-constrained services (exec_time is None)
        if exec_time is None:
            exec_time = 0.01  # Use minimal default (10ms) for energy calculation
        
        # If execution time is not reliable, add uncertainty factor
        if not is_reliable:
            exec_time *= 1.2

        # Robustly get cpu/gpu capacity
        node_cpu_cap = getattr(getattr(node, 'cpu', None), 'capacity', 1.0)
        node_gpu_cap = getattr(getattr(node, 'gpu', None), 'capacity', 0.0)

        cpu_ratio = cpu_needed / max(node_cpu_cap, 1.0)
        gpu_ratio = gpu_needed / max(node_gpu_cap, 1.0) if node_gpu_cap > 0 else 0.0

        # Power during task execution
        avg_power = base_power + (cpu_ratio + gpu_ratio) * active_power
        task_energy = avg_power * exec_time  # Joules

        return task_energy
    
