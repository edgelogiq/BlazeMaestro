"""DQN Environment for service orchestration."""

from typing import Any, List, Tuple, Optional, Dict
import numpy as np
from .orchestrator_utils import OrchestratorUtils

# Safety margin applied to the period budget in the soft feasibility check.
# Rejects placements where exec_time + dep_latency > period * (1 - PERIOD_JITTER_MARGIN),
# providing headroom so that jitter at runtime does not push a borderline placement
# over the period limit.
PERIOD_JITTER_MARGIN = 0.03  # 3%


class DQNEnvironment:
    """Encapsulates the RL environment for DQN orchestration."""

    def __init__(
        self,
        services: List[Any],
        nodes: List[Any],
        latency_matrix: Optional[List[List[float]]] = None,
        debug: bool = False,
        fallback_assignment: Optional[List[int]] = None,
    ):
        self.services = services
        self.nodes = nodes
        self.latency_matrix = latency_matrix or []
        self.debug = debug
        self.num_services = len(services)
        self.num_nodes = len(nodes)
        self.fallback_assignment = fallback_assignment
        self.state: Dict[str, Any] = {}

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        node_caps = []
        for i in range(self.num_nodes):
            caps = self.nodes[i].metrics.get_capacity()
            node_caps.append({
                "cpu": float(caps.get("cpu", 1.0)),
                "ram_gb": float(caps.get("ram_gb", 1.0)),
                "gpu_gb": float(caps.get("gpu_gb", 0.0)),
            })
        
        self.state = {
            "s_idx": 0,
            "node_cpu": np.zeros(self.num_nodes, dtype=np.float32),
            "node_ram": np.zeros(self.num_nodes, dtype=np.float32),
            "node_gpu": np.array([nc["gpu_gb"] for nc in node_caps], dtype=np.float32),
            "node_caps": node_caps,
            "assign": [],
            "lat": 0.0,
        }
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Get state representation with enhanced dependency latency awareness."""
        s_idx = self.state["s_idx"]
        
        # Calculate dynamic state size based on enhanced features
        # Service features: 9 (was 5) + per-node features: 10 (was 7) + global: 1
        state_size = 9 + self.num_nodes * 10 + 1
        
        if s_idx >= self.num_services:
            return np.zeros(state_size, dtype=np.float32)
        
        svc = self.services[s_idx]
        cpu, ram, gpu, exec_time, period = self._service_demands(svc)
        
        max_cpu = max((nc["cpu"] for nc in self.state["node_caps"]), default=1.0)
        max_ram = max((nc["ram_gb"] for nc in self.state["node_caps"]), default=1.0)
        max_gpu = max((nc["gpu_gb"] for nc in self.state["node_caps"]), default=1.0)
        
        svc_cpu_norm = cpu / max(max_cpu, 1.0)
        svc_ram_norm = ram / max(max_ram, 1.0)
        svc_gpu_norm = gpu / max(max_gpu, 1.0) if max_gpu > 0 else 0.0
        
        rate = 1.0 / period if period > 0 and period < float('inf') else 0.0
        
        # ENHANCEMENT: Add period constraint features
        if period > 0 and period < float('inf'):
            period_slack = max(0.0, period - exec_time)  # Time available for latency
            slack_ratio = period_slack / period  # Normalized slack (0-1)
            is_critical = 1.0 if period_slack < 0.1 else 0.0  # Less than 100ms slack
        else:
            period_slack = 1.0  # Normalized to 1.0 for no constraint
            slack_ratio = 1.0
            is_critical = 0.0
        
        # ENHANCEMENT: Check if service has dependencies
        dependencies = getattr(svc, 'dependencies', []) or []
        has_dependencies = 1.0 if dependencies else 0.0
        
        state_features = [
            s_idx / max(self.num_services, 1), 
            svc_cpu_norm, 
            svc_ram_norm, 
            svc_gpu_norm, 
            rate,
            period_slack,  # NEW
            slack_ratio,   # NEW
            is_critical,   # NEW
            has_dependencies,  # NEW
        ]
        
        # ENHANCEMENT: Calculate per-node dependency latency for current service
        per_node_dep_latency = self._calculate_per_node_dep_latency(svc, dependencies)
        
        for i in range(self.num_nodes):
            caps = self.state["node_caps"][i]
            state_features.append(self.state["node_cpu"][i] / max(caps["cpu"], 1.0))
            state_features.append(self.state["node_ram"][i] / max(caps["ram_gb"], 1.0))
            state_features.append(self.state["node_gpu"][i] / max(caps["gpu_gb"], 1.0))
            state_features.append(self.state["node_cpu"][i] / max(caps["cpu"], 1.0))
            state_features.append(self.state["node_ram"][i] / max(caps["ram_gb"], 1.0))
            state_features.append(self.state["node_gpu"][i] / max(caps["gpu_gb"], 1.0))
            exec_frac = exec_time / period if period > 0 and period < float('inf') else 0.0
            state_features.append(exec_frac)
            
            # NEW: Add per-node dependency latency (min, max, avg)
            state_features.extend(per_node_dep_latency[i])
        
        state_features.append(self.state["lat"] / 1.0)
        
        return np.array(state_features, dtype=np.float32)
    
    def _calculate_per_node_dep_latency(self, svc: Any, dependencies: List[Any]) -> List[List[float]]:
        """Calculate dependency latency stats for each candidate node.
        
        Returns:
            List of [min_latency, max_latency, avg_latency] for each node
        """
        result = []
        
        if not dependencies or not self.latency_matrix:
            # No dependencies or no latency data: return zeros
            return [[0.0, 0.0, 0.0] for _ in range(self.num_nodes)]
        
        # Find where dependencies are currently placed
        dep_node_indices = []
        for dep in dependencies:
            dep_id = getattr(dep, 'service_id', getattr(dep, 'service', None))
            if dep_id is None:
                continue
            
            # Find dependency service index
            dep_svc_idx = None
            for i, s in enumerate(self.services):
                if str(getattr(s, 'service_id', '')) == str(dep_id):
                    dep_svc_idx = i
                    break
            
            if dep_svc_idx is None:
                continue
            
            # Check if dependency is already assigned
            if dep_svc_idx < len(self.state["assign"]):
                dep_node_idx = self.state["assign"][dep_svc_idx]
                dep_node_indices.append(dep_node_idx)
            else:
                # Dependency not yet placed - check if it has a locked node
                dep_svc = self.services[dep_svc_idx]
                locked_nid = getattr(dep_svc, 'locked_node_id', None)
                if locked_nid:
                    for nidx, node in enumerate(self.nodes):
                        if str(getattr(node, 'node_id', '')) == str(locked_nid):
                            dep_node_indices.append(nidx)
                            break
        
        # Calculate latency stats for each candidate node
        for target_node_idx in range(self.num_nodes):
            if not dep_node_indices:
                result.append([0.0, 0.0, 0.0])
                continue
            
            latencies = []
            for dep_node_idx in dep_node_indices:
                try:
                    lat = float(self.latency_matrix[target_node_idx][dep_node_idx])
                    latencies.append(lat)
                except (IndexError, TypeError, ValueError):
                    pass
            
            if latencies:
                result.append([min(latencies), max(latencies), sum(latencies) / len(latencies)])
            else:
                result.append([0.0, 0.0, 0.0])
        
        return result

    def get_valid_actions(
        self,
        allow_violations: bool = False,
        allow_period_violations: bool = False,
        dependency_wait_func = None,
    ) -> List[int]:
        """Get valid node indices for current service."""
        s_idx = self.state["s_idx"]
        if s_idx >= self.num_services:
            return list(range(self.num_nodes))
        
        svc = self.services[s_idx]
        locked_nid = getattr(svc, 'locked_node_id', None)
        locked_layer = getattr(svc, 'locked_layer_id', None)
        valid_indices = []
        
        utilization_limit = 0.75 if not allow_violations else 0.95
        
        for n_idx in range(self.num_nodes):
            node = self.nodes[n_idx]
            if locked_nid is not None and str(getattr(node, 'node_id', '')) != str(locked_nid):
                continue
            if locked_layer is not None and getattr(node, 'layer_id', None) != locked_layer:
                continue
            
            node_id = str(getattr(node, 'node_id', ''))
            cpu, ram, gpu, exec_time, period = self._service_demands(svc, node_id)
            gpu_shared = bool(self.nodes[0].metrics.get_property("gpu_shared") or getattr(self.nodes[0], "gpu_shared", False)) if self.num_nodes > 0 else False
            ram_needed = ram + (gpu if gpu_shared else 0.0)

            if period > 0 and period < float('inf'):
                cpu_utilization_factor = min(1.0, exec_time / period)
                effective_cpu_needed = cpu * cpu_utilization_factor
            else:
                effective_cpu_needed = cpu

            caps = self.state["node_caps"][n_idx]
            used_cpu = self.state["node_cpu"][n_idx]
            used_ram = self.state["node_ram"][n_idx]
            avail_gpu = self.state["node_gpu"][n_idx]
            
            cpu_ok = (used_cpu + effective_cpu_needed) <= caps["cpu"] * utilization_limit
            ram_ok = (used_ram + ram_needed) <= caps["ram_gb"] * utilization_limit
            gpu_ok = gpu == 0 or avail_gpu >= gpu
            
            # Period check - TWO LEVELS:
            # 1. HARD: exec_time must fit within period (hard real-time constraint)
            # 2. SOFT: exec_time + hard_dep_latency must fit (only 'hard' constraint_level dependencies)
            period_ok = True
            if not allow_period_violations:
                if period > 0 and period < float('inf'):
                    # Hard constraint: execution time must fit in period
                    if exec_time > period:
                        period_ok = False
                    # Soft constraint: only hard dependencies count toward period budget.
                    # A jitter safety margin is subtracted from the period so that
                    # borderline placements are rejected before runtime jitter pushes
                    # them over the real deadline.
                    elif dependency_wait_func:
                        dep_wait_hard = dependency_wait_func(
                            s_idx,
                            n_idx,
                            self.state["assign"],
                            fallback_assignment=self.fallback_assignment,
                            hard_only=True,  # Only count 'hard' constraint_level dependencies
                        )
                        current_time = exec_time + dep_wait_hard
                        effective_period = period * (1.0 - PERIOD_JITTER_MARGIN)
                        period_ok = current_time <= effective_period
            
            if cpu_ok and ram_ok and gpu_ok and period_ok:
                valid_indices.append(n_idx)
        
        return valid_indices

    def step(
        self,
        action: int,
        dependency_wait_func = None,
        nodes: List[Any] = None,
        allow_violations: bool = False,
        allow_period_violations: bool = False,
    ) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return (next_state, reward, done)."""
        s_idx = self.state["s_idx"]
        n_idx = action
        svc = self.services[s_idx]
        caps = self.state["node_caps"][n_idx]

        node_id = None
        if nodes:
            node_id = str(getattr(nodes[n_idx], "node_id", ""))
        cpu, ram, gpu, exec_time, period = self._service_demands(svc, node_id)
        gpu_shared = bool(nodes[n_idx].metrics.get_property("gpu_shared") or getattr(nodes[n_idx], "gpu_shared", False)) if nodes else False
        ram_needed = ram + (gpu if gpu_shared else 0.0)
        
        if period > 0 and period < float('inf'):
            cpu_utilization_factor = min(1.0, exec_time / period)
            effective_cpu = cpu * cpu_utilization_factor
        else:
            effective_cpu = cpu
        
        used_cpu = self.state["node_cpu"][n_idx]
        used_ram = self.state["node_ram"][n_idx]
        avail_gpu = self.state["node_gpu"][n_idx]
        
        resource_violation_penalty = 2000.0
        period_violation_penalty = 1000.0
        
        # Check violations (align with get_valid_actions)
        dep_wait = 0.0
        if dependency_wait_func:
            dep_wait = dependency_wait_func(
                s_idx,
                n_idx,
                self.state["assign"],
                fallback_assignment=self.fallback_assignment,
            )
        penalty = 0.0
        has_period_violation = period > 0 and period < float('inf') and (exec_time + dep_wait) > period
        
        if has_period_violation:
            penalty += period_violation_penalty
            if not allow_violations:
                return self.get_state(), -10000.0, True  # Hard reject: -10000 penalty
        if caps["cpu"] - used_cpu < effective_cpu:
            penalty += resource_violation_penalty
            if not allow_violations:
                return self.get_state(), -penalty, True
        if caps["ram_gb"] - used_ram < ram_needed:
            penalty += resource_violation_penalty
            if not allow_violations:
                return self.get_state(), -penalty, True
        if gpu > 0.0 and avail_gpu < gpu:
            penalty += resource_violation_penalty
            if not allow_violations:
                return self.get_state(), -penalty, True
        
        # Update state
        self.state["node_cpu"][n_idx] += effective_cpu
        self.state["node_ram"][n_idx] += ram_needed
        self.state["node_gpu"][n_idx] -= gpu
        
        # Calculate latency
        
        if s_idx > 0:
            prev = self.state["assign"][-1]
            try:
                self.state["lat"] += float(self.latency_matrix[prev][n_idx]) if self.latency_matrix else 0.0
            except (IndexError, TypeError):
                pass
        
        self.state["lat"] += dep_wait
        
        # Calculate reward
        current_time = exec_time + dep_wait
        period_violation = max(0.0, current_time - period)
        
        reward = -penalty
        # Always discourage dependency latency, even if period is loose.
        # This pushes the policy to keep dependent services closer.
        if dep_wait > 0:
            dep_latency_weight = 2000.0  # seconds -> strong penalty
            reward -= dep_latency_weight * dep_wait
        if period > 0 and period < float('inf'):
            # CRITICAL CHANGE: Much stronger period violation penalties to prevent deployment failures
            # Goal: Make the model actively avoid period violations, not just accept them with penalties
            
            if period_violation > 0:
                # Severe exponential penalty for ANY period violation
                violation_severity = period_violation / period
                # Quadratic penalty grows exponentially (1% over = 10 penalty, 5% over = 250 penalty)
                period_pen = 10000.0 * (violation_severity ** 2)
            else:
                period_pen = 0.0
            
            # Tighter margin threshold: warn at 15% instead of 20% to prevent marginal cases
            period_margin = period - current_time
            margin_ratio = period_margin / period
            if margin_ratio < 0.15:
                # Strong penalty when margin drops below 15% to prevent creeping into violations
                margin_penalty = 1000.0 * ((0.15 - margin_ratio) / 0.15) ** 2
                period_pen += margin_penalty
            
            # Much stronger dependency latency penalty (was 1000.0)
            dep_ratio = dep_wait / period
            period_pen += 5000.0 * (dep_ratio ** 2)
            
            reward -= period_pen
        
        # Node diversity bonus (kept smaller to not override period constraints)
        node_usage_count = sum(1 for i in range(self.num_nodes) if self.state["node_cpu"][i] > 0)
        reward += node_usage_count * 1.0  # Reduced from 2.0

        # Energy soft objective: penalize placing services on battery-powered nodes.
        # Uses the shared OrchestratorUtils.estimate_energy() formula (power_active * exec_time).
        # Grid-powered nodes return 0.0 so this only affects battery-aware deployments.
        if nodes and n_idx < len(nodes):
            energy_cost = OrchestratorUtils.estimate_energy(svc, nodes[n_idx])
            if energy_cost > 0.0:
                reward -= 100.0 * energy_cost
        
        self.state["assign"].append(n_idx)
        self.state["s_idx"] += 1
        done = self.state["s_idx"] >= self.num_services

        if done:
            assignment = {}
            for i, svc in enumerate(self.services):
                sid = str(getattr(svc, "service_id", i))
                if i < len(self.state["assign"]):
                    nid = str(getattr(self.nodes[self.state["assign"][i]], "node_id", self.state["assign"][i]))
                    assignment[sid] = nid
            violations = OrchestratorUtils(self.services, self.nodes, self.latency_matrix).validate_placement(assignment)
            if violations:
                reward -= 50000.0 * len(violations)  # Massive penalty for any final violations

            # Critical-path penalty to minimize end-to-end latency.
            crit_path = self._critical_path_latency(assignment)
            if crit_path > 0.0:
                reward -= 2000.0 * crit_path

        return self.get_state(), reward, done

    def _service_demands(self, svc: Any, node_id: Optional[str] = None) -> Tuple[float, float, float, float, float]:
        """Extract service demands.

        Uses unified implementation from OrchestratorUtils.
        """
        return OrchestratorUtils.calculate_service_demands(svc, node_id)

    def _critical_path_latency(self, assignment: Dict[str, str]) -> float:
        """Calculate critical path latency for a placement.

        Uses full dependency latency to capture end-to-end timing impact.
        """
        if not assignment:
            return 0.0

        utils = OrchestratorUtils(self.services, self.nodes, self.latency_matrix)
        service_map = {str(getattr(svc, "service_id", "")): svc for svc in self.services}

        service_latencies: Dict[str, float] = {}
        for svc_id, node_id in assignment.items():
            svc = service_map.get(svc_id)
            if svc is None:
                continue
            node_idx = utils.node_index_map.get(str(node_id))
            if node_idx is None:
                continue
            _, _, _, exec_time, _ = utils._service_demands(svc, str(node_id))
            dep_latency = utils._latency_cost_full(svc, node_idx, assignment)
            service_latencies[svc_id] = float(exec_time) + float(dep_latency)

        if not service_latencies:
            return 0.0

        visiting: set[str] = set()
        memo: Dict[str, float] = {}

        def dfs(svc_id: str) -> float:
            if svc_id in memo:
                return memo[svc_id]
            if svc_id in visiting:
                return 0.0
            visiting.add(svc_id)
            svc = service_map.get(svc_id)
            if svc is None:
                visiting.remove(svc_id)
                memo[svc_id] = 0.0
                return 0.0

            max_dep = 0.0
            for dep in getattr(svc, "dependencies", []) or []:
                dep_id = getattr(dep, "service_id", None)
                if isinstance(dep_id, str) and dep_id in service_latencies:
                    max_dep = max(max_dep, dfs(dep_id))

            total = max_dep + service_latencies.get(svc_id, 0.0)
            memo[svc_id] = total
            visiting.remove(svc_id)
            return total

        return max(dfs(sid) for sid in service_latencies.keys())

    def get_assignment(self) -> List[int]:
        """Get current assignment."""
        return list(self.state["assign"])
