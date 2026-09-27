"""
Orchestrator Utilities

Shared utility methods for orchestrators and orchestration strategies.
These methods are used by both concrete orchestrators and the OrchestrationContext.
"""

from typing import Any, List, Optional, Dict
from BlazeAIoT.Core.Utils import BlazeLogger

# Safety margin applied to the period budget in the soft constraint revalidation check.
# A placement whose exec_time + hard_dep_latency exceeds period * (1 - PERIOD_JITTER_MARGIN)
# is reported as a soft-period violation, allowing the hysteresis mechanism to trigger
# reassignment before runtime jitter actually pushes the latency over the deadline.
PERIOD_JITTER_MARGIN = 0.03  # 3%


class OrchestratorUtils:
    """Utilities for orchestrator implementations."""
    
    def __init__(self, services, nodes, latency_matrix):
        self.services = services
        self.nodes = nodes
        self.latency_matrix = latency_matrix
        self.node_map = {str(getattr(n, 'node_id', '')): n for n in nodes}
        self.node_index_map = {str(getattr(n, 'node_id', '')): i for i, n in enumerate(nodes)}

    @staticmethod
    def calculate_service_demands(svc: Any, node_id: Optional[str] = None) -> tuple:
        """Calculate service resource demands - unified implementation for all orchestrators.
        
        Uses the most realistic approach:
        - CPU/RAM: Prefers last observed, then average, then template spec
        - Execution time: Per-node if available, otherwise container max
        - Period: Service-level execution_rate_hz (task requirement, not hardware-dependent)
        
        Args:
            svc: Service object
            node_id: Optional node ID for per-node demands
            
        Returns:
            Tuple of (cpu, ram, gpu, exec_time, period)
        """
        cpu = 0.0
        ram = 0.0
        
        # Aggregate container requirements - prefer realistic observed values
        for c in getattr(svc, "containers", []):
            cpu_min = getattr(c, "cpu_usage_min", None)
            cpu_max = getattr(c, "cpu_usage_max", None)
            cpu_last = getattr(c, "cpu_usage_last", None)
            ram_min = getattr(c, "ram_usage_min_gb", None)
            ram_max = getattr(c, "ram_usage_max_gb", None)
            ram_last = getattr(c, "ram_usage_last_gb", None)

            # CPU: prefer last observed (current steady-state), then average, then spec
            if cpu_last is not None:
                cpu += float(cpu_last)
            elif cpu_min is not None and cpu_max is not None:
                cpu += float((cpu_min + cpu_max) / 2.0)
            else:
                cpu += float(getattr(c, "cpu", 0.0))

            # RAM: same strategy
            if ram_last is not None:
                ram += float(ram_last)
            elif ram_min is not None and ram_max is not None:
                ram += float((ram_min + ram_max) / 2.0)
            else:
                ram += float(getattr(c, "ram_gb", 0.0))
        
        gpu = sum(getattr(c, "gpu_gb", 0.0) for c in getattr(svc, "containers", []))

        # Get execution time (prefer get_execution_time() if available)
        exec_times = [c.exec_time for c in getattr(svc, "containers", []) if c.exec_time is not None]
        exec_time = max(exec_times) if exec_times else None
        if hasattr(svc, 'get_execution_time'):
            exec_time, is_reliable = svc.get_execution_time()
            if exec_time is not None and not is_reliable:
                exec_time *= 1.1  # 10% buffer for unreliable timing

        # Use service-level execution_rate_hz by default (task requirement)
        execution_rate_hz = float(getattr(svc, "execution_rate_hz", 1.0))

        # Use per-node exec_time and execution_rate_hz if available (hardware dependent)
        if node_id:
            node_usage = getattr(svc, "instance_usage_by_node", {}).get(node_id)
            if node_usage:
                if node_usage.get("exec_time_max") is not None:
                    exec_time = float(node_usage["exec_time_max"])
                if node_usage.get("execution_rate_hz_max") is not None:
                    execution_rate_hz = float(node_usage["execution_rate_hz_max"])

        # Calculate period from execution rate
        period = 1.0 / execution_rate_hz if execution_rate_hz > 0 else float('inf')

        # Return 0.0 for exec_time if None (non-time-constrained service)
        return float(cpu), float(ram), float(gpu), float(exec_time if exec_time is not None else 0.0), float(period)

    def validate_placement(self, placement: Dict[str, str], log_warnings: bool = True) -> List[str]:
        """Validate a placement against all constraints."""
        violations = []
        logger = BlazeLogger.get_instance()
        for svc in self.services:
            if getattr(svc, 'exclude_from_orchestration', False):
                continue
            if getattr(svc, 'is_daemonset', False):
                continue
            sid = str(getattr(svc, 'service_id', ''))
            nid = placement.get(sid)
            if not nid:
                violations.append(f"Service {sid} not placed")
                continue
            
            node = self.node_map.get(nid)
            if not node:
                violations.append(f"Node {nid} for service {sid} not found")
                continue
            
            # Check period
            _, _, _, exec_time, period = self._service_demands(svc, nid)
            if period < float('inf'):
                # HARD CONSTRAINT: Local execution time must fit within rate period
                if exec_time > period:
                    violations.append(f"Hard period violation for {sid}: exec_time={exec_time:.2f}s > period={period:.2f}s")
                    if log_warnings:
                        logger.warning(
                            f"[HARD_PERIOD_VIOLATION] {sid}@{nid}: exec_time={exec_time:.6f}s > period={period:.6f}s"
                        )
                
                # SOFT CONSTRAINT: Only hard dependencies count toward period budget.
                # A jitter safety margin is applied so that placements with insufficient
                # slack are flagged before runtime jitter pushes them over the deadline.
                node_idx = self.node_index_map.get(nid)
                if node_idx is not None:
                    dep_latency_hard = self._latency_cost_hard_only(svc, node_idx, placement)
                    total_time = exec_time + dep_latency_hard
                    effective_period = period * (1.0 - PERIOD_JITTER_MARGIN)
                    if total_time > effective_period:
                        violations.append(
                            f"Soft period violation for {sid}: {total_time:.2f}s > {effective_period:.2f}s "
                            f"(hard deps only, {PERIOD_JITTER_MARGIN:.0%} jitter margin)"
                        )
                        if log_warnings:
                            logger.warning(
                                f"[SOFT_PERIOD_VIOLATION] {sid}@{nid}: exec_time={exec_time:.6f}s, "
                                f"hard_dep_latency={dep_latency_hard:.6f}s, total={total_time:.6f}s > "
                                f"effective_period={effective_period:.6f}s (margin={PERIOD_JITTER_MARGIN:.0%})"
                            )
        return violations

    def _service_demands(self, svc: Any, node_id: Optional[str] = None) -> tuple:
        """Calculate service resource demands.

        Uses unified static implementation. Kept as instance method for backward compatibility.
        """
        return self.calculate_service_demands(svc, node_id)

    def _latency_cost_hard_only(self, svc: Any, candidate_idx: int, placement: Dict[str, str]) -> float:
        """Calculate latency for only 'hard' constraint-level dependencies (timing-critical)."""
        svc_map = {getattr(s, "service_id", str(i)): s for i, s in enumerate(self.services)}
        visiting: set[str] = set()
        root_id = str(getattr(svc, "service_id", ""))

        def dfs(cur_svc: Any, cur_idx: int) -> float:
            total = 0.0
            for dep in getattr(cur_svc, "dependencies", []) or []:
                if getattr(dep, "blocking", True) is False:
                    continue
                # Only include dependencies with constraint_level='hard'
                constraint_level = getattr(dep, "constraint_level", "hard")
                if constraint_level != "hard":
                    continue
                
                dep_id = getattr(dep, "service_id", None)
                if not isinstance(dep_id, str): continue
                upstream_nid = placement.get(dep_id)
                if upstream_nid is None: continue
                upstream_idx = self.node_index_map.get(upstream_nid)
                if upstream_idx is None: continue
                
                try:
                    net_lat = float(self.latency_matrix[upstream_idx][cur_idx])
                    transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                except Exception:
                    net_lat, transfer_time = 0.0, 0.0
                
                edge_cost = net_lat + transfer_time
                if dep_id in visiting:
                    total += edge_cost
                    continue
                
                visiting.add(dep_id)
                dep_svc = svc_map.get(dep_id)
                if dep_svc:
                    total += edge_cost + dfs(dep_svc, upstream_idx)
                else:
                    total += edge_cost
                visiting.remove(dep_id)
            return total

        total = dfs(svc, candidate_idx)

        # Also check feedback loop dependencies with 'hard' constraint level
        if getattr(svc, "control_loop", False):
            for down_svc in self.services:
                for dep in getattr(down_svc, "dependencies", []) or []:
                    if getattr(dep, "feedback", False) is not True:
                        continue
                    constraint_level = getattr(dep, "constraint_level", "hard")
                    if constraint_level != "hard":
                        continue
                    if getattr(dep, "service_id", None) != root_id:
                        continue
                    down_id = str(getattr(down_svc, "service_id", ""))
                    down_nid = placement.get(down_id)
                    if down_nid is None:
                        continue
                    down_idx = self.node_index_map.get(down_nid)
                    if down_idx is None:
                        continue
                    try:
                        net_lat = float(self.latency_matrix[candidate_idx][down_idx])
                        transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                    except Exception:
                        net_lat, transfer_time = 0.0, 0.0
                    total += net_lat + transfer_time

        return total

    def _latency_cost_full(self, svc: Any, candidate_idx: int, placement: Dict[str, str]) -> float:
        """Calculate latency including the entire dependency tree."""
        svc_map = {getattr(s, "service_id", str(i)): s for i, s in enumerate(self.services)}
        visiting: set[str] = set()
        root_id = str(getattr(svc, "service_id", ""))

        def dfs(cur_svc: Any, cur_idx: int) -> float:
            total = 0.0
            for dep in getattr(cur_svc, "dependencies", []) or []:
                if getattr(dep, "blocking", True) is False:
                    continue
                dep_id = getattr(dep, "service_id", None)
                if not isinstance(dep_id, str): continue
                upstream_nid = placement.get(dep_id)
                if upstream_nid is None: continue
                upstream_idx = self.node_index_map.get(upstream_nid)
                if upstream_idx is None: continue
                
                try:
                    net_lat = float(self.latency_matrix[upstream_idx][cur_idx])
                    transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                except Exception:
                    net_lat, transfer_time = 0.0, 0.0
                
                edge_cost = net_lat + transfer_time
                if dep_id in visiting:
                    total += edge_cost
                    continue
                
                visiting.add(dep_id)
                dep_svc = svc_map.get(dep_id)
                if dep_svc:
                    total += edge_cost + dfs(dep_svc, upstream_idx)
                else:
                    total += edge_cost
                visiting.remove(dep_id)
            return total

        total = dfs(svc, candidate_idx)

        if getattr(svc, "control_loop", False):
            for down_svc in self.services:
                for dep in getattr(down_svc, "dependencies", []) or []:
                    if getattr(dep, "feedback", False) is not True:
                        continue
                    if getattr(dep, "service_id", None) != root_id:
                        continue
                    down_id = str(getattr(down_svc, "service_id", ""))
                    down_nid = placement.get(down_id)
                    if down_nid is None:
                        continue
                    down_idx = self.node_index_map.get(down_nid)
                    if down_idx is None:
                        continue
                    try:
                        net_lat = float(self.latency_matrix[candidate_idx][down_idx])
                        transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                    except Exception:
                        net_lat, transfer_time = 0.0, 0.0
                    total += net_lat + transfer_time

        return total
    
    @staticmethod
    def get_latency_matrix(metrics_facade: Any) -> Optional[List[List[float]]]:
        """Fetch latency matrix from metrics facade.
        
        Args:
            metrics_facade: Metrics facade with get_latency_matrix() method
            
        Returns:
            Latency matrix or None if not available
        """
        try:
            if metrics_facade and hasattr(metrics_facade, 'get_latency_matrix'):
                return metrics_facade.get_latency_matrix()
        except Exception as e:
            BlazeLogger.get_instance().debug(f"Failed to get latency matrix: {e}")
        return None
    
    @staticmethod
    def estimate_energy(service: Any, node: Any) -> float:
        """Estimate energy cost for placing a service on a node.
        
        Uses battery awareness if available, otherwise returns 0 (grid-powered).
        
        Args:
            service: Service object
            node: Node object
            
        Returns:
            Energy cost in Joules
        """
        try:
            # Check if node is battery-powered
            battery_powered = getattr(node, 'battery_powered', False)
            if not battery_powered:
                return 0.0  # Grid-powered node (no battery cost)
            
            # Battery-powered node - estimate energy consumption
            # Use active power consumption for services
            power_w = float(getattr(node, 'power_active', 5.0))
            
            # Service execution time (seconds)
            exec_time = 0.0
            if hasattr(service, 'get_execution_time'):
                exec_time, _ = service.get_execution_time()
                # Handle non-time-constrained services
                if exec_time is None:
                    exec_time = 0.01  # Use minimal default (10ms)
            else:
                containers = getattr(service, 'containers', [])
                exec_times = [c.exec_time for c in containers if c.exec_time is not None]
                exec_time = max(exec_times) if exec_times else 0.01
            
            # Energy = Power * Time (in Joules)
            energy = power_w * exec_time
            return float(energy)
            
        except Exception as e:
            BlazeLogger.get_instance().debug(f"Failed to estimate energy: {e}")
            return 0.0
    
    @staticmethod
    def count_node_capacity_violations(services: List[Any], nodes: List[Any], 
                                      assignment: List[int]) -> int:
        """Count how many nodes exceed capacity.
        
        Args:
            services: List of services
            nodes: List of nodes
            assignment: Service -> node index assignment
            
        Returns:
            Number of nodes with capacity violations
        """
        violations = 0
        node_usage = {}
        
        for s_idx, n_idx in enumerate(assignment):
            service = services[s_idx]
            node = nodes[n_idx]
            
            # Aggregate container requirements
            cpu = sum(getattr(c, "cpu", 0) for c in getattr(service, "containers", []))
            ram = sum(getattr(c, "ram_gb", 0.0) for c in getattr(service, "containers", []))
            gpu = sum(getattr(c, "gpu_gb", 0.0) for c in getattr(service, "containers", []))
            
            if n_idx not in node_usage:
                node_usage[n_idx] = {'cpu': 0.0, 'ram': 0.0, 'gpu': 0.0}
            
            node_usage[n_idx]['cpu'] += cpu
            node_usage[n_idx]['ram'] += ram
            node_usage[n_idx]['gpu'] += gpu
        
        for n_idx, usage in node_usage.items():
            node = nodes[n_idx]
            caps = node.metrics.get_capacity()
            if usage['cpu'] > caps.get('cpu', 0.0) or \
               usage['ram'] > caps.get('ram_gb', 0.0) or \
               usage['gpu'] > caps.get('gpu_gb', 0.0):
                violations += 1
                
        return violations
