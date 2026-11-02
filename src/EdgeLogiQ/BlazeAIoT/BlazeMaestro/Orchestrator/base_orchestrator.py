from abc import ABC, abstractmethod
from typing import List, Sequence, Any, Optional, Dict
from BlazeAIoT.Core.Utils import BlazeLogger
import os
from ..Topology.service import Service
from ..Topology.node import Node
from .orchestrator_utils import OrchestratorUtils

# Compatible with Simulator's Service/Node types
# - services: list of Service
# - nodes: list of Node
# - latency_matrix: 2D list/array of floats in seconds

class BaseOrchestrator(ABC):
    """
    Base orchestrator class (maintained for backward compatibility).
    
    Most sophisticated logic has been consolidated into OrchestrationContext.
    Concrete orchestrators should implement _propose_assignment() and run_single_episode_training().
    
    Note: New code should use OrchestrationStrategy pattern instead.
    """
    
    def __init__(self, services: Sequence[Service], nodes: Sequence[Node], metrics: Any):
        # Filter out services that shouldn't be orchestrated
        all_services = list(services)
        self.services = [
            s for s in all_services 
            if not (getattr(s, 'is_daemonset', False) or getattr(s, 'exclude_from_orchestration', False))
        ]
        
        # Log if services were filtered
        if len(self.services) < len(all_services):
            num_filtered = len(all_services) - len(self.services)
            daemonset_count = sum(1 for s in all_services if getattr(s, 'is_daemonset', False))
            excluded_count = sum(1 for s in all_services if getattr(s, 'exclude_from_orchestration', False))
            msg = f"[Orchestrator] Filtered out {num_filtered} service(s) - not orchestrated"
            if daemonset_count:
                msg += f" ({daemonset_count} daemonset"
                if excluded_count:
                    msg += f", {excluded_count} explicitly excluded"
                msg += ")"
            elif excluded_count:
                msg += f" ({excluded_count} explicitly excluded)"
            print(msg)
        
        all_nodes = list(nodes)
        self.nodes = all_nodes
        self.metrics = metrics
        self.num_services = len(self.services)
        self.num_nodes = len(self.nodes)
        
        # Track placement history
        self._placement_history = {}
        self._history_window = 3

    def _get_latency_matrix(self):
        """Get fresh latency matrix from metrics provider."""
        if getattr(self, "latency_matrix", None) is not None:
            matrix = self.latency_matrix
            if isinstance(matrix, (list, tuple)) and len(matrix) > 0 and isinstance(matrix[0], (list, tuple)):
                return [[float(x) for x in row] for row in matrix]
        if hasattr(self.metrics, 'get_latency_matrix'):
            try:
                return self.metrics.get_latency_matrix()
            except:
                pass
        return [[0.0] * self.num_nodes for _ in range(self.num_nodes)]

    def _assignment_score(self, assignment: Dict[str, str], orchestrator_name: str = None):
        """
        Returns (violations, energy) for a given assignment.
        
        Args:
            assignment: Service to node assignment mapping
            orchestrator_name: Optional name of orchestrator for logging context
        
        DEPRECATED: Use OrchestrationContext._assignment_score() instead.
        Kept for backward compatibility with existing concrete orchestrators.
        """
        violations = 0
        total_energy = 0.0
        logger = BlazeLogger.get_instance()
        
        # Build service index for dependency lookups
        node_by_id = {str(getattr(n, 'node_id', '')): n for n in self.nodes}
        latency_matrix = self._get_latency_matrix()
        
        period_utils = OrchestratorUtils(self.services, self.nodes, self._get_latency_matrix())
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
                logger.warning(
                    f"[ASSIGNMENT_VIOLATION] missing placement for service={sid_str}"
                )
                continue
            
            node = node_by_id.get(nid)
            if node is None:
                violations += 1
                logger.warning(
                    f"[ASSIGNMENT_VIOLATION] unknown node '{nid}' for service={sid_str}"
                )
                continue
            
            # Period part of V(x), shared with every orchestrator
            if not period_utils.service_period_ok(svc, nid, assignment):
                violations += 1
            
            # Add energy cost
            total_energy += self._estimate_energy(svc, node)
        
        return violations, total_energy


    def _apply_hysteresis(self, new_assignment: Dict[str, str], current_assignment: Optional[Dict[str, str]], hysteresis_threshold: float) -> Dict[str, str]:
        """
        Apply hysteresis with flip-flop prevention.
        
        DEPRECATED: Use OrchestrationContext._apply_hysteresis() instead.
        Kept for backward compatibility with existing concrete orchestrators.
        """
        if current_assignment is not None and len(current_assignment) == len(new_assignment):
            # Re-validate the current assignment against the CURRENT environment state
            # This is critical for dynamic scenarios where the environment changes
            from .orchestrator_utils import OrchestratorUtils
            validation_utils = OrchestratorUtils(self.services, self.nodes, self._get_latency_matrix())
            cur_violations_list = validation_utils.validate_placement(current_assignment, log_warnings=False)
            cur_violations = len(cur_violations_list)

            # Feasibility = V(x) for both placements; energy = E(x) of paper Eqs. 1-3
            new_violations = len(validation_utils.validate_placement(new_assignment, log_warnings=False))
            new_energy = sum(OrchestratorUtils.placement_energy(new_assignment, self.services, self.nodes).values())
            cur_energy = sum(OrchestratorUtils.placement_energy(current_assignment, self.services, self.nodes).values())
            
            if os.getenv('DEBUG_HYSTERESIS'):
                print(f"[HYSTERESIS] new_score=(violations={new_violations}, energy={new_energy:.2f}), "
                      f"cur_score=(violations={cur_violations} (re-validated), energy={cur_energy:.2f}), threshold={hysteresis_threshold}")
            
            # Case 1: Current has violations → Always reassign if new is better or equal
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
            utils = OrchestratorUtils(self.services, self.nodes, self._get_latency_matrix())
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
        
        if not new_assignment and current_assignment:
            if os.getenv('DEBUG_HYSTERESIS'):
                print(f"[HYSTERESIS] Keeping current: new assignment is empty (orchestrator failed to place services)")
            return current_assignment.copy()
        
        # First assignment or valid size mismatch - accept new
        self._update_placement_history(new_assignment)
        return new_assignment
    
    def _count_flip_flops(self, new_assignment: Dict[str, str], current_assignment: Dict[str, str]) -> int:
        """Count how many services would flip-flop (return to a recently-used node).
        
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
            
            # Look at last 2 placements (excluding current which is last in history)
            # History format: [..., older, current]
            if len(history) >= 2:
                recent_nodes = history[-2:-1]  # Get 2nd-to-last (skip current)
                if new_node in recent_nodes:
                    flip_flop_count += 1
                    import os
                    if os.getenv('DEBUG_HYSTERESIS'):
                        print(f"[FLIP-FLOP] Service {service_id}: {current_node} → {new_node} "
                              f"(was on {new_node} recently, history: {history[-3:]})")
        
        return flip_flop_count
    
    def _update_placement_history(self, assignment: Dict[str, str]):
        """Update placement history for all services in the assignment."""
        for service_id, node_id in assignment.items():
            if service_id not in self._placement_history:
                self._placement_history[service_id] = []
            
            # Only add if different from last placement (avoid duplicates during "keep current")
            history = self._placement_history[service_id]
            if not history or history[-1] != node_id:
                history.append(node_id)
                
                # Keep only last N placements
                if len(history) > self._history_window:
                    self._placement_history[service_id] = history[-self._history_window:]

    def _estimate_energy(self, service: Service, node: Node) -> float:
        """Estimate energy consumption for a service on a node.
        Returns energy cost ONLY for battery-powered nodes.
        Grid-powered nodes return 0 (unlimited power).
        For battery nodes, uses actual power model:
        Power = base_power + (cpu_ratio + gpu_ratio) × active_power
        Energy = Power × execution_time
        """
        # Check if node is battery-powered
        battery_powered = getattr(node, 'battery_powered', False)
        if not battery_powered:
            # Grid-powered node: energy doesn't matter (return 0)
            return 0.0

        # Battery-powered node: calculate realistic energy consumption
        base_power = getattr(node, 'power_base', 5.0)  # watts
        active_power = getattr(node, 'power_active', 30.0)  # watts

        cpu_needed = sum(getattr(c, 'cpu', 0.0) for c in getattr(service, 'containers', []))
        gpu_needed = sum(getattr(c, 'gpu_gb', 0.0) for c in getattr(service, 'containers', []))
        
        # Use new execution time extraction method
        exec_time, is_reliable = service.get_execution_time() if hasattr(service, 'get_execution_time') else (None, False)
        
        # Handle non-time-constrained services (exec_time is None)
        if exec_time is None:
            exec_time = 0.01  # Use minimal default (10ms) for energy calculation
        
        # If execution time is not reliable, add uncertainty factor to energy calculation
        if not is_reliable:
            exec_time *= 1.2  # Add 20% buffer for uncertainty

        # Robustly get cpu/gpu capacity
        node_cpu_cap = getattr(getattr(node, 'cpu', None), 'capacity', 1.0)
        node_gpu_cap = getattr(getattr(node, 'gpu', None), 'capacity', 0.0)

        cpu_ratio = cpu_needed / max(node_cpu_cap, 1.0)
        gpu_ratio = gpu_needed / max(node_gpu_cap, 1.0) if node_gpu_cap > 0 else 0.0

        # Power during task execution
        avg_power = base_power + (cpu_ratio + gpu_ratio) * active_power
        task_energy = avg_power * exec_time  # Joules

        return task_energy

    def _service_demands(self, svc: Any, node_id: Optional[str] = None) -> tuple:
        """Calculate service resource demands.

        Returns tuple of (cpu, ram, gpu, exec_time, period).
        Uses unified implementation from OrchestratorUtils.

        Args:
            svc: Service object
            node_id: Optional node ID for per-node demands

        Returns:
            Tuple of resource demands (cpu, ram, gpu, exec_time, period)
        """
        return OrchestratorUtils.calculate_service_demands(svc, node_id)

    def _dependency_wait(self, s_idx: int, n_idx: int, assignment: List[int]) -> float:
        """Calculate latency waiting time due to service dependencies.

        Accounts for network latency and data transfer time for dependent services.

        Args:
            s_idx: Service index
            n_idx: Node index being assigned
            assignment: Current assignment (list of node indices)

        Returns:
            Wait time in seconds
        """
        svc = self.services[s_idx]
        deps = getattr(svc, "dependencies", []) or []
        if not deps:
            return 0.0

        wait = 0.0
        throughput_mbps = 200.0  # Assumed network throughput
        latency_matrix = self._get_latency_matrix()

        for dep in deps:
            if getattr(dep, "blocking", True) is False:
                continue
            # Find upstream service
            up_idx = None
            for i, s in enumerate(self.services):
                dep_id = getattr(dep, "service_id", dep)  # Handle both Dependency objects and IDs
                if getattr(s, "service_id", None) == dep_id:
                    up_idx = i
                    break

            if up_idx is None or up_idx >= len(assignment):
                continue

            try:
                # Get network latency between nodes
                up_node_idx = int(assignment[up_idx])
                cur_node_idx = int(n_idx)
                net_lat = float(latency_matrix[up_node_idx][cur_node_idx]) if latency_matrix is not None else 0.0
            except (IndexError, ValueError, TypeError):
                net_lat = 0.0

            # Add data transfer time
            size_mb = float(getattr(dep, "size_mb", 0.0))
            transfer = (size_mb / max(1.0, throughput_mbps)) if size_mb > 0 else 0.0
            wait += max(0.0, net_lat + transfer)

        # Add feedback leg for control-loop services
        if getattr(svc, "control_loop", False):
            try:
                latency_matrix = self._get_latency_matrix()
            except Exception:
                latency_matrix = None
            if latency_matrix is not None:
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
                            net_lat = float(latency_matrix[n_idx][down_node_idx])
                        except Exception:
                            net_lat = 0.0
                        transfer = (float(getattr(dep, "size_mb", 0.0)) / max(1.0, throughput_mbps)) if getattr(dep, "size_mb", 0.0) > 0 else 0.0
                        wait += max(0.0, net_lat + transfer)

        return wait

    def _topological_sort(self, services: Sequence[Any]) -> List[Any]:
        """Sort services in topological order based on dependencies.
        
        Uses Kahn's algorithm. Falls back to original order if cycle detected.
        
        Args:
            services: List of services to sort
            
        Returns:
            Topologically sorted list of services
        """
        svc_map = {getattr(s, "service_id", str(i)): s for i, s in enumerate(services)}
        indegree = {sid: 0 for sid in svc_map}
        graph: Dict[str, List[str]] = {sid: [] for sid in svc_map}
        
        for svc_id, svc in svc_map.items():
            for dep in getattr(svc, "dependencies", []) or []:
                dep_id = getattr(dep, "service_id", None)
                if not isinstance(dep_id, str):
                    continue
                # Order on blocking dependencies only: asynchronous feedback edges would
                # create cycles (e.g. driver <- control loop) and fall back to file order.
                if getattr(dep, "blocking", True) is False:
                    continue
                if dep_id in svc_map:
                    indegree[svc_id] += 1
                    graph.setdefault(dep_id, []).append(svc_id)
        
        # Queue of nodes with indegree 0
        q = [sid for sid, d in indegree.items() if d == 0]
        order_ids: List[str] = []
        
        while q:
            u = q.pop(0)
            order_ids.append(u)
            for v in graph.get(u, []):
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.append(v)
        
        # If cycle, fall back to original order
        if len(order_ids) != len(svc_map):
            return list(services)
        
        # Map ids back to objects
        return [svc_map[sid] for sid in order_ids if sid in svc_map]

    def _is_feasible(self, svc: Any, node_id: str, avail: Dict[str, float], 
                    caps: Dict[str, float], gpu_shared: bool) -> bool:
        """Check if service can be feasibly placed on node.
        
        Args:
            svc: Service object
            node_id: Node ID (unused but kept for interface compatibility)
            avail: Available resources on node
            caps: Node capacity
            gpu_shared: Whether GPU memory is shared with RAM
            
        Returns:
            True if feasible, False otherwise
        """
        cpu, ram, gpu, exec_time, period = self._service_demands(svc, node_id)
        needed_ram = ram + (gpu if gpu_shared else 0.0)
        
        # For period-based services, calculate effective CPU needed based on utilization
        if period > 0 and period < float('inf'):
            cpu_utilization_factor = min(1.0, exec_time / period)
            effective_cpu_needed = cpu * cpu_utilization_factor
        else:
            effective_cpu_needed = cpu
        
        # Execution rate constraint with period
        if period < float('inf') and exec_time > period:
            return False
        if avail.get("cpu", 0.0) < effective_cpu_needed:
            return False
        if avail.get("ram_gb", 0.0) < needed_ram:
            return False
        if gpu > 0.0 and caps.get("gpu_gb", 0.0) < gpu:
            return False
        return True

    def _get_service_measured_usage(self, svc: Any) -> tuple[float, float, float]:
        """Get measured resource usage for a service if available.

        Returns:
            Tuple of (cpu_used, ram_used_gb, gpu_used_gb)
        """
        cpu_used = 0.0
        ram_used = 0.0
        gpu_used = 0.0

        if hasattr(svc, 'get_measured_usage'):
            try:
                usage = svc.get_measured_usage() or {}
                for _, container_usage in usage.items():
                    if not isinstance(container_usage, dict):
                        continue
                    cpu_used += float(container_usage.get('cpu_usage_last', 0.0))
                    ram_used += float(container_usage.get('ram_usage_last_gb', 0.0))
                    gpu_used += float(container_usage.get('gpu_usage_last_gb', 0.0))
                return cpu_used, ram_used, gpu_used
            except Exception:
                pass

        # Fallback to container attributes if measured usage is unavailable
        for c in getattr(svc, "containers", []):
            cpu_last = getattr(c, "cpu_usage_last", None)
            cpu_max = getattr(c, "cpu_usage_max", None)
            if cpu_last is not None:
                cpu_used += float(cpu_last)
            elif cpu_max is not None:
                cpu_used += float(cpu_max)
            else:
                cpu_used += float(getattr(c, "cpu", 0.0))

            ram_last = getattr(c, "ram_usage_last_gb", None)
            ram_max = getattr(c, "ram_usage_max_gb", None)
            if ram_last is not None:
                ram_used += float(ram_last)
            elif ram_max is not None:
                ram_used += float(ram_max)
            else:
                ram_used += float(getattr(c, "ram_gb", 0.0))

            gpu_last = getattr(c, "gpu_usage_last_gb", None)
            gpu_max = getattr(c, "gpu_usage_max_gb", None)
            if gpu_last is not None:
                gpu_used += float(gpu_last)
            elif gpu_max is not None:
                gpu_used += float(gpu_max)
            else:
                gpu_used += float(getattr(c, "gpu_gb", 0.0))

        return cpu_used, ram_used, gpu_used

    def _load_idle_baseline(self) -> dict:
        """Load per-node idle baseline (cpu/ram) from platform.xml.

        Returns:
            Dict with keys: cpu_by_host, ram_by_host, cpu_avg, ram_avg
        """
        if hasattr(self, '_idle_baseline_cache') and self._idle_baseline_cache is not None:
            return self._idle_baseline_cache

        import os
        import xml.etree.ElementTree as ET
        import statistics

        cpu_by_host: Dict[str, float] = {}
        ram_by_host: Dict[str, float] = {}
        cpu_vals = []
        ram_vals = []

        candidate_paths = []
        metrics_path = getattr(self.metrics, 'platform_xml_path', None) if hasattr(self, 'metrics') else None
        if metrics_path:
            candidate_paths.append(metrics_path)

        # Fallback to repo config default
        candidate_paths.append(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../../../../config/platform.xml')
            )
        )

        platform_xml = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        if platform_xml:
            try:
                tree = ET.parse(platform_xml)
                root = tree.getroot()
                for host in root.findall('.//host'):
                    host_id = host.get('id') or ''
                    cpu_prop = host.find(".//prop[@id='debug.cpu_used']")
                    ram_prop = host.find(".//prop[@id='debug.ram_used_gb']")
                    if cpu_prop is not None:
                        try:
                            cpu_val = float(cpu_prop.get('value', '0') or 0)
                            cpu_by_host[host_id] = cpu_val
                            cpu_vals.append(cpu_val)
                        except Exception:
                            pass
                    if ram_prop is not None:
                        try:
                            ram_val = float(ram_prop.get('value', '0') or 0)
                            ram_by_host[host_id] = ram_val
                            ram_vals.append(ram_val)
                        except Exception:
                            pass
            except Exception:
                pass

        cpu_avg = statistics.mean(cpu_vals) if cpu_vals else 0.1
        ram_avg = statistics.mean(ram_vals) if ram_vals else 0.2

        self._idle_baseline_cache = {
            'cpu_by_host': cpu_by_host,
            'ram_by_host': ram_by_host,
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
        }
        return self._idle_baseline_cache

    def _current_load_usage(self, current_assignment: Optional[Dict[str, str]] = None,
                            exclude_services: Optional[Sequence[Any]] = None) -> Dict[str, Dict[str, float]]:
        """Compute per-node current load usage from measured service metrics.

        Args:
            current_assignment: Current service -> node placement
            exclude_services: Services to exclude from baseline load (e.g., optimizable set)

        Returns:
            Dict[node_id] -> {'cpu': x, 'ram': y, 'gpu': z}
        """
        exclude_ids = set()
        if exclude_services:
            exclude_ids = {str(getattr(s, 'service_id', '')) for s in exclude_services}

        usage: Dict[str, Dict[str, float]] = {}
        for n in self.nodes:
            nid = str(getattr(n, 'node_id', ''))
            usage[nid] = {'cpu': 0.0, 'ram': 0.0, 'gpu': 0.0}

        # Add idle baseline per node from platform.xml
        baseline = self._load_idle_baseline()
        import random
        for n in self.nodes:
            nid = str(getattr(n, 'node_id', ''))
            if nid not in usage:
                usage[nid] = {'cpu': 0.0, 'ram': 0.0, 'gpu': 0.0}
            jitter = random.uniform(0.95, 1.05)
            usage[nid]['cpu'] += float(baseline['cpu_by_host'].get(nid, baseline['cpu_avg'])) * jitter
            usage[nid]['ram'] += float(baseline['ram_by_host'].get(nid, baseline['ram_avg'])) * jitter

        if not self.services:
            return usage

        for svc in self.services:
            sid = str(getattr(svc, 'service_id', ''))
            if sid in exclude_ids:
                continue

            assigned_nodes: list[str] = []

            if current_assignment and sid in current_assignment:
                assigned_nodes = [str(current_assignment[sid])]
            else:
                locked_node_id = getattr(svc, 'locked_node_id', None)
                locked_layer_id = getattr(svc, 'locked_layer_id', None)
                is_daemonset = getattr(svc, 'is_daemonset', False)

                if locked_node_id is not None:
                    assigned_nodes = [str(locked_node_id)]
                elif is_daemonset and locked_layer_id is not None:
                    # Daemonset applies to all nodes in its layer
                    for n in self.nodes:
                        if str(getattr(n, 'layer_id', None)) == str(locked_layer_id):
                            assigned_nodes.append(str(getattr(n, 'node_id', '')))

            if not assigned_nodes:
                continue

            cpu_used, ram_used, gpu_used = self._get_service_measured_usage(svc)
            for nid in assigned_nodes:
                if nid not in usage:
                    usage[nid] = {'cpu': 0.0, 'ram': 0.0, 'gpu': 0.0}
                usage[nid]['cpu'] += cpu_used
                usage[nid]['ram'] += ram_used
                usage[nid]['gpu'] += gpu_used

        return usage

    def _latency_cost_full(self, svc: Any, candidate_idx: int, placement: Dict[str, str],
                          latency_matrix: Any, node_index: Dict[str, int], services: Sequence[Any]) -> float:
        """Calculate latency including the entire dependency tree.

        Traverses dependencies recursively so that upstream-to-downstream paths
        (e.g., A -> B -> C) accumulate all hops (A->B latency + B->C latency).
        Cycles are broken via a small visited set per call.
        """
        svc_map = {getattr(s, "service_id", str(i)): s for i, s in enumerate(services)}
        visiting: set[str] = set()
        root_id = str(getattr(svc, "service_id", ""))

        def dfs(cur_svc: Any, cur_idx: int) -> float:
            total = 0.0
            for dep in getattr(cur_svc, "dependencies", []) or []:
                if getattr(dep, "blocking", True) is False:
                    continue
                dep_id = getattr(dep, "service_id", None)
                if not isinstance(dep_id, str):
                    continue
                upstream_id = placement.get(dep_id)
                if upstream_id is None:
                    continue
                upstream_idx = node_index.get(upstream_id)
                if upstream_idx is None:
                    continue
                try:
                    net_lat = float(latency_matrix[upstream_idx][cur_idx])
                    transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                except Exception:
                    net_lat = 0.0
                    transfer_time = 0.0
                edge_cost = net_lat + transfer_time

                # Avoid cycles while still counting this edge
                if dep_id in visiting:
                    total += edge_cost
                    continue

                visiting.add(dep_id)
                dep_svc = svc_map.get(dep_id)
                if dep_svc is not None:
                    total += edge_cost + dfs(dep_svc, upstream_idx)
                else:
                    total += edge_cost
                visiting.remove(dep_id)
            return total

        total = dfs(svc, candidate_idx)

        # Add feedback leg for control-loop services (downstream command path)
        if getattr(svc, "control_loop", False):
            for down_svc in services:
                for dep in getattr(down_svc, "dependencies", []) or []:
                    if getattr(dep, "feedback", False) is not True:
                        continue
                    if getattr(dep, "service_id", None) != root_id:
                        continue
                    down_id = str(getattr(down_svc, "service_id", ""))
                    down_nid = placement.get(down_id)
                    if down_nid is None:
                        continue
                    down_idx = node_index.get(down_nid)
                    if down_idx is None:
                        continue
                    try:
                        net_lat = float(latency_matrix[candidate_idx][down_idx])
                        transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                    except Exception:
                        net_lat = 0.0
                        transfer_time = 0.0
                    total += net_lat + transfer_time

        return total

    def _latency_cost_hard_only(self, svc: Any, candidate_idx: int, placement: Dict[str, str],
                                latency_matrix: Any, node_index: Dict[str, int], services: Sequence[Any]) -> float:
        """Calculate latency for only 'hard' constraint-level dependencies (timing-critical).
        
        Similar to _latency_cost_full but only includes dependencies with constraint_level='hard'.
        This is used for period constraint checking.
        """
        svc_map = {getattr(s, "service_id", str(i)): s for i, s in enumerate(services)}
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
                if not isinstance(dep_id, str):
                    continue
                upstream_id = placement.get(dep_id)
                if upstream_id is None:
                    continue
                upstream_idx = node_index.get(upstream_id)
                if upstream_idx is None:
                    continue
                try:
                    net_lat = float(latency_matrix[upstream_idx][cur_idx])
                    transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                except Exception:
                    net_lat = 0.0
                    transfer_time = 0.0
                edge_cost = net_lat + transfer_time

                # Avoid cycles while still counting this edge
                if dep_id in visiting:
                    total += edge_cost
                    continue

                visiting.add(dep_id)
                dep_svc = svc_map.get(dep_id)
                if dep_svc is not None:
                    total += edge_cost + dfs(dep_svc, upstream_idx)
                else:
                    total += edge_cost
                visiting.remove(dep_id)
            return total

        total = dfs(svc, candidate_idx)

        # Add feedback leg for control-loop services (downstream command path)
        # Only include feedback dependencies with 'hard' constraint level
        if getattr(svc, "control_loop", False):
            for down_svc in services:
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
                    down_idx = node_index.get(down_nid)
                    if down_idx is None:
                        continue
                    try:
                        net_lat = float(latency_matrix[candidate_idx][down_idx])
                        transfer_time = (dep.size_mb / 200.0) if getattr(dep, 'size_mb', 0) > 0 else 0.0
                    except Exception:
                        net_lat = 0.0
                        transfer_time = 0.0
                    total += net_lat + transfer_time

        return total

    def _balance_cost(self, avail: Dict[str, float], caps: Dict[str, float]) -> float:
        """Calculate resource balance cost.
        
        Encourages balanced utilization across CPU and RAM.
        Smaller cost is better (perfectly balanced = 0.0).
        
        Args:
            avail: Available resources on node
            caps: Node capacity
            
        Returns:
            Balance cost (0.0 = perfect balance, 1.0 = maximum imbalance)
        """
        def frac(key: str) -> float:
            cap = max(1e-6, float(caps.get(key, 0.0)))
            return max(0.0, min(1.0, float(avail.get(key, 0.0)) / cap))
        
        cpu_f = frac("cpu")
        ram_f = frac("ram_gb")
        return float(abs(cpu_f - ram_f))

    def optimize(self, services: Optional[Sequence[Service]] = None, current_assignment: Optional[Dict[str, str]] = None, hysteresis_threshold: float = 0.0) -> Dict[str, str]:
        """
        Common orchestration flow: propose assignment, apply hysteresis, return assignment.
        Subclasses must implement _propose_assignment().
        """
        self._current_assignment = current_assignment if current_assignment else {}
        if services is not None:
            orig_services = self.services
            self.services = list(services)
            self.num_services = len(self.services)
        proposal = self._propose_assignment(services)
        result = self._apply_hysteresis(proposal, current_assignment, hysteresis_threshold)
        if services is not None:
            self.services = orig_services
            self.num_services = len(self.services)
        return result

    @abstractmethod
    def _propose_assignment(self, services: Optional[Sequence[Service]] = None) -> Dict[str, str]:
        """
        Subclasses must implement this to return a proposed assignment (service_id -> node_id).
        """
        pass

    @abstractmethod
    def run_single_episode_training(self, all_nodes, services, latency) -> int:
        """
        Subclasses must implement this to train for a single episode and return violation count.
        Used for training DQN or Unsupervised orchestrators in isolation.
        Args:
            all_nodes: List of all available nodes
            services: List of services to orchestrate
            latency: 2D list/array of network latencies between nodes
        Returns:
            int: Number of violations after training episode
        """
        pass