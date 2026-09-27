import random
import numpy as np
from typing import List

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.service_simulator import Service, ContainerSpec, DependencySpec


class DynamicScenario:
    """Manages baseline loads for nodes over time.
    
    This class provides consistent baseline load management across all simulation scripts.
    Baseline loads represent background system activity (OS, monitoring, other services)
    that exists before any orchestrator-managed services are deployed.
    
    Default baselines:
    - CPU: 10-15% of node capacity
    - RAM: 15-20% of node capacity
    - Time-varying with sinusoidal fluctuation (±2-5%)
    """
    
    def __init__(self):
        self.node_baseline_loads = {}
        
    def initialize_baseline_loads(self, nodes: List):
        """Initialize baseline loads for all nodes: 20-35% CPU and 15-40% RAM.
        
        Args:
            nodes: List of Node objects to initialize baseline loads for
        """
        for node in nodes:
            node_id = node.node_id
            cpu_capacity = getattr(node, 'cpu', 0.0)
            ram_capacity = getattr(node, 'ram', 0.0)
            
            # Base percentages: 10-15% CPU, 15-20% RAM
            base_cpu_pct = random.uniform(0.20, 0.35)
            base_ram_pct = random.uniform(0.15, 0.40)
            
            self.node_baseline_loads[node_id] = {
                'base_cpu': cpu_capacity * base_cpu_pct,
                'base_ram': ram_capacity * base_ram_pct,
                'cpu_variance': random.uniform(0.02, 0.05),  # ±2-5% variation
                'ram_variance': random.uniform(0.02, 0.05),
                'phase_offset': random.uniform(0, 2 * np.pi)  # Random phase for sine wave
            }
    
    def get_baseline_load(self, node_id: str, step: int) -> dict:
        """Get time-varying baseline load for a node at a specific time step.
        
        Args:
            node_id: Node identifier
            step: Current time step (for time-varying loads)
            
        Returns:
            Dictionary with 'cpu' and 'ram' baseline loads (absolute values, not percentages)
        """
        if node_id not in self.node_baseline_loads:
            return {'cpu': 0.0, 'ram': 0.0}
        
        baseline = self.node_baseline_loads[node_id]
        
        # Sinusoidal variation over time
        time_factor = np.sin(step * 0.5 + baseline['phase_offset'])
        
        cpu_load = baseline['base_cpu'] * (1.0 + baseline['cpu_variance'] * time_factor)
        ram_load = baseline['base_ram'] * (1.0 + baseline['ram_variance'] * time_factor)
        
        return {
            'cpu': max(0.0, cpu_load),
            'ram': max(0.0, ram_load)
        }


class TrainUtils:
    @staticmethod
    def strict_validate_placement(assignments: dict, services, nodes, latency_matrix) -> list:
        """Validate placement using strict orchestrator rules.

        Args:
            assignments: Service -> node mapping
            services: List of Service objects
            nodes: List of Node objects
            latency_matrix: Network latency matrix
        """
        if assignments is None:
            return []
        from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.orchestrator_utils import OrchestratorUtils
        return OrchestratorUtils(services, nodes, latency_matrix).validate_placement(assignments)

    @staticmethod
    def validate_locked_nodes(services, nodes):
        """Validate that all locked nodes exist in the current topology."""
        node_ids = {str(getattr(n, "node_id", "")) for n in nodes}
        missing = []
        for svc in services:
            locked_nid = getattr(svc, "locked_node_id", None)
            if locked_nid is not None and str(locked_nid) not in node_ids:
                missing.append((getattr(svc, "service_id", ""), locked_nid))
        if missing:
            details = ", ".join([f"{sid} -> {nid}" for sid, nid in missing])
            raise ValueError(f"Locked node(s) not found in platform: {details}")
    @staticmethod
    def apply_service_scenario(services, scenario_name):
        """Apply dynamic load scenarios to services (execution rates, exec times).
        
        Args:
            services: List of Service objects to modify
            scenario_name: One of 'normal', 'extreme_rate', 'heavy_exec', 'light_load', 
                          'peak_traffic', 'stress_test', 'increased_workload'
        """
        import random
        
        # Store baseline rates if not already stored
        for i in range(1, min(7, len(services))):
            if not hasattr(services[i], '_baseline_rate'):
                services[i]._baseline_rate = services[i].execution_rate_hz
            if not hasattr(services[i].containers[0], '_baseline_exec_time'):
                exec_t = services[i].containers[0].exec_time
                services[i].containers[0]._baseline_exec_time = exec_t if exec_t is not None else 0.01
        
        if scenario_name == "normal":
            # Reset to baseline rates with small variation (matches compare script T=0-10min)
            services[1].execution_rate_hz = 10.0    # imu
            services[2].execution_rate_hz = 10.0    # encoder
            services[3].execution_rate_hz = 7.0     # slam_toolbox
            services[4].execution_rate_hz = 10.0    # robot_localization
            services[5].execution_rate_hz = 10.0    # pure_pursuit
            services[6].execution_rate_hz = 10.0    # emergency_stop
            # Apply random variation to all compute services (not sensors)
            for i in range(1, 7):
                services[i].execution_rate_hz *= random.uniform(0.95, 1.05)
        
        elif scenario_name == "peak_traffic":
            # CRITICAL: Matches compare script T=10-15min (1.5x rate multiplier)
            # This is where DQN violations occur - train specifically on this pattern!
            baseline_rates = [10.0, 10.0, 7.0, 10.0, 10.0, 10.0]  # imu, encoder, slam, loc, pursuit, stop
            for i in range(1, 7):
                services[i].execution_rate_hz = baseline_rates[i-1] * 1.5 * random.uniform(0.95, 1.05)
            # robot_localization at 15Hz → period=0.0667s (tight!)
            # emergency_stop at 15Hz → period=0.0667s (tight!)
        
        elif scenario_name == "stress_test":
            # Even more aggressive than peak_traffic (2.0x multiplier)
            # Forces DQN to learn ultra-conservative placements
            baseline_rates = [10.0, 10.0, 7.0, 10.0, 10.0, 10.0]
            for i in range(1, 7):
                services[i].execution_rate_hz = baseline_rates[i-1] * 2.0 * random.uniform(0.95, 1.05)
            # robot_localization at 20Hz → period=0.05s (very tight!)
        
        elif scenario_name == "increased_workload":
            # Longer execution times (matches compare script T=15-20min)
            # Combined with normal rates creates tight margins
            for i in range(1, 7):
                services[i].execution_rate_hz = getattr(services[i], '_baseline_rate', 10.0) * random.uniform(0.95, 1.05)
                baseline_exec = getattr(services[i].containers[0], '_baseline_exec_time', None)
                if baseline_exec is None:
                    baseline_exec = services[i].containers[0].exec_time
                if baseline_exec is None:
                    baseline_exec = 0.01
                services[i].containers[0].exec_time = baseline_exec * 1.2 * random.uniform(0.95, 1.05)
        
        elif scenario_name == "extreme_rate":
            # High rates - will cause latency violations if not placed optimally
            services[1].execution_rate_hz = 30.0    # imu: 33ms period
            services[2].execution_rate_hz = 25.0    # encoder: 40ms period
            services[3].execution_rate_hz = 20.0    # slam_toolbox: 50ms period
            services[4].execution_rate_hz = 18.0    # robot_localization: 55ms period
            services[5].execution_rate_hz = 15.0    # pure_pursuit: 67ms period
            services[6].execution_rate_hz = 20.0    # emergency_stop: 50ms period
            # Apply random variation to all compute services
            for i in range(1, 7):
                services[i].execution_rate_hz *= random.uniform(0.95, 1.05)
        
        elif scenario_name == "heavy_exec":
            # Long execution times - stresses latency constraints
            services[1].execution_rate_hz = 10.0    # imu
            services[2].execution_rate_hz = 8.0     # encoder
            services[3].execution_rate_hz = 9.0     # slam_toolbox
            services[4].execution_rate_hz = 10.0    # robot_localization: longer exec time
            services[5].execution_rate_hz = 8.0     # pure_pursuit: longer exec time
            services[6].execution_rate_hz = 10.0    # emergency_stop: longer exec time
            # Increase execution times for higher-level services
            services[4].containers[0].exec_time = 0.015  # robot_localization
            services[5].containers[0].exec_time = 0.012  # pure_pursuit
            services[6].containers[0].exec_time = 0.030  # emergency_stop (higher GPU latency)
            # Apply random variation to all compute services
            for i in range(1, 7):
                services[i].execution_rate_hz *= random.uniform(0.95, 1.05)
        
        elif scenario_name == "light_load":
            # Low rates - opportunity for energy optimization (matches compare script T=25-30min)
            baseline_rates = [10.0, 10.0, 7.0, 10.0, 10.0, 10.0]
            for i in range(1, 7):
                services[i].execution_rate_hz = baseline_rates[i-1] * 0.5 * random.uniform(0.95, 1.05)
    
    @staticmethod
    def build_services():
        services = [
            # Sensors on frank-1 (realistic ranges + exec_time)
            Service(
                "camera",
                [ContainerSpec(
                    "camera",
                    cpu=0.5,
                    ram_gb=0.12,
                    exec_time=0.01,
                    cpu_usage_min=0.20,
                    cpu_usage_max=0.80,
                    cpu_usage_last=0.50,
                    ram_usage_min_gb=0.08,
                    ram_usage_max_gb=0.20,
                    ram_usage_last_gb=0.12,
                )],
                execution_rate_hz=30.0,
            ),
            Service(
                "imu",
                [ContainerSpec(
                    "imu",
                    cpu=0.25,
                    ram_gb=0.08,
                    exec_time=0.001,
                    cpu_usage_min=0.05,
                    cpu_usage_max=0.35,
                    cpu_usage_last=0.20,
                    ram_usage_min_gb=0.05,
                    ram_usage_max_gb=0.12,
                    ram_usage_last_gb=0.08,
                )],
                execution_rate_hz=100.0,
            ),
            Service(
                "encoder",
                [ContainerSpec(
                    "encoder",
                    cpu=0.30,
                    ram_gb=0.08,
                    exec_time=0.002,
                    cpu_usage_min=0.08,
                    cpu_usage_max=0.40,
                    cpu_usage_last=0.22,
                    ram_usage_min_gb=0.05,
                    ram_usage_max_gb=0.12,
                    ram_usage_last_gb=0.08,
                )],
                execution_rate_hz=50.0,
            ),
            # SLAM toolbox
            Service(
                "slam_toolbox",
                [ContainerSpec(
                    "slam",
                    cpu=0.70,
                    ram_gb=0.22,
                    exec_time=0.01,
                    cpu_usage_min=0.30,
                    cpu_usage_max=0.90,
                    cpu_usage_last=0.70,
                    ram_usage_min_gb=0.15,
                    ram_usage_max_gb=0.30,
                    ram_usage_last_gb=0.22,
                )],
                dependencies=[
                    DependencySpec("camera", size_mb=2.0),
                    DependencySpec("imu", size_mb=0.1),
                    DependencySpec("encoder", size_mb=0.1)
                ],
                execution_rate_hz=10.0,
            ),
            # robot_localization
            Service(
                "robot_localization",
                [ContainerSpec(
                    "ekf",
                    cpu=0.60,
                    ram_gb=0.18,
                    exec_time=0.012,
                    cpu_usage_min=0.25,
                    cpu_usage_max=0.80,
                    cpu_usage_last=0.60,
                    ram_usage_min_gb=0.10,
                    ram_usage_max_gb=0.22,
                    ram_usage_last_gb=0.18,
                )],
                dependencies=[DependencySpec("slam_toolbox", size_mb=0.5)],
                execution_rate_hz=10.0,
            ),
            # Pure Pursuit
            Service(
                "pure_pursuit",
                [ContainerSpec(
                    "pure_pursuit",
                    cpu=0.40,
                    ram_gb=0.12,
                    exec_time=0.005,
                    cpu_usage_min=0.15,
                    cpu_usage_max=0.60,
                    cpu_usage_last=0.40,
                    ram_usage_min_gb=0.08,
                    ram_usage_max_gb=0.16,
                    ram_usage_last_gb=0.12,
                )],
                dependencies=[DependencySpec("robot_localization", size_mb=0.2)],
                execution_rate_hz=10.0,
            ),
            # Emergency Stop (requires GPU)
            Service(
                "emergency_stop",
                [ContainerSpec(
                    "e_stop",
                    cpu=1.0,
                    ram_gb=0.6,
                    gpu_gb=6,
                    exec_time=0.018,
                    cpu_usage_min=0.40,
                    cpu_usage_max=1.20,
                    cpu_usage_last=0.90,
                    ram_usage_min_gb=0.30,
                    ram_usage_max_gb=1.00,
                    ram_usage_last_gb=0.60,
                )],
                dependencies=[DependencySpec("camera", size_mb=2.0, constraint_level="soft")],
                execution_rate_hz=10.0,
            ),
        ]
        # Lock sensors to frank-1
        for s in services[:3]:
            s.locked_node_id = "frank-1"
        
        # Lock slam_toolbox to edge layer (frank-1 is edge) since all its dependencies are on frank-1
        services[3].locked_layer_id = "edge"

        return services
    
    @staticmethod
    def choose_nodes(nodes_list, frac):
        k = max(1, int(round(frac * len(nodes_list))))
        return set(n.node_id for n in random.sample(nodes_list, k))
    
    @staticmethod
    def calculate_metrics(assignments: dict, services, nodes, latency_matrix, time_step_minutes: int = 5, scenario=None, step: int = 0):
        """Calculate metrics for the current time step.
        
        Args:
            time_step_minutes: Duration of this time step in minutes (for battery calculations)
            scenario: Optional DynamicScenario object for baseline loads
            step: Current time step number
        """
        node_lookup = {node.node_id: node for node in nodes}
        service_lookup = {svc.service_id: svc for svc in services}
        node_index = {node.node_id: i for i, node in enumerate(nodes)}
        service_index = {svc.service_id: i for i, svc in enumerate(services)}
        service_map = {getattr(svc, "service_id", str(i)): svc for i, svc in enumerate(services)}

        def _latency_cost_full(svc, target_node_id: str) -> float:
            """Calculate latency including the entire dependency tree (matches orchestrator logic)."""
            target_idx = node_index.get(target_node_id)
            if target_idx is None:
                return 0.0

            visiting: set[str] = set()
            root_id = str(getattr(svc, "service_id", ""))

            def dfs(cur_svc, cur_idx: int) -> float:
                total = 0.0
                for dep in getattr(cur_svc, "dependencies", []) or []:
                    if getattr(dep, "blocking", True) is False:
                        continue
                    dep_id = getattr(dep, "service_id", None)
                    if not isinstance(dep_id, str):
                        continue
                    upstream_nid = assignments.get(dep_id)
                    if upstream_nid is None:
                        continue
                    upstream_idx = node_index.get(upstream_nid)
                    if upstream_idx is None:
                        continue

                    try:
                        net_lat = float(latency_matrix[upstream_idx][cur_idx])
                        transfer_time = (dep.size_mb / 200.0) if getattr(dep, "size_mb", 0) > 0 else 0.0
                    except Exception:
                        net_lat, transfer_time = 0.0, 0.0

                    edge_cost = net_lat + transfer_time
                    if dep_id in visiting:
                        total += edge_cost
                        continue

                    visiting.add(dep_id)
                    dep_svc = service_map.get(dep_id)
                    if dep_svc:
                        total += edge_cost + dfs(dep_svc, upstream_idx)
                    else:
                        total += edge_cost
                    visiting.remove(dep_id)
                return total

            total = dfs(svc, target_idx)

            if getattr(svc, "control_loop", False):
                for down_svc in services:
                    for dep in getattr(down_svc, "dependencies", []) or []:
                        if getattr(dep, "feedback", False) is not True:
                            continue
                        if getattr(dep, "service_id", None) != root_id:
                            continue
                        down_id = str(getattr(down_svc, "service_id", ""))
                        down_nid = assignments.get(down_id)
                        if down_nid is None:
                            continue
                        down_idx = node_index.get(down_nid)
                        if down_idx is None:
                            continue
                        try:
                            net_lat = float(latency_matrix[target_idx][down_idx])
                            transfer_time = (dep.size_mb / 200.0) if getattr(dep, "size_mb", 0) > 0 else 0.0
                        except Exception:
                            net_lat, transfer_time = 0.0, 0.0
                        total += net_lat + transfer_time

            return total
        
        # We'll calculate critical path latency after processing all services
        service_latencies = {}  # Maps service_id -> (exec_time, network_latency)
        total_energy = 0.0
        
        # Initialize node usage with baseline loads if scenario is provided
        node_usage = {}
        for node in nodes:
            if scenario is not None:
                baseline = scenario.get_baseline_load(node.node_id, step)
                node_usage[node.node_id] = {
                    "cpu": baseline['cpu'],
                    "ram": baseline['ram'],
                    "gpu": 0.0,
                    "baseline_cpu": baseline['cpu'],
                    "baseline_ram": baseline['ram']
                }
            else:
                node_usage[node.node_id] = {"cpu": 0.0, "ram": 0.0, "gpu": 0.0}
        
        violations = []
        notified_unreliable = set()  # Track services with unreliable execution times
        
        battery_usage = {}
        for node in nodes:
            if getattr(node, 'battery_powered', False):
                voltage = getattr(node, 'battery_voltage', 3.7)
                capacity_ah = getattr(node, 'battery_capacity_ah', 0.0)
                battery_usage[node.node_id] = {
                    "energy_j": 0.0,
                    "idle_energy_j": 0.0,
                    "total_energy_j": 0.0,
                    "energy_ah": 0.0,
                    "voltage": voltage,
                    "capacity_ah": capacity_ah
                }
        
        if assignments is None:
            return None, None, None, None
        
        # Calculate per-service metrics
        for svc in services:
            node_id = assignments.get(svc.service_id)
            if not node_id or node_id not in node_lookup:
                violations.append(f"{svc.service_id}: Not assigned to any valid node")
                continue

            node = node_lookup[node_id]
            node_idx = node_index[node_id]
            svc_idx = service_index[svc.service_id]

            # Check locked node constraint
            locked_nid = getattr(svc, 'locked_node_id', None)
            if locked_nid is not None and str(node_id) != str(locked_nid):
                violations.append(f"{svc.service_id}: Locked to {locked_nid} but assigned to {node_id}")

            # Resource requirements - use new execution time extraction
            if hasattr(svc, 'get_execution_time'):
                exec_time, is_reliable = svc.get_execution_time()
                if exec_time is not None and not is_reliable:
                    exec_time *= 1.1  # Match orchestrator buffer for unreliable timing
                    if svc.service_id not in notified_unreliable:
                        print(f"  ⚠️  {svc.service_id}: Using estimated execution time ({exec_time:.2f}s)")
                        notified_unreliable.add(svc.service_id)
            else:
                # Fallback for backward compatibility - filter out None values
                exec_times = [c.exec_time for c in svc.containers if c.exec_time is not None]
                exec_time = max(exec_times) if exec_times else None
                is_reliable = False

            execution_rate_hz = getattr(svc, 'execution_rate_hz', 1.0)
            instance_usage = getattr(svc, "instance_usage_by_node", {}).get(node_id)
            if instance_usage:
                if instance_usage.get("exec_time_max") is not None:
                    exec_time = float(instance_usage["exec_time_max"])
                if instance_usage.get("execution_rate_hz_max") is not None:
                    execution_rate_hz = float(instance_usage["execution_rate_hz_max"])
            
            cpu_needed = sum(c.cpu for c in svc.containers)
            ram_needed = sum(c.ram_gb for c in svc.containers)
            gpu_needed = sum(c.gpu_gb for c in svc.containers)

            # Period-based CPU allocation (only for time-constrained services)
            period = 1.0 / execution_rate_hz if execution_rate_hz > 0 else float('inf')
            if exec_time is not None and period > 0 and period < float('inf'):
                cpu_utilization_factor = min(1.0, exec_time / period)
                effective_cpu = cpu_needed * cpu_utilization_factor
            else:
                effective_cpu = cpu_needed

            # Check GPU sharing
            gpu_shared = getattr(node, 'gpu_shared', False)
            if gpu_shared and gpu_needed > 0:
                ram_needed += gpu_needed

            # Track resource usage
            node_usage[node_id]["cpu"] += effective_cpu
            node_usage[node_id]["ram"] += ram_needed
            node_usage[node_id]["gpu"] += gpu_needed

            # Calculate network latency for this service
            # Only count network latency from actual dependencies, not artificial sequential ordering
            service_network_latency = 0.0
            throughput_mbps = 200.0
            
            # Network latency for dependencies
            for dep in getattr(svc, 'dependencies', []) or []:
                dep_node_id = assignments.get(dep.service_id)
                if dep_node_id and dep_node_id in node_index:
                    dep_node_idx = node_index[dep_node_id]
                    net_lat = float(latency_matrix[dep_node_idx][node_idx])
                    transfer_time = (dep.size_mb / max(1.0, throughput_mbps)) if dep.size_mb > 0 else 0.0
                    service_network_latency += net_lat + transfer_time

            # Check period constraint (strict, no tolerance)
            # Period constraint includes execution time AND network latency
            # Skip for non-time-constrained services (exec_time is None)
            if exec_time is not None:
                dep_latency_full = _latency_cost_full(svc, node_id)
                total_service_time = exec_time + dep_latency_full
                if period < float('inf') and total_service_time > period:
                    violation_msg = f"{svc.service_id}: total_time ({total_service_time:.2f}s = exec:{exec_time:.2f}s + net:{dep_latency_full:.2f}s) > period ({period:.2f}s)"
                    violations.append(violation_msg)
                    
                    # Record violation in service for potential reassignment
                    if hasattr(svc, 'record_violation'):
                        svc.record_violation('period_constraint', {
                            'timestamp': time_step_minutes * 60,  # Convert to seconds
                            'node_id': node_id,
                            'exec_time': exec_time,
                            'network_latency': dep_latency_full,
                            'total_time': total_service_time,
                            'period': period,
                            'is_reliable': is_reliable
                        })

            # Energy calculation (Jetson + Motors if available)
            jetson_base = getattr(node, 'power_jetson_base', getattr(node, 'power_base', 5.0))
            jetson_active = getattr(node, 'power_jetson_active', getattr(node, 'power_active', 30.0))
            motor_max = getattr(node, 'power_motor_max', 0.0)
            node_cpu_cap = node.cpu
            node_gpu_cap = node.gpu.capacity if node.gpu else 0.0

            cpu_ratio = cpu_needed / max(node_cpu_cap, 1.0)
            gpu_ratio = gpu_needed / max(node_gpu_cap, 1.0) if node_gpu_cap > 0 else 0.0
            util_ratio = min(1.0, cpu_ratio + gpu_ratio)
            jetson_power = jetson_base + (jetson_active - jetson_base) * util_ratio

            avg_power = jetson_power

            # Energy over time step duration
            # For non-time-constrained services (exec_time is None), use a conservative default
            effective_exec_time = exec_time if exec_time is not None else 0.01  # 10ms default
            execution_rate_hz = getattr(svc, 'execution_rate_hz', 1.0)
            executions_per_step = execution_rate_hz * (time_step_minutes * 60)  # Convert to seconds
            service_energy = avg_power * effective_exec_time * executions_per_step  # Joules
            
            if getattr(node, 'battery_powered', False) and node_id in battery_usage:
                total_energy += service_energy
                battery_usage[node_id]["energy_j"] += service_energy
                voltage = battery_usage[node_id]["voltage"]
                battery_usage[node_id]["energy_ah"] = battery_usage[node_id]["energy_j"] / (voltage * 3600.0)
                # Debug: print per-step energy calculation for frank-1 (disabled)
                # if node_id == "frank-1":
                #     print(f"[DEBUG][frank-1] Service: {svc.service_id}, Execs/step: {executions_per_step:.2f}, Exec time: {exec_time:.2f}s, CPU: {cpu_needed}, Power: {avg_power:.2f}W, Service energy: {service_energy:.2f}J, Cumulative: {battery_usage[node_id]['energy_j']:.2f}J, Ah: {battery_usage[node_id]['energy_ah']:.6f}")

            # Store service latency information for critical path calculation
            # Use 0 for exec_time if not available (non-time-constrained services)
            effective_exec_for_latency = exec_time if exec_time is not None else 0.0
            service_latencies[svc.service_id] = {
                'exec_time': effective_exec_for_latency,
                'network_latency': service_network_latency,
                'total_time': effective_exec_for_latency + service_network_latency
            }
        
        # Calculate critical path latency through the dependency graph
        # Use dynamic programming: latency to complete a service = max(latency of dependencies) + service time
        critical_path_latencies = {}
        visiting = set()  # Track services currently being visited to detect cycles
        
        def get_critical_path_latency(svc_id):
            """Calculate the critical path latency to complete a service (includes all upstream dependencies)."""
            if svc_id in critical_path_latencies:
                return critical_path_latencies[svc_id]
            
            if svc_id in visiting:
                # Circular dependency detected - break the cycle by returning 0
                return 0.0
            
            if svc_id not in service_latencies:
                # Service not assigned or error
                critical_path_latencies[svc_id] = 0.0
                return 0.0
            
            visiting.add(svc_id)
            
            svc = service_lookup.get(svc_id)
            if not svc:
                critical_path_latencies[svc_id] = 0.0
                visiting.discard(svc_id)
                return 0.0
            
            svc_info = service_latencies[svc_id]
            max_dep_latency = 0.0
            
            # Find maximum latency among dependencies
            for dep in getattr(svc, 'dependencies', []) or []:
                dep_id = getattr(dep, 'service_id', None)
                if dep_id:
                    dep_latency = get_critical_path_latency(dep_id)
                    max_dep_latency = max(max_dep_latency, dep_latency)
            
            # Critical path to this service = max dependency latency + this service's total time
            result = max_dep_latency + svc_info['total_time']
            critical_path_latencies[svc_id] = result
            visiting.discard(svc_id)
            return result
        
        # Calculate critical path for all services
        for svc_id in service_latencies.keys():
            get_critical_path_latency(svc_id)
        
        # Total latency is the maximum critical path (end-to-end latency of the longest path)
        total_latency = max(critical_path_latencies.values()) if critical_path_latencies else 0.0
        
        # Check capacity violations
        for node in nodes:
            node_id = node.node_id
            usage = node_usage[node_id]
            
            cpu_cap = node.cpu
            ram_cap = node.ram
            gpu_cap = node.gpu.capacity if node.gpu else 0.0
            
            if usage["cpu"] > cpu_cap + 0.01:
                violations.append(f"{node_id}: CPU overload ({usage['cpu']:.2f}/{cpu_cap:.2f} cores)")
                # MUCH heavier penalty for resource violations
                total_energy += 5000.0
            if usage["ram"] > ram_cap + 0.01:
                violations.append(f"{node_id}: RAM overload ({usage['ram']:.2f}/{ram_cap:.2f} GB)")
                # MUCH heavier penalty for resource violations
                total_energy += 5000.0
            if usage["gpu"] > gpu_cap + 0.01:
                violations.append(f"{node_id}: GPU overload ({usage['gpu']:.2f}/{gpu_cap:.2f} GB)")
                # MUCH heavier penalty for resource violations
                total_energy += 5000.0
        
        # Add idle power for battery nodes over the time step
        # Also account for baseline load power consumption
        time_step_seconds = time_step_minutes * 60
        for node_id, usage_data in battery_usage.items():
            node = node_lookup[node_id]
            jetson_base = getattr(node, 'power_jetson_base', getattr(node, 'power_base', 3.0))
            jetson_active = getattr(node, 'power_jetson_active', getattr(node, 'power_active', 30.0))
            motor_max = getattr(node, 'power_motor_max', 0.0)
            
            # Idle energy (base power)
            idle_energy = jetson_base * time_step_seconds
            
            # Baseline load energy (if baseline loads are present)
            baseline_energy = 0.0
            if node_id in node_usage:
                usage = node_usage[node_id]
                if 'baseline_cpu' in usage and 'baseline_ram' in usage:
                    # Calculate energy from baseline load
                    node_cpu_cap = node.cpu
                    baseline_cpu_ratio = usage['baseline_cpu'] / max(node_cpu_cap, 1.0)
                    baseline_power = jetson_base + (jetson_active - jetson_base) * min(1.0, baseline_cpu_ratio)
                    baseline_energy = baseline_power * time_step_seconds

            motor_util = getattr(node, 'motor_utilization', None)
            if motor_util is None:
                motor_util = random.uniform(0.0, 1.0) if motor_max > 0 else 0.0
            motor_energy = float(motor_max) * float(motor_util) * time_step_seconds
            
            usage_data["idle_energy_j"] = idle_energy
            usage_data["baseline_energy_j"] = baseline_energy
            usage_data["motor_energy_j"] = motor_energy
            usage_data["total_energy_j"] = usage_data["energy_j"] + idle_energy + baseline_energy + motor_energy
            voltage = usage_data["voltage"]
            usage_data["energy_ah"] = usage_data["total_energy_j"] / (voltage * 3600.0)
        
        # Add capacity information to node_usage for utilization percentage calculation
        node_usage_with_capacity = {}
        for node in nodes:
            node_id = node.node_id
            usage = node_usage[node_id]
            cpu_cap = node.cpu
            ram_cap = node.ram
            gpu_cap = node.gpu.capacity if node.gpu else 0.0
            
            node_usage_with_capacity[node_id] = {
                "cpu_used": usage["cpu"],
                "cpu_capacity": cpu_cap,
                "cpu_util_pct": (usage["cpu"] / cpu_cap * 100) if cpu_cap > 0 else 0.0,
                "ram_used": usage["ram"],
                "ram_capacity": ram_cap,
                "ram_util_pct": (usage["ram"] / ram_cap * 100) if ram_cap > 0 else 0.0,
                "gpu_used": usage["gpu"],
                "gpu_capacity": gpu_cap,
                "gpu_util_pct": (usage["gpu"] / gpu_cap * 100) if gpu_cap > 0 else 0.0,
            }
        
        return total_latency, total_energy, violations, battery_usage, node_usage_with_capacity