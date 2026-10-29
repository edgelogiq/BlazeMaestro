"""
Dynamic comparison for a ROS-based scenario:
- frank-1 runs camera (Intel RealSense, 30Hz), IMU (100Hz), encoder (50Hz)
- slam_toolbox (anywhere), depends on camera/IMU/encoder
- robot_localization (anywhere), depends on slam_toolbox
- pure_pursuit (anywhere), depends on robot_localization
- emergency_stop (anywhere), depends on camera (requires GPU), sends stop signal
"""

# Install logger workaround BEFORE any BlazeAIoT imports to avoid circular imports
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

import argparse
import os
import copy
import simpy
import numpy as np
import sys
import random
from typing import List
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.platform_topology import PlatformTopology
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.service_simulator import Service
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_loader import ServiceDefinitionLoader
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.dqn_orchestrator import DQNOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.nsga2_orchestrator import NSGA2Orchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.unsupervised_orchestrator import GreedyWeightedOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.autoencoder_orchestrator import AutoEncoderOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.simulator_facade import SimulatorMetricsFacade
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.network_util import NetworkUtil
from train_utils import TrainUtils, DynamicScenario
from collections import defaultdict
from tabulate import tabulate
import json


class ROSDynamicScenario(DynamicScenario):
    """Extends DynamicScenario with ROS-specific scenario changes over time."""
    
    def __init__(self, duration_minutes: int = 30, time_step_minutes: int = 5):
        super().__init__()
        self.duration_minutes = duration_minutes
        self.time_step_minutes = time_step_minutes
        self.num_steps = duration_minutes // time_step_minutes
        self.current_step = 0
        
    def get_time_label(self) -> str:
        """Get human-readable time label for current step."""
        return f"T={self.current_step * self.time_step_minutes}min"
    
    def apply_scenario_changes(self, services: List[Service], nodes: List, step: int):
        """Apply time-dependent changes to services and nodes.
        
        Scenarios:
        - T=0-10min: Normal operation (baseline)
        - T=10-15min: Peak traffic (1.5x execution rates, higher CPU usage)
        - T=15-20min: One edge node becomes degraded (50% capacity)
        - T=20-25min: Recovery + network congestion
        - T=25-30min: Low traffic period (0.5x execution rates)
        """
        time_min = step * self.time_step_minutes
        
        # Reset to baseline first
        for svc in services:
            svc.execution_rate_hz = getattr(svc, '_baseline_rate', svc.execution_rate_hz)
            # Reset execution times
            for container in svc.containers:
                baseline_time = getattr(container, '_baseline_exec_time', None)
                if baseline_time is not None:
                    container.exec_time = baseline_time
                # Ensure minimal nonzero CPU/RAM
                if getattr(container, 'cpu', 0) < 0.05:
                    container.cpu = 0.05
                if getattr(container, 'ram_gb', 0) < 0.01:
                    container.ram_gb = 0.01
        
        # Apply scenario changes
        if time_min < 10:
            # T=0-10min: Normal operation (already at baseline, but add small variation)
            scenario_desc = "Normal operation"
            for svc in services:
                svc.execution_rate_hz *= random.uniform(0.95, 1.05)
        elif time_min < 15:
            # T=10-15min: Peak traffic (1.5x rates, with variation)
            scenario_desc = "Peak traffic (1.5x execution rates)"
            for svc in services:
                if not hasattr(svc, '_baseline_rate'):
                    setattr(svc, '_baseline_rate', svc.execution_rate_hz)
                baseline = getattr(svc, '_baseline_rate', svc.execution_rate_hz)
                svc.execution_rate_hz = baseline * 1.5 * random.uniform(0.95, 1.05)
        elif time_min < 20:
            # T=15-20min: Increased service execution times (simulating node degradation)
            scenario_desc = "Increased workload (1.2x execution times)"
            for svc in services:
                for container in svc.containers:
                    if not hasattr(container, '_baseline_exec_time'):
                        setattr(container, '_baseline_exec_time', container.exec_time)
                    baseline_time = getattr(container, '_baseline_exec_time', container.exec_time)
                    container.exec_time = baseline_time * 1.2 * random.uniform(0.95, 1.05)
        elif time_min < 25:
            # T=20-25min: Network congestion (simulated by higher jitter in separate step)
            scenario_desc = "Network congestion period"
            for svc in services:
                svc.execution_rate_hz *= random.uniform(0.95, 1.05)
        else:
            # T=25-30min: Low traffic (0.5x rates, with variation)
            scenario_desc = "Low traffic (0.5x execution rates)"
            for svc in services:
                if not hasattr(svc, '_baseline_rate'):
                    setattr(svc, '_baseline_rate', svc.execution_rate_hz)
                baseline = getattr(svc, '_baseline_rate', svc.execution_rate_hz)
                svc.execution_rate_hz = baseline * 0.5 * random.uniform(0.95, 1.05)
        return scenario_desc
    

def main():
    parser = argparse.ArgumentParser(description="Compare orchestrators for ROS scenario using platform.xml.")
    parser.add_argument("--platform", type=str, default="src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/Simulator/platform.xml")
    parser.add_argument("--checkpoint_dir", type=str, help="Path to checkpoints")
    parser.add_argument("--services-xml", type=str, default=None, help="Path to services.xml for service definitions")
    parser.add_argument("--output-dir", type=str, default="out/dynamic_ros_comparison", help="Output directory for results")
    parser.add_argument("--enable-jitter", action="store_true", help="Enable network jitter")
    parser.add_argument("--jitter-percent", type=float, default=10.0, help="Base jitter percentage")
    parser.add_argument("--enable-packet-loss", action="store_true", help="Enable packet loss")
    parser.add_argument("--packet-loss-rate", type=float, default=0.01, help="Packet loss rate")
    parser.add_argument("--duration", type=int, default=30, help="Simulation duration in minutes (default: 30)")
    parser.add_argument("--time-step", type=int, default=5, help="Time step in minutes (default: 5)")
    parser.add_argument("--hysteresis-threshold", type=float, default=0.05, help="Hysteresis threshold (0.05 = 5%% improvement required)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--enable-wifi", action="store_true", help="Enable WiFi mobility simulation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    env = simpy.Environment()
    clusters, all_nodes = PlatformTopology(env, args.platform).build()
    if args.services_xml and os.path.exists(args.services_xml):
        service_loader = ServiceDefinitionLoader(args.services_xml, simpy_env=None)
        services = service_loader.load_services()
    else:
        services = TrainUtils.build_services()
    TrainUtils.validate_locked_nodes(services, all_nodes)
    
    # Initialize baseline rates and execution times for all services (before scenario modifications)
    for svc in services:
        if not hasattr(svc, '_baseline_rate'):
            setattr(svc, '_baseline_rate', svc.execution_rate_hz)
        for container in svc.containers:
            if not hasattr(container, '_baseline_exec_time'):
                setattr(container, '_baseline_exec_time', container.exec_time)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    latency = NetworkUtil.build_latency_matrix_with_jitter(
        args.platform,
        all_nodes,
        enable_jitter=args.enable_jitter,
        jitter_percent=args.jitter_percent,
        enable_packet_loss=args.enable_packet_loss,
        packet_loss_rate=args.packet_loss_rate,
        seed=args.seed)
    
    if args.enable_jitter:
        print(f"🌐 Network jitter enabled: ±{args.jitter_percent}% base")
    else:
        print("🌐 Network jitter disabled")

    wifi_controller = None
    if args.enable_wifi:
        wifi_controller = WiFiMobilityController()
        # Default WiFi setup (matches simulator config)
        wifi_controller.register_access_point("edge-gpu1", 0.0, 0.0, 20.0)
        wifi_controller.register_access_point("edge-gpu2", 30.0, 0.0, 20.0)
        wifi_controller.register_robot("frank-2", 0.0, 0.0, 0.2, 0.0)
        wifi_controller.register_wifi_link("frank-2-wifi6", "frank-2")

    metrics_facade = SimulatorMetricsFacade(all_nodes, latency_matrix=latency, wifi_controller=wifi_controller)
    orchestrators = {}
    orchestrators["NSGA-II"] = NSGA2Orchestrator(services, all_nodes, latency, generations=30, metrics=metrics_facade)
    
    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        dqn_ckpt_path = os.path.join(args.checkpoint_dir, "DQN_checkpoint.pt")
        dqn_orch = DQNOrchestrator(services, all_nodes, latency, episodes=0)
        dqn_orch.load_checkpoint(dqn_ckpt_path)
        orchestrators["DQN"] = dqn_orch

        greedy_orch = GreedyWeightedOrchestrator(services, all_nodes, latency, metrics=metrics_facade)
        orchestrators["GreedyWeighted"] = greedy_orch

        autoencoder_ckpt_path = os.path.join(args.checkpoint_dir, "AutoEncoder_checkpoint.pt")
        autoencoder_orch = AutoEncoderOrchestrator(services, all_nodes, latency, metrics=metrics_facade)
        autoencoder_orch.load_checkpoint(autoencoder_ckpt_path)
        orchestrators["Autoencoder"] = autoencoder_orch

    print(f"\n{'='*80}")
    print(f"ROS Dynamic Scenario Comparison")
    print(f"Duration: {args.duration} minutes, Time step: {args.time_step} minutes")
    print(f"Hysteresis threshold: {args.hysteresis_threshold*100:.1f}%")
    print(f"Platform: {args.platform}")
    print(f"Orchestrators: {', '.join(orchestrators.keys())}")
    print(f"{'='*80}\n")

    # Dynamic scenario loop
    
    scenario = ROSDynamicScenario(duration_minutes=args.duration, time_step_minutes=args.time_step)
    
    # Initialize baseline loads for all nodes
    scenario.initialize_baseline_loads(all_nodes)
    print(f"🔧 Initialized baseline loads for {len(all_nodes)} nodes")
    for node in all_nodes:
        baseline = scenario.get_baseline_load(node.node_id, 0)
        print(f"   {node.node_id}: baseline ~{baseline['cpu']:.2f} CPU, ~{baseline['ram']:.2f} GB RAM")
    print()
    
    time_series_results = {name: [] for name in orchestrators.keys()}
    current_assignments = {name: None for name in orchestrators.keys()}
    reassignment_counts = {name: 0 for name in orchestrators.keys()}
    cumulative_battery_usage = {name: defaultdict(float) for name in orchestrators.keys()}

    sim_time_s = 0.0
    step_seconds = float(args.time_step) * 60.0
    for step in range(scenario.num_steps):
        if wifi_controller is not None:
            wifi_controller.update(sim_time_s)
            latency = metrics_facade.get_latency_matrix(all_nodes)
            for orch in orchestrators.values():
                orch.latency_matrix = latency
            sim_time_s += step_seconds
        time_label = scenario.get_time_label()
        scenario_desc = scenario.apply_scenario_changes(services, all_nodes, step)
        print(f"\n{'─'*80}")
        print(f"⏰ Time Step {step+1}/{scenario.num_steps}: {time_label}")
        print(f"📋 Scenario: {scenario_desc}")
        print(f"{'─'*80}\n")
        for name, orch in orchestrators.items():
            print(f"Running {name}...")
            current = current_assignments[name]
            inference_failed = False
            try:
                new_assignments = orch.optimize(
                    services,
                    current_assignment=current,
                    hysteresis_threshold=args.hysteresis_threshold
                )
            except RuntimeError as exc:
                print(f"  [ERROR] {name} failed: {exc}")
                inference_failed = True
                new_assignments = current if current is not None else {}
            if new_assignments is None:
                print(f"  [ERROR] {name} returned None")
                inference_failed = True
                new_assignments = current if current is not None else {}
            if inference_failed and current is not None:
                print(f"  [ERROR] {name} no assignment; keeping current")
            changed = False
            if current is not None:
                changed = any(new_assignments.get(sid) != current.get(sid) for sid in new_assignments)
                if changed:
                    reassignment_counts[name] += 1
                    print(f"  ↻ Reassignment #{reassignment_counts[name]}")
                else:
                    print(f"  ✓ Kept current assignment")
            current_assignments[name] = copy.deepcopy(new_assignments)
            total_latency, total_energy, violations, battery_usage, node_usage = TrainUtils.calculate_metrics(
                new_assignments, services, all_nodes, latency,
                time_step_minutes=args.time_step,
                scenario=scenario,
                step=step
            )
            if total_latency is None:
                continue
            strict_violations = TrainUtils.strict_validate_placement(new_assignments, services, all_nodes, latency)
            if strict_violations:
                violations = strict_violations
            if inference_failed:
                if not violations:
                    violations = []
                violations.append("Inference failure: no assignment returned")
            for node_id, usage in battery_usage.items():
                cumulative_battery_usage[name][node_id] += usage["energy_ah"]
            result = {
                "time": time_label,
                "step": step,
                "latency": total_latency,
                "energy": total_energy,
                "violations": len(violations),
                "assignments": copy.deepcopy(new_assignments),
                "battery_usage": copy.deepcopy(battery_usage),
                "node_usage": copy.deepcopy(node_usage),
                "reassigned": changed if current is not None else False
            }
            time_series_results[name].append(result)
            violation_status = "✓" if not violations else f"✗ {len(violations)}"
            print(f"  Latency: {total_latency:.4f}s | Energy: {total_energy:.2f}J | Violations: {violation_status}")
            
            if violations:
                for v in violations[:3]:
                    print(f"    - {v}")
                if len(violations) > 3:
                    print(f"    ... and {len(violations)-3} more")

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - Dynamic Scenario Results")
    print(f"{'='*80}\n")
    
    summary_table = []
    for name in orchestrators.keys():
        results = time_series_results[name]
        if not results:
            continue
        avg_latency = np.mean([r["latency"] for r in results])
        total_energy = sum([r["energy"] for r in results])
        total_violations = sum([r["violations"] for r in results])
        reassignments = reassignment_counts[name]
        battery_nodes = [n for n in all_nodes if getattr(n, 'battery_powered', False)]
        battery_status = []
        for node in battery_nodes:
            node_id = getattr(node, 'node_id', '')
            used_ah = cumulative_battery_usage[name].get(node_id, 0.0)
            capacity_ah = getattr(node, 'battery_capacity_ah', 0.0)
            remaining_pct = ((capacity_ah - used_ah) / capacity_ah * 100) if capacity_ah > 0 else 0
            battery_status.append(f"{node_id}: {remaining_pct:.1f}%")
        battery_str = "\n".join(battery_status) if battery_status else "-"
        summary_table.append([
            name,
            f"{avg_latency:.4f}s",
            f"{total_energy:.2f}J",
            f"{total_violations}",
            f"{reassignments}",
            battery_str
        ])
    summary_str = tabulate(
        summary_table,
        headers=["Orchestrator", "Avg Latency", "Total Energy", "Total Violations", "Reassignments", "Battery Remaining"],
        tablefmt="grid"
    )
    print(summary_str)
    output_file = os.path.join(args.output_dir, "dynamic_results.json")
    
    output_data = {
        "config": {
            "duration_minutes": args.duration,
            "time_step_minutes": args.time_step,
            "hysteresis_threshold": args.hysteresis_threshold,
            "jitter_enabled": args.enable_jitter,
            "seed": args.seed
        },
        "results": time_series_results,
        "reassignment_counts": reassignment_counts,
        "cumulative_battery_usage": {k: dict(v) for k, v in cumulative_battery_usage.items()}
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\n📊 Detailed results saved to: {output_file}")
    summary_file = os.path.join(args.output_dir, "summary_table.txt")
    with open(summary_file, 'w') as f:
        f.write(summary_str + "\n")
    print(f"📋 Summary table saved to: {summary_file}")

if __name__ == "__main__":
    main()
