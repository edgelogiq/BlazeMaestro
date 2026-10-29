#!/usr/bin/env python3
"""
Test script to validate reassignment stability analysis with scenarios
that intentionally trigger reassignments to verify the analysis works correctly.
"""

# Install logger workaround BEFORE any BlazeAIoT imports to avoid circular imports
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

import argparse
import os
import simpy
import random
import sys
import json
from tabulate import tabulate
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.platform_topology import PlatformTopology
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_loader import ServiceDefinitionLoader
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.dqn_orchestrator import DQNOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.nsga2_orchestrator import NSGA2Orchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.unsupervised_orchestrator import GreedyWeightedOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.autoencoder_orchestrator import AutoEncoderOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.simulator_facade import SimulatorMetricsFacade
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.network_util import NetworkUtil
from train_utils import TrainUtils, DynamicScenario
import copy
import numpy as np
from collections import defaultdict


def apply_scenario(services, scenario_name):
    """
    Apply different scenarios to trigger reassignments.
    
    Scenarios:
    - normal: Standard load
    - extreme_rate: Very high execution rates (WILL cause violations on weak nodes)
    - heavy_exec: Long execution times (WILL cause violations)
    - light_load: Low rates (opportunity for energy optimization)
    """
    
    # Always ensure a minimal nonzero load for all services
    for svc in services:
        for container in svc.containers:
            # Set a minimal nonzero CPU and RAM if not already set
            if getattr(container, 'cpu', 0) < 0.05:
                container.cpu = 0.05
            if getattr(container, 'ram_gb', 0) < 0.01:
                container.ram_gb = 0.01

    if scenario_name == "normal":
        # Standard baseline with minimal but nonzero load and slight variation
        services[1].execution_rate_hz = 12.0  # 83ms period
        services[2].execution_rate_hz = 8.0   # 125ms period
        services[3].execution_rate_hz = 7.0   # 143ms period
        # Add a small random variation to simulate real-world drift
        for svc in services[1:4]:
            svc.execution_rate_hz *= random.uniform(0.95, 1.05)

    elif scenario_name == "extreme_rate":
        # High rates, but not zero elsewhere
        services[1].execution_rate_hz = 30.0   # 33ms period
        services[2].execution_rate_hz = 25.0   # 40ms period
        services[3].execution_rate_hz = 20.0   # 50ms period
        for svc in services[1:4]:
            svc.execution_rate_hz *= random.uniform(0.95, 1.05)

    elif scenario_name == "heavy_exec":
        # Long execution times, but not zero elsewhere
        services[1].execution_rate_hz = 10.0   # 100ms period
        services[2].execution_rate_hz = 8.0    # 125ms period
        services[1].containers[0].exec_time = 0.08  # 80ms
        services[2].containers[0].exec_time = 0.10  # 100ms
        for svc in services[1:4]:
            svc.execution_rate_hz *= random.uniform(0.95, 1.05)

    elif scenario_name == "light_load":
        # Low load, but not zero
        services[1].execution_rate_hz = 3.0    # 333ms period
        services[2].execution_rate_hz = 2.0    # 500ms period
        services[3].execution_rate_hz = 2.0    # 500ms period
        for svc in services[1:4]:
            svc.execution_rate_hz *= random.uniform(0.95, 1.05)
        
    return scenario_name


def main():
    parser = argparse.ArgumentParser(description="Test reassignment stability with various scenarios.")
    parser.add_argument("--platform", type=str, default="src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/Simulator/platform.xml")
    parser.add_argument("--checkpoint-dir", type=str, help="Path to DQN checkpoint")
    parser.add_argument("--services-xml", type=str, default=None, help="Path to services.xml for service definitions")
    parser.add_argument("--output-dir", type=str, default="out/reassignment_test", help="Output directory")
    parser.add_argument("--hysteresis-threshold", type=float, default=0.05, help="Hysteresis threshold")
    parser.add_argument("--enable-jitter", action="store_true", help="Enable network jitter")
    parser.add_argument("--jitter-percent", type=float, default=10.0, help="Base jitter percentage")
    parser.add_argument("--enable-packet-loss", action="store_true", help="Enable packet loss")
    parser.add_argument("--packet-loss-rate", type=float, default=0.01, help="Packet loss rate")
    parser.add_argument("--enable-wifi", action="store_true", help="Enable WiFi mobility simulation")
    parser.add_argument("--num-steps", type=int, default=12, help="Number of time steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug-hysteresis", action="store_true", help="Enable hysteresis debug output")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Enable debug output if requested
    if args.debug_hysteresis:
        os.environ['DEBUG_HYSTERESIS'] = '1'
        print("🔍 DEBUG: Hysteresis logging enabled\n")
    
    # Setup platform
    env = simpy.Environment()
    clusters, all_nodes = PlatformTopology(env, args.platform).build()
    if args.services_xml and os.path.exists(args.services_xml):
        service_loader = ServiceDefinitionLoader(args.services_xml, simpy_env=None)
        services = service_loader.load_services()
    else:
        services = TrainUtils.build_services()
    TrainUtils.validate_locked_nodes(services, all_nodes)
    
    # Filter non-schedulable nodes to match training environment
    locked_node_ids = set()
    for s in services:
        locked_nid = getattr(s, 'locked_node_id', None)
        if locked_nid is not None:
            locked_node_ids.add(str(locked_nid))
            
    filtered_nodes = [
        n for n in all_nodes
        if getattr(n, 'schedulable', True) or str(getattr(n, 'node_id', '')) in locked_node_ids
    ]
    
    # Build latency matrix
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    latency = NetworkUtil.build_latency_matrix_with_jitter(
        args.platform,
        filtered_nodes,
        enable_jitter=args.enable_jitter,
        jitter_percent=args.jitter_percent,
        enable_packet_loss=args.enable_packet_loss,
        packet_loss_rate=args.packet_loss_rate,
        seed=args.seed)
    
    wifi_controller = None
    if args.enable_wifi:
        wifi_controller = WiFiMobilityController()
        # Default WiFi setup (matches simulator config)
        wifi_controller.register_access_point("edge-gpu1", 0.0, 0.0, 20.0)
        wifi_controller.register_access_point("edge-gpu2", 30.0, 0.0, 20.0)
        wifi_controller.register_robot("frank-2", 0.0, 0.0, 0.2, 0.0)
        wifi_controller.register_wifi_link("frank-2-wifi6", "frank-2")

    metrics_facade = SimulatorMetricsFacade(filtered_nodes, latency_matrix=latency, wifi_controller=wifi_controller)
    
    # Create orchestrators
    orchestrators = {}
    orchestrators["NSGA-II"] = NSGA2Orchestrator(
        services, filtered_nodes, latency, generations=5, metrics=metrics_facade
    )
    
    # Always add these for testing validation loop (no training)
    orchestrators["DQN"] = DQNOrchestrator(services, filtered_nodes, latency, metrics=metrics_facade, episodes=0)
    orchestrators["GreedyWeighted"] = GreedyWeightedOrchestrator(services, filtered_nodes, latency, metrics=metrics_facade)
    orchestrators["Autoencoder"] = AutoEncoderOrchestrator(services, filtered_nodes, latency, metrics=metrics_facade)

    if args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        dqn_ckpt_path = os.path.join(args.checkpoint_dir, "DQN_checkpoint.pt")
        if os.path.exists(dqn_ckpt_path):
            orchestrators["DQN"].load_checkpoint(dqn_ckpt_path)

        autoencoder_ckpt_path = os.path.join(args.checkpoint_dir, "AutoEncoder_checkpoint.pt")
        if os.path.exists(autoencoder_ckpt_path):
            orchestrators["Autoencoder"].load_checkpoint(autoencoder_ckpt_path)
    
    print(f"\n{'='*100}")
    print(f"REASSIGNMENT STABILITY TEST")
    print(f"Platform: {args.platform}")
    print(f"Orchestrators: {', '.join(orchestrators.keys())}")
    print(f"Hysteresis: {args.hysteresis_threshold*100:.1f}%")
    print(f"{'='*100}\n")
    
    # Initialize baseline loads (10-15% CPU, 15-20% RAM)
    scenario_manager = DynamicScenario()
    scenario_manager.initialize_baseline_loads(filtered_nodes)
    print(f"🔧 Initialized baseline loads for {len(filtered_nodes)} nodes")
    for node in filtered_nodes:
        baseline = scenario_manager.get_baseline_load(node.node_id, 0)
        print(f"   {node.node_id}: baseline ~{baseline['cpu']:.2f} CPU, ~{baseline['ram']:.2f} GB RAM")
    print()
    
    # Define scenario sequence to trigger various reassignment behaviors
    # This sequence is designed to FORCE reassignments by creating violations
    scenarios = [
        "normal",          # Step 1: Baseline - all services placed optimally
        "normal",          # Step 2: Stay stable (hysteresis should keep)
        "extreme_rate",    # Step 3: VIOLATION! Force reassignment to stronger nodes
        "extreme_rate",    # Step 4: Stay on stronger nodes
        "normal",          # Step 5: Return to normal (may reassign back OR stay due to hysteresis)
        "extreme_rate",    # Step 6: VIOLATION again! Must reassign
        "light_load",      # Step 7: Low load - opportunity for energy optimization
        "light_load",      # Step 8: Stay optimized
        "heavy_exec",      # Step 9: Long execution times - VIOLATION! Different trigger
        "normal",          # Step 10: Return to normal
        "extreme_rate",    # Step 11: High load again (test pattern learning)
        "normal",          # Step 12: Final return to baseline
    ]
    
    # Run simulation
    time_series_results = {name: [] for name in orchestrators.keys()}
    current_assignments = {name: None for name in orchestrators.keys()}
    reassignment_counts = {name: 0 for name in orchestrators.keys()}
    cumulative_battery_usage = {name: defaultdict(float) for name in orchestrators.keys()}
    
    sim_time_s = 0.0
    step_seconds = 5.0 * 60.0
    for step, scenario in enumerate(scenarios[:args.num_steps]):
        if wifi_controller is not None:
            wifi_controller.update(sim_time_s)
            latency = metrics_facade.get_latency_matrix(filtered_nodes)
            for orch in orchestrators.values():
                orch.latency_matrix = latency
            sim_time_s += step_seconds
        if args.services_xml and os.path.exists(args.services_xml):
            service_loader = ServiceDefinitionLoader(args.services_xml, simpy_env=None)
            services = service_loader.load_services()
        else:
            services = TrainUtils.build_services()  # Reset services to default
        TrainUtils.validate_locked_nodes(services, all_nodes)
        scenario_name = apply_scenario(services, scenario)
        
        print(f"\n{'─'*100}")
        print(f"⏰ Step {step+1}/{min(args.num_steps, len(scenarios))}: Scenario = {scenario_name}")
        print(f"{'─'*100}\n")
        
        for name, orch in orchestrators.items():
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

            # Check for reassignment
            reassigned = False
            if current is not None:
                changed = any(new_assignments.get(sid) != current.get(sid) for sid in new_assignments)
                if changed:
                    reassignment_counts[name] += 1
                    reassigned = True
                    print(f"  {name}: ↻ Reassignment #{reassignment_counts[name]}")
                    # Show what changed
                    for sid in new_assignments:
                        if new_assignments.get(sid) != current.get(sid):
                            print(f"    • {sid}: {current.get(sid)} → {new_assignments.get(sid)}")
                else:
                    print(f"  {name}: ✓ Kept current assignment")
            else:
                print(f"  {name}: Initial assignment")

            current_assignments[name] = copy.deepcopy(new_assignments)

            # Calculate metrics with baseline loads, then apply strict orchestrator validation
            total_latency, total_energy, violations, battery_usage, node_usage = TrainUtils.calculate_metrics(
                new_assignments, services, filtered_nodes, latency, 
                time_step_minutes=5,
                scenario=scenario_manager,
                step=step
            )

            strict_violations = TrainUtils.strict_validate_placement(new_assignments, services, filtered_nodes, latency)
            if strict_violations:
                violations = strict_violations
            if inference_failed:
                if not violations:
                    violations = []
                violations.append("Inference failure: no assignment returned")

            if total_latency is None:
                continue

            # Print number of violations for this orchestrator at this step
            if len(violations) > 0:
                print(f"  {name}: Violations = {len(violations)}")
                for v in violations:
                    print(f"    - {v}")
            else:
                print(f"  {name}: Violations = 0")

            for node_id, usage in battery_usage.items():
                cumulative_battery_usage[name][node_id] += usage["energy_ah"]

            result = {
                "time": f"T={step*5}min",
                "step": step,
                "scenario": scenario_name,
                "latency": total_latency,
                "energy": total_energy,
                "violations": len(violations),
                "assignments": copy.deepcopy(new_assignments),
                "battery_usage": copy.deepcopy(battery_usage),
                "node_usage": copy.deepcopy(node_usage),
                "reassigned": reassigned
            }
            time_series_results[name].append(result)
    
    # Save results
    output_file = os.path.join(args.output_dir, "dynamic_results.json")
    output_data = {
        "config": {
            "num_steps": args.num_steps,
            "hysteresis_threshold": args.hysteresis_threshold,
            "scenarios": scenarios[:args.num_steps],
            "seed": args.seed
        },
        "results": time_series_results,
        "reassignment_counts": reassignment_counts,
        "cumulative_battery_usage": {k: dict(v) for k, v in cumulative_battery_usage.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}\n")
    
    summary_table = []
    for name in orchestrators.keys():
        results = time_series_results[name]
        if not results:
            continue
        
        avg_latency = np.mean([r["latency"] for r in results])
        total_energy = sum([r["energy"] for r in results])
        total_violations = sum([r["violations"] for r in results])
        reassignments = reassignment_counts[name]
        
        summary_table.append([
            name,
            f"{avg_latency:.4f}s",
            f"{total_energy:.2f}J",
            total_violations,
            reassignments
        ])
    
    print(tabulate(
        summary_table,
        headers=["Orchestrator", "Avg Latency", "Total Energy", "Violations", "Reassignments"],
        tablefmt="grid"
    ))
    
    print(f"\n📊 Results saved to: {output_file}")
    print(f"\nNow run the stability validator:")
    print(f"  python src/validate.py --results {output_file} --output-dir {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
