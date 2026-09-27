"""
Train the DQN (supervised) orchestrator with the same configuration and scenarios.
Checkpoints and assignments are saved in the same output directory for easy comparison.
"""
# Install logger workaround BEFORE any BlazeAIoT imports to avoid circular imports
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

import argparse
import os
import numpy as np
import random
import math
import torch

# Check and log CUDA availability
cuda_available = torch.cuda.is_available()
cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
print(f"\n🔧 CUDA Configuration: Available={cuda_available}, Device={cuda_device}\n")

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.platform_topology import PlatformTopology
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.dqn_orchestrator import DQNOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.autoencoder_orchestrator import AutoEncoderOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.simgrid_util import SimgridUtil
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.network_util import NetworkUtil
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.topology_util import TopologyUtil
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_loader import ServiceDefinitionLoader
from train_utils import TrainUtils, DynamicScenario
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_scenario_manager import RobotTravelScenario, get_wifi_scenario_manager
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_orchestration_adapter import WiFiOrchestrationAdapter
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_training_scenario import WiFiDynamicScenario
import simpy
import json
from tqdm import tqdm


def run_training(OrchestratorClass, name, all_nodes, services, latency, episodes_per_scenario, out_dir, args, scenario_manager, reuse_orch=None):
    episodes_to_run = episodes_per_scenario
    
    if reuse_orch is not None:
        print(f"\nContinuing {name} training from previous scenario for {episodes_to_run} episodes...")
        orch = reuse_orch
        # Update orchestrator with new services and nodes
        orch.services = services
        orch.num_services = len(services)
        orch.nodes = all_nodes
        orch.num_nodes = len(all_nodes)
        orch.latency_matrix = latency
    else:
        print(f"\nTraining {name} for {episodes_to_run} episodes...")
        orch = OrchestratorClass(services, all_nodes, latency)
    violation_counts = []
    final_assignment = None

    def _build_random_assignment() -> dict:
        """Randomize current assignment within eligibility constraints."""
        node_ids = []
        node_to_layer = {}
        node_labels = {}
        for n in all_nodes:
            nid = getattr(n, 'node_id', None) or getattr(n, 'id', None)
            if not nid:
                continue
            node_ids.append(nid)
            layer_id = getattr(n, 'layer_id', None) or getattr(n, 'layer', None)
            node_to_layer[nid] = layer_id
            labels = getattr(n, 'labels', None) or getattr(n, 'node_labels', None)
            if isinstance(labels, dict):
                node_labels[nid] = labels

        assignment = {}
        if not node_ids:
            return assignment

        for svc in services:
            sid = str(getattr(svc, 'service_id', None) or getattr(svc, 'id', None))
            if not sid:
                continue
            eligible = svc.get_eligible_nodes(
                all_node_ids=node_ids,
                node_to_layer_map=node_to_layer,
                node_labels=node_labels if node_labels else None,
                current_node_id=None
            )
            if eligible:
                assignment[sid] = random.choice(eligible)

        return assignment
    
    # Progress bar for training episodes
    pbar = tqdm(range(episodes_to_run), desc=f"Training {name}", unit="episode", leave=True)
    
    for ep in pbar:
        current_assignment = _build_random_assignment() if episodes_to_run == 1 else None
        try:
            assignment = orch.optimize(services, current_assignment=current_assignment)
        except RuntimeError as exc:
            violations = [str(exc)]
            violation_counts.append(len(violations))
            final_assignment = {}
            pbar.set_postfix({"Violations": len(violations), "Avg": f"{sum(violation_counts)/len(violation_counts):.1f}"})
            continue
        
        # Use TrainUtils.calculate_metrics for consistent violation checking with baseline loads
        total_latency, total_energy, violations, battery_usage, node_usage = TrainUtils.calculate_metrics(
            assignment, services, all_nodes, latency,
            time_step_minutes=5,
            scenario=scenario_manager,
            step=ep
        )
        strict_violations = TrainUtils.strict_validate_placement(assignment, services, all_nodes, latency)
        if strict_violations:
            violations = strict_violations
        if total_latency is None:
            violations = [f"Assignment returned None or invalid"]
        
        violation_counts.append(len(violations))
        final_assignment = assignment
        # Update progress bar with current violation count
        pbar.set_postfix({"Violations": len(violations), "Avg": f"{sum(violation_counts)/len(violation_counts):.1f}"})
        
        # Only show violations if very severe or at major milestones
        if len(violations) > 10 or (ep % max(500, episodes_to_run//4) == 0 and ep > 0 and violations):
            tqdm.write(f"[{name}][Episode {ep+1}/{episodes_to_run}] Violations: {len(violations)}")
    # Close progress bar
    pbar.close()
    
    # Print training summary
    avg_violations = sum(violation_counts) / len(violation_counts) if violation_counts else 0
    final_violations = violation_counts[-1] if violation_counts else 0
    print(f"✓ Training Summary for {name}:")
    print(f"  Episodes: {episodes_to_run}, Avg violations: {avg_violations:.1f}, Final violations: {final_violations}")
    if final_assignment:
        assigned_services = sum(1 for v in final_assignment.values() if v)
        print(f"  Services assigned: {assigned_services}/{len(services)}")
    
    # Save assignment and checkpoint
    with open(os.path.join(out_dir, f"assignments_{name}.json"), "w") as f:
        json.dump({"service_to_node": assignment}, f, indent=2)
    ckpt_path = os.path.join(out_dir, f"{name}_checkpoint.pt")
    try:
        orch.save_checkpoint(ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    except Exception as e:
        print(f"[WARN] Could not save checkpoint: {e}")
    print(f"Training complete for {name}. Artifacts in {out_dir}")
    return orch


def main():
    parser = argparse.ArgumentParser(description="Train DQN orchestrator with unified config")
    parser.add_argument("--episodes-per-scenario", type=int, default=200, help="Episodes per scenario")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-dir", type=str, default="out/joint_train")
    parser.add_argument("--bg-duration", type=float, default=60.0)
    parser.add_argument("--bg-min", type=float, default=0.2)
    parser.add_argument("--bg-max", type=float, default=0.7)
    parser.add_argument("--scenarios", type=str, default="all", help="Comma-separated list of scenarios (cpu50,ram50,mix50,gpu_hot,layer_hot,robot_travel,all). Empty=single scenario.")
    parser.add_argument("--enable-jitter", action="store_true", help="Enable network jitter")
    parser.add_argument("--jitter-percent", type=float, default=10.0, help="Base jitter percentage")
    parser.add_argument("--enable-packet-loss", action="store_true", help="Enable packet loss")
    parser.add_argument("--packet-loss-rate", type=float, default=0.01, help="Packet loss rate")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list: dqn,autoencoder or 'all'")
    parser.add_argument("--enable-wifi", action="store_true", help="Enable WiFi mobility simulation")
    parser.add_argument("--platform-xml", type=str, default=None, help="Path to platform.xml (default: src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/Simulator/platform.xml)")
    parser.add_argument("--services-xml", type=str, default=None, help="Path to services.xml for service definitions")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.platform_xml:
        platform_xml = args.platform_xml
    else:
        platform_xml = os.path.join(os.path.dirname(__file__), "EdgeLogiQ/BlazeAIoT/BlazeMaestro/Simulator/platform.xml")

    tmp_env = simpy.Environment()
    tmp_layers, tmp_nodes = PlatformTopology(tmp_env, platform_xml).build()
    layers_by_id = {l.layer_id: list(l.get_nodes()) for l in tmp_layers}
    all_nodes_flat = list(tmp_nodes)

    scenario_arg = (args.scenarios or "").strip()
    if scenario_arg:
        names = [s.strip() for s in scenario_arg.split(",") if s.strip()]
        if "all" in names:
            names = ["cpu50", "ram50", "mix50", "gpu_hot", "layer_hot", "robot_travel"]
    else:
        names = ["cpu50", "ram50", "mix50", "gpu_hot", "layer_hot", "robot_travel"]

    # Always initialize SimGrid engine once at start (user requirement: always use SimGrid)
    engine = SimgridUtil.get_engine(platform_xml)
    if engine is None:
        raise RuntimeError("SimGrid not available but required (user requested always use SimGrid).")

    scenario_configs = []
    for name in names:
        if name == "robot_travel":
            # WiFi robot travel scenario
            config = {"name": "robot_travel", "type": "wifi", "hot_sets": [], "preassigned": False}
            scenario_configs.append(config)
            config_pre = {"name": "robot_travel_preassigned", "type": "wifi", "hot_sets": [], "preassigned": True}
            scenario_configs.append(config_pre)
            continue
        if name == "layer_hot":
            for lid, lnodes in layers_by_id.items():
                hot_sets = [
                    (lnodes, {"cpu": 0.8, "ram": 0.8, "gpu": 0.0, "dur": args.bg_duration})
                ]
                config = {"name": f"layer_hot:{lid}", "hot_sets": hot_sets, "preassigned": False}
                scenario_configs.append(config)
                # Add preassigned version
                config_pre = {
                    "name": f"layer_hot:{lid}_preassigned",
                    "hot_sets": hot_sets,
                    "preassigned": True,
                }
                scenario_configs.append(config_pre)
            continue
        hot_sets = []
        if name == "cpu50":
            sel = TrainUtils.choose_nodes(all_nodes_flat, 0.5)
            hot_sets.append((sel, {"cpu": 0.8, "ram": 0.1, "gpu": 0.0, "dur": args.bg_duration}))
        elif name == "ram50":
            sel = TrainUtils.choose_nodes(all_nodes_flat, 0.5)
            hot_sets.append((sel, {"cpu": 0.1, "ram": 0.8, "gpu": 0.0, "dur": args.bg_duration}))
        elif name == "mix50":
            sel = TrainUtils.choose_nodes(all_nodes_flat, 0.5)
            hot_sets.append((sel, {"cpu": 0.8, "ram": 0.8, "gpu": 0.0, "dur": args.bg_duration}))
        elif name == "gpu_hot":
            sel = set(
                n.node_id
                for n in all_nodes_flat
                if (
                    hasattr(n, "gpu") and getattr(n.gpu, "capacity", 0.0) > 0.0
                ) or (
                    hasattr(n, "node_id") and isinstance(n.node_id, str) and "-gpu" in n.node_id
                )
            )
            if sel:
                hot_sets.append((sel, {"cpu": 0.3, "ram": 0.3, "gpu": 0.8, "dur": args.bg_duration}))
        elif name == "tight_constraints":
            # Constraint-heavy scenario: high utilization on most nodes
            fog_cpu_nodes = ["fog-cpu2", "fog-cpu3", "fog-cpu5"]
            fog_gpu_nodes = ["fog-gpu1", "fog-gpu2"]
            hot_sets.append((fog_cpu_nodes, {"cpu": 0.7, "ram": 0.6, "gpu": 0.0, "dur": args.bg_duration}))
            hot_sets.append((fog_gpu_nodes, {"cpu": 0.6, "ram": 0.7, "gpu": 0.5, "dur": args.bg_duration}))
        elif name == "resource_exhaustion":
            # Extreme scenario: most nodes heavily loaded
            exhausted_nodes = ["fog-cpu2", "fog-cpu3", "fog-cpu5", "cloud-cpu1"]
            hot_sets.append((exhausted_nodes, {"cpu": 0.8, "ram": 0.75, "gpu": 0.0, "dur": args.bg_duration}))
        else:
            print(f"Unknown scenario '{name}', skipping")
            continue
        config = {"name": name, "hot_sets": hot_sets, "preassigned": False}
        scenario_configs.append(config)
        # Add preassigned version
        config_pre = {"name": f"{name}_preassigned", "hot_sets": hot_sets, "preassigned": True}
        scenario_configs.append(config_pre)
        
    total_episodes = args.episodes_per_scenario * len(scenario_configs)
        
    # Build platform topology once for all episodes (major speedup)
    print("Building platform topology (one-time setup)...")
    base_env = simpy.Environment()
    base_layers, all_base_nodes = PlatformTopology(base_env, platform_xml).build()
    
    # Load services from XML if provided, otherwise use hardcoded definitions
    if args.services_xml:
        print(f"Loading services from {args.services_xml}...")
        service_loader = ServiceDefinitionLoader(args.services_xml, simpy_env=None)
        base_services = service_loader.load_services()
        print(f"Loaded {len(base_services)} services from XML")
    else:
        base_services = TrainUtils.build_services()
    TrainUtils.validate_locked_nodes(base_services, all_base_nodes)
    
    # Filter non-schedulable nodes BEFORE creating orchestrators
    # This ensures training and runtime see the same node count
    locked_node_ids = set()
    for s in base_services:
        locked_nid = getattr(s, 'locked_node_id', None)
        if locked_nid is not None:
            locked_node_ids.add(str(locked_nid))
    
    base_nodes = [
        n for n in all_base_nodes
        if getattr(n, 'schedulable', True) or str(getattr(n, 'node_id', '')) in locked_node_ids
    ]
    
    num_filtered = len(all_base_nodes) - len(base_nodes)
    if num_filtered > 0:
        filtered_ids = [str(getattr(n, 'node_id', '')) for n in all_base_nodes if n not in base_nodes]
        print(f"Pre-filtered {num_filtered} non-schedulable node(s): {', '.join(filtered_ids)}")
    
    # Build latency matrix (with or without jitter)
    latency = NetworkUtil.build_latency_matrix_with_jitter(
        platform_xml,
        base_nodes,
        enable_jitter=args.enable_jitter,
        jitter_percent=args.jitter_percent,
        enable_packet_loss=args.enable_packet_loss,
        packet_loss_rate=args.packet_loss_rate,
        seed=args.seed)
    print(f"Platform built: {len(base_nodes)} nodes, {len(base_services)} services")
    
    # Initialize baseline loads (10-15% CPU, 15-20% RAM)
    scenario_manager = DynamicScenario()
    scenario_manager.initialize_baseline_loads(base_nodes)
    print(f"\n🔧 Initialized baseline loads for {len(base_nodes)} nodes")
    for node in base_nodes[:5]:  # Show first 5 as sample
        baseline = scenario_manager.get_baseline_load(node.node_id, 0)
        print(f"   {node.node_id}: baseline ~{baseline['cpu']:.2f} CPU, ~{baseline['ram']:.2f} GB RAM")
    if len(base_nodes) > 5:
        print(f"   ... and {len(base_nodes)-5} more nodes")
    print()

    # Select orchestrators to train
    selected_models = [m.strip().lower() for m in (args.models or "").split(",") if m.strip()]
    if not selected_models or "all" in selected_models:
        selected_models = ["dqn", "autoencoder"]

    train_list: list = []
    if "dqn" in selected_models:
        dqn_orch = { "name": "DQN",
            "orchestrator": DQNOrchestrator(base_services, base_nodes, latency, episodes=10),
            "pbar": tqdm(range(total_episodes), desc="DQN Training", unit="episode", leave=True),
            "violations": [],
        }
        train_list.append(dqn_orch)
    
    if "autoencoder" in selected_models:
        autoencoder_orch = {
            "name": "AutoEncoder",
            "orchestrator": AutoEncoderOrchestrator(base_services, base_nodes, latency),
            "pbar": tqdm(range(total_episodes), desc="AutoEncoder Training", unit="episode", leave=True),
            "violations": [],
        }
        train_list.append(autoencoder_orch)

    # Define dynamic service scenarios for training
    # CRITICAL: Include "peak_traffic" to match compare script's 1.5x rate multiplier at T=10-15min
    service_scenarios = [
        "normal",           # Baseline with ±5% variation
        "peak_traffic",     # 1.5x rates (matches compare script!) - IMPORTANT for DQN
        "peak_traffic",     # Doubled probability - this is where violations occur
        "stress_test",      # 2.0x rates - even more aggressive
        "increased_workload",  # 1.2x exec_time (matches compare script T=15-20min)
        "extreme_rate",     # Fixed high rates (18-30Hz)
        "heavy_exec",       # Long execution times
        "light_load"        # 0.5x rates (matches compare script T=25-30min)
    ]
    
    for item in train_list or []:
        name = item["name"]
        orch = item["orchestrator"]
        pbar = item["pbar"]
        reassignment_counts = []
        for ep in pbar:
            # Randomly apply service scenario variations
            service_scenario = random.choice(service_scenarios)
            
            config = random.choice(scenario_configs)
            
            # Handle WiFi robot travel scenario separately
            if config.get("type") == "wifi" and args.enable_wifi:
                env = simpy.Environment()
                layers, all_nodes = PlatformTopology(env, platform_xml).build()
                
                # Initialize WiFi components
                wifi_controller = WiFiMobilityController()
                wifi_adapter = WiFiOrchestrationAdapter(wifi_controller)
                
                # Create and initialize WiFi scenario
                robot_travel_scenario = RobotTravelScenario(duration_seconds=600.0, travel_speed=0.33)
                wifi_adapter.initialize_with_scenario(robot_travel_scenario, time_step=1.0)
                
                # Create WiFi-aware training scenario
                wifi_scenario_manager = WiFiDynamicScenario(
                    duration_minutes=10,
                    time_step_minutes=1,
                    wifi_adapter=wifi_adapter
                )
                wifi_scenario_manager.initialize_baseline_loads(all_nodes)
                
                env.run(until=1e-6)
                
                # Load services
                if args.services_xml:
                    service_loader = ServiceDefinitionLoader(args.services_xml, simpy_env=None)
                    services = service_loader.load_services()
                else:
                    services = TrainUtils.build_services()
                
                # Apply dynamic service scenario
                TrainUtils.apply_service_scenario(services, service_scenario)
                TrainUtils.validate_locked_nodes(services, all_nodes)
                
                # Filter nodes
                locked_node_ids = set()
                for s in services:
                    locked_nid = getattr(s, 'locked_node_id', None)
                    if locked_nid is not None:
                        locked_node_ids.add(str(locked_nid))
                
                filtered_nodes = [
                    n for n in all_nodes
                    if getattr(n, 'schedulable', True) or str(getattr(n, 'node_id', '')) in locked_node_ids
                ]
                
                # Update orchestrator with current WiFi positions
                orch.nodes = filtered_nodes
                orch.num_nodes = len(filtered_nodes)
                
                # Get position-aware latency matrix
                updated_latency = wifi_adapter.update_latency_matrix_with_positions(latency, filtered_nodes)
                orch.latency_matrix = updated_latency
                
                orch.services = services
                orch.num_services = len(services)
                
                # Training with WiFi scenario
                try:
                    assignment = orch.optimize(services)
                except RuntimeError as exc:
                    violations = [str(exc)]
                    dqn_violation_count = len(violations)
                else:
                    total_latency, total_energy, violations, battery_usage, node_usage = TrainUtils.calculate_metrics(
                        assignment, services, filtered_nodes, updated_latency,
                        time_step_minutes=1,
                        scenario=wifi_scenario_manager,
                        step=ep
                    )
                    dqn_violation_count = len(violations) if violations else 0
                
            else:
                # Standard scenario handling
                env = simpy.Environment()
                layers, all_nodes = PlatformTopology(env, platform_xml).build()
                # Apply scenario-specific background loads
                for node in all_nodes:
                    node_id = getattr(node, "node_id", "")
                    applied = False
                    for node_set, loads in config.get("hot_sets", []):
                        if node_id in node_set:
                            TopologyUtil.add_background_load(
                                env,
                                node,
                                cpu_frac=loads.get("cpu", 0.0),
                                ram_frac=loads.get("ram", 0.0),
                                gpu_frac=loads.get("gpu", 0.0),
                                duration=loads.get("dur", args.bg_duration)
                            )
                            applied = True
                            break
                
                # Apply minimal baseline loads to all nodes using DynamicScenario
                # This ensures consistent 10-15% CPU and 15-20% RAM baseline
                for node in all_nodes:
                    baseline = scenario_manager.get_baseline_load(node.node_id, ep)
                    # Convert absolute values to fractions for TopologyUtil
                    cpu_capacity = getattr(node, 'cpu', 1.0)
                    ram_capacity = getattr(node, 'ram', 8.0)
                    cpu_frac = baseline['cpu'] / cpu_capacity if cpu_capacity > 0 else 0.1
                    ram_frac = baseline['ram'] / ram_capacity if ram_capacity > 0 else 0.15
                    TopologyUtil.add_background_load(env, node, cpu_frac=cpu_frac, ram_frac=ram_frac, gpu_frac=0.0, duration=args.bg_duration)
                
                env.run(until=1e-6)
                
                # Load services from XML if provided, otherwise use hardcoded definitions
                if args.services_xml:
                    service_loader = ServiceDefinitionLoader(args.services_xml, simpy_env=None)
                    services = service_loader.load_services()
                else:
                    services = TrainUtils.build_services()
                
                # Apply dynamic service scenario (extreme_rate, heavy_exec, light_load, normal)
                TrainUtils.apply_service_scenario(services, service_scenario)
                TrainUtils.validate_locked_nodes(services, all_nodes)
                
                # Filter non-schedulable nodes to match base_nodes filtering (consistent with training init)
                locked_node_ids = set()
                for s in services:
                    locked_nid = getattr(s, 'locked_node_id', None)
                    if locked_nid is not None:
                        locked_node_ids.add(str(locked_nid))
                
                filtered_nodes = [
                    n for n in all_nodes
                    if getattr(n, 'schedulable', True) or str(getattr(n, 'node_id', '')) in locked_node_ids
                ]
                
                # Update orchestrator's nodes/services before any optimize call
                orch.nodes = filtered_nodes
                orch.num_nodes = len(filtered_nodes)
                orch.latency_matrix = latency
                
                # Preassigned scenario: generate initial assignment and count reassignments
                initial_assignment = None
                reassigned_count = 0
                if config.get("preassigned", False):
                    # Use orchestrator to generate initial assignment
                    try:
                        initial_assignment = orch.optimize(services)
                        # Next optimize call is the actual training step
                        new_assignment = orch.optimize(services)
                    except RuntimeError:
                        initial_assignment = {}
                        new_assignment = {}
                    # Count reassignments
                    reassigned_count = sum(1 for k in new_assignment if initial_assignment.get(k) != new_assignment.get(k))
                    # Training step should use the same filtered node set
                    dqn_violation_count = orch.run_single_episode_training(filtered_nodes, services, latency)
                else:
                    # Training step should use the same filtered node set
                    dqn_violation_count = orch.run_single_episode_training(filtered_nodes, services, latency)
                reassignment_counts.append(reassigned_count)
            
            item['violations'].append(dqn_violation_count if isinstance(dqn_violation_count, int) else dqn_violation_count[1] if isinstance(dqn_violation_count, tuple) else 0)
            avg_dqn_viol = sum(item['violations']) / len(item['violations'])
            avg_reassign = sum(reassignment_counts) / len(reassignment_counts) if reassignment_counts else 0
            pbar.set_postfix({'Violations': f'{dqn_violation_count} (avg={avg_dqn_viol:.1f})', 'Reassign': f'{avg_reassign:.1f}'})
        pbar.close()

    # Save final checkpoints for both orchestrators in their respective scenario directories
    for item in train_list:
        orch = item["orchestrator"]  
        name = item["name"]

        final_dir = os.path.join(args.export_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        ckpt_path = os.path.join(final_dir, f"{name}_checkpoint.pt")
        try:
            orch.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
        except Exception as e:
            print(f"[WARN] Could not save checkpoint for {name}: {e}")
    
    # Print summary
    print("\n=== Training Summary ===")
    for idx, item in enumerate(train_list):
        name = item["name"]
        violations = item["violations"]
        avg_viol = sum(violations)/len(violations) if violations else 0
        final_viol = violations[-1] if violations else 0
        print(f"  {name}: Episodes: {total_episodes}, Avg violations: {avg_viol:.1f}, Final violations: {final_viol}")
    print(f"  Training completed across {len(scenario_configs)} scenarios")
    # Print reassignment summary for preassigned scenarios
    print("\nReassignment stats (preassigned scenarios):")
    for idx, item in enumerate(train_list):
        name = item["name"]
        # Only print for preassigned scenarios
        print(f"  {name}: Avg reassignments per episode (preassigned): (see progress bar above)")

if __name__ == "__main__":
    main()
