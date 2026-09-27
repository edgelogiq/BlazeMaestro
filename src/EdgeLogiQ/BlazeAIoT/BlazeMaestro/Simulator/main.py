"""
Main script to run the simulator with a simple random agent.
"""
import simpy
import numpy as np
import random
import sys
import os

# Ensure the repository's src directory is on sys.path when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../.."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.platform_topology import PlatformTopology
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator import simgrid_compat as sg
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.battery_manager import BatteryManager
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.fixed_drain_battery_manager import (
    FixedDrainBatteryManager,
)
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service import Service, ContainerSpec
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_loader import ServiceDefinitionLoader
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.nsga2_orchestrator import NSGA2Orchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Energy import UnsupervisedEnergyEstimator

def main():
    """
    Main function to set up and run the simulation, executed as a SimGrid actor.
    """
    # 1. Initialize SimPy environment
    env = simpy.Environment()

    # 2. Create clusters and nodes from platform.xml
    platform_file = os.path.join(os.path.dirname(__file__), "platform.xml")
    clusters, all_nodes = PlatformTopology(env, platform_file).build()

    # 3. Workloads: Services loaded from services.xml (uses min/max usage metrics when available)
    services_xml = os.path.join(SRC_ROOT, "src/config/services.xml")
    service_loader = ServiceDefinitionLoader(services_xml, simpy_env=env)
    services: list[Service] = service_loader.load_services()
    # Example dependencies: you can add deps by constructing Service with dependencies=[DependencySpec("other-svc", size_mb=50)]
    services_map = {s.service_id: s for s in services}

    # 4. No Gym env: we directly schedule services using an orchestrator

    print("--- Starting Simulation ---")
    terminated = False

    # Battery management (abstracted). 1 Wh = 3600 J.
    batt: BatteryManager = FixedDrainBatteryManager()
    batt.set_poweroff_fn(lambda hid: sg.Host.by_name(hid).turn_off())

    # Per-host battery (Wh) and power config (Watts)
    HOST_WH = {"frank-1": 80.0, "frank-2": 80.0}
    HOST_P_IDLE = {"frank-1": 5.0, "frank-2": 5.0}
    HOST_P_ACTIVE = {"frank-1": 35.0, "frank-2": 35.0}

    for hid, wh in HOST_WH.items():
        batt.register(hid, wh * 3600.0)
    # Power model (Watts)
    P_IDLE_DEFAULT = 5.0
    P_ACTIVE_DEFAULT = 35.0

    def compute_actor(sim_seconds: float):
        """Run compute for a given duration in simulated seconds on the current host using SimGrid exec."""
        host = sg.this_actor.get_host()
        # speed() returns flop/s; multiply by seconds to get total flop cost
        try:
            if hasattr(host, "speed"):
                sp_attr = getattr(host, "speed")
                val = sp_attr() if callable(sp_attr) else sp_attr
                sp = float(val) if isinstance(val, (int, float)) else 0.0
            else:
                raise AttributeError
        except Exception:
            try:
                val2 = host.pstate_speed(host.pstate())
                sp = float(val2) if isinstance(val2, (int, float)) else 0.0
            except Exception:
                sp = 0.0
        flops = max(0.0, float(sp) * float(sim_seconds))
        if flops > 0:
            sg.this_actor.execute(flops)
    # Build a latency matrix using SimGrid routing between nodes
    print("Building latency matrix...")
    latency_matrix = np.zeros((len(all_nodes), len(all_nodes)), dtype=float)
    for i, n_from in enumerate(all_nodes):
        for j, n_to in enumerate(all_nodes):
            if i == j:
                latency_matrix[i, j] = 0.0
                continue
            try:
                route = sg.Host.by_name(n_from.node_id).route_to(sg.Host.by_name(n_to.node_id))
                if isinstance(route, tuple) and len(route) == 2:
                    latency_matrix[i, j] = float(route[1])
                else:
                    # Sum link latencies if iterable
                    total = 0.0
                    links = []
                    try:
                        links = list(route) if route is not None else []
                    except Exception:
                        links = []
                    for link in links:
                        try:
                            total += float(link.latency())
                        except Exception:
                            pass
                    latency_matrix[i, j] = total
            except Exception:
                latency_matrix[i, j] = 0.0

    # Train an unsupervised energy estimator (autoencoder) if torch is available; falls back otherwise
    uee = UnsupervisedEnergyEstimator(all_nodes)
    # Add a modest set of samples across services and nodes
    max_nodes_for_training = min(12, len(all_nodes))
    for svc in services:
        for node in all_nodes[:max_nodes_for_training]:
            uee.add_training_sample(svc, node)
    uee.fit_unsupervised_model()

    # Use NSGA2Orchestrator to decide placements; fall back to random if unavailable
    try:
        orch = NSGA2Orchestrator(services, all_nodes, latency_matrix, energy_estimator=uee, population_size=20, generations=40)
        assignment = orch.optimize(services)  # dict service_id -> node_id
        print(f"Orchestrator assignment (service -> node): {assignment}")
    except Exception as e:
        print(f"NSGA2 orchestrator unavailable ({e}); using random placement.")
        assignment = [random.randrange(len(all_nodes)) for _ in services]

    for s_idx, svc in enumerate(services):
        # assignment may be dict or list depending on the fallback
        if isinstance(assignment, dict):
            node_id = assignment.get(svc.service_id)
            node = next((n for n in all_nodes if getattr(n, 'node_id', None) == node_id), None)
        else:
            node = all_nodes[int(assignment[s_idx])]
        if node is None:
            print(f"Cannot schedule {svc.service_id}: unknown node {node_id}")
            continue
        total_ram = sum(c.ram_gb for c in svc.containers)
        total_gpu = sum(c.gpu_gb for c in svc.containers)
        ram_needed = total_ram + (total_gpu if getattr(node, "gpu_shared", False) else 0)
        has_gpu = (total_gpu == 0) or (node.gpu and node.gpu.level >= total_gpu)
        if node.ram.level >= ram_needed and has_gpu:
            print(f"Scheduling {svc.service_id} on {node.node_id}")
            env.process(svc.run(node, services_map=services_map))
            if batt.has_host(node.node_id):
                # Filter out None exec_time values for non-time-constrained services
                exec_times = [c.exec_time for c in svc.containers if c.exec_time is not None]
                duration = max(exec_times) if exec_times else 0.0
                before = batt.remaining(node.node_id)
                batt.on_task(node.node_id, duration, HOST_P_ACTIVE.get(node.node_id, P_ACTIVE_DEFAULT))
                after = batt.remaining(node.node_id)
                print(f"  -> Battery {node.node_id}: -{(before-after):.0f} J, remaining {after:.0f} J")
        else:
            print(f"Cannot schedule {svc.service_id} on {node.node_id}: insufficient resources")
    terminated = True


    # 6. Run the SimPy simulation
    print("\n--- Running SimPy Simulation ---")
    env.run()
    print("\n--- Simulation Finished ---")
    
    # 7. Clean up SimGrid
    # gym_env.close()

if __name__ == "__main__":
    # Enable energy plugin if available
    engine = sg.Engine(["--cfg=plugin:energy"])
    platform_file = os.path.join(os.path.dirname(__file__), "platform.xml")
    engine.load_platform(platform_file)
    sg.Actor.create("main_actor", sg.Host.by_name("cluster-1-master"), main)
    engine.run()
