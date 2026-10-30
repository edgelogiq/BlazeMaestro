"""Revision-campaign harness: replays the Table VI dynamic scenario.

Scenario (paper, Simulation Evaluation): 600 s, one decision every 10 s (60 decisions),
six cycles of 10 decisions: normal (I, two cycles), peak 1.5x rates (II), heavy 1.2x
execution times (III), congestion (IV), low traffic 0.5x rates (V). WiFi mobility and
network jitter enabled. Output JSON keeps the schema of out/compare_hysteresis_*.json
and adds decision times and per-step migration counts.

Example:
    PYTHONPATH=src python src/campaign/run_campaign.py --seed 42 --hysteresis 0.0 \
        --checkpoint-dir out/final_backup --output out/campaign/smoke/seed42_h0.json
"""

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.blaze_logger_mock import install_logger_workaround
install_logger_workaround()

import argparse
import zlib
import contextlib
import copy
import io
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import simpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.platform_topology import PlatformTopology
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Topology.service_loader import ServiceDefinitionLoader
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.dqn_orchestrator import DQNOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.nsga2_orchestrator import NSGA2Orchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.unsupervised_orchestrator import GreedyWeightedOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.autoencoder_orchestrator import AutoEncoderOrchestrator
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.simulator_facade import SimulatorMetricsFacade
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.orchestrator_utils import OrchestratorUtils
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.network_util import NetworkUtil
from train_utils import TrainUtils, DynamicScenario
from campaign.baselines import (KinitosInspiredBaseline, KubeDefaultBaseline, LayerPinnedBaseline, OneShotBaseline,
                                StaticInitialBaseline)

PHASES = ["normal"] * 20 + ["peak"] * 10 + ["heavy_exec"] * 10 + ["congestion"] * 10 + ["low_traffic"] * 10
UNREACHABLE_S = 5.0  # same value the WiFi model uses for a disconnected robot


def load_trace(trace_dir, nodes):
    """Per-iteration measured latencies (seconds) from a real BlazeMaestro run.

    Uses latency_matrix_prometheus.csv (node->node, ms) and the *_latency_to_frank-2_ms
    columns of dqn.csv for the robot links. Missing or zero measurements are returned as
    NaN so the platform-XML value is kept for that pair. Returns a list of dicts
    {(i, j): seconds}, one per iteration of dqn.csv.
    """
    import csv
    idx = {n.node_id: i for i, n in enumerate(nodes)}
    prom = list(csv.DictReader(open(os.path.join(trace_dir, "latency_matrix_prometheus.csv"))))
    orch = list(csv.DictReader(open(os.path.join(trace_dir, "dqn.csv"))))
    steps = []
    for k, row in enumerate(orch):
        pr = prom[min(k, len(prom) - 1)]
        pairs = {}
        for col, val in pr.items():
            if "->" not in col:
                continue
            a, b = col.split("->")
            if a in idx and b in idx and a != b and float(val) > 0:
                pairs[(idx[a], idx[b])] = float(val) / 1000.0
        for col, val in row.items():
            if col.endswith("_latency_to_frank-2_ms"):
                a = col[: -len("_latency_to_frank-2_ms")]
                if a in idx and "frank-2" in idx and a != "frank-2" and val and float(val) > 0:
                    v = float(val) / 1000.0
                    pairs[(idx[a], idx["frank-2"])] = v
                    pairs[(idx["frank-2"], idx[a])] = v
        steps.append(pairs)
    return steps


def scenario_events(name: str, step: int):
    """Infrastructure events on top of the workload phases.

    Returns (unreachable node ids, latency multiplier on the robot's links).
    - table6: none (paper's Table VI scenario).
    - disruption: fog-gpu1 (preferred GPU host) becomes unreachable during phases II-III
      (steps 20-39); then the robot WiFi degrades, ramping x1 -> x4 over steps 40-49,
      holding x4 for 50-54 and recovering over 55-59 (mirrors the tc/netem ramp of the
      physical experiment).
    - robot_pressure: the robot WiFi link is degraded x3 for the whole run, so keeping
      latency-critical services off the robot costs period margin; placing them on the
      battery node costs energy (differential-energy case, R2-5).
    """
    if name == "disruption":
        down = {"fog-gpu1"} if 20 <= step < 40 else set()
        if 40 <= step < 50:
            factor = 1.0 + 3.0 * (step - 40) / 9.0
        elif 50 <= step < 55:
            factor = 4.0
        elif 55 <= step < 60:
            factor = 4.0 - 3.0 * (step - 54) / 5.0
        else:
            factor = 1.0
        return down, factor
    if name == "robot_pressure":
        return set(), 3.0
    return set(), 1.0


ALL_POLICIES = ["NSGA-II", "DQN", "GreedyWeighted", "Autoencoder",
                "Static-initial", "OneShot-NSGA-II", "K8s-default", "Kinitos-inspired", "All-edge", "All-fog"]


def sub_seed(*parts) -> int:
    """Deterministic seed derived from (run seed, step, label): independent of which
    other policies share the run, so a given seed yields the same scenario and the same
    stochastic draws for a policy in every campaign."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def apply_phase(services, phase: str, congestion_factor: float, base_latency, latency_ref, rng=random):
    """Mutate services in place for the given phase; returns the latency matrix to use."""
    for svc in services:
        svc.execution_rate_hz = svc._baseline_rate
        for c in svc.containers:
            c.exec_time = c._baseline_exec_time
            if getattr(c, "cpu", 0) < 0.05:
                c.cpu = 0.05
            if getattr(c, "ram_gb", 0) < 0.01:
                c.ram_gb = 0.01
    for svc in services:
        if phase in ("normal", "congestion"):
            svc.execution_rate_hz = svc._baseline_rate * rng.uniform(0.95, 1.05)
        elif phase == "peak":
            svc.execution_rate_hz = svc._baseline_rate * 1.5 * rng.uniform(0.95, 1.05)
        elif phase == "heavy_exec":
            for c in svc.containers:
                if c._baseline_exec_time is not None:
                    c.exec_time = c._baseline_exec_time * 1.2 * rng.uniform(0.95, 1.05)
        elif phase == "low_traffic":
            svc.execution_rate_hz = svc._baseline_rate * 0.5 * rng.uniform(0.95, 1.05)
    if phase == "congestion" and congestion_factor != 1.0:
        return ScaledLatency(latency_ref, congestion_factor)
    return latency_ref


def latency_per_service(placement, services, nodes, latency):
    """L_s(x) of the paper (Eq. transitive_latency): link latency plus transfer time along
    blocking dependencies, recursively up the tree. Returns {service_id: L_s}."""
    idx = {n.node_id: i for i, n in enumerate(nodes)}
    by_id = {s.service_id: s for s in services}
    memo = {}

    def ls(sid, stack=()):
        if sid in memo:
            return memo[sid]
        svc, nid = by_id.get(sid), placement.get(sid)
        total = 0.0
        if svc is not None and nid in idx:
            for d in svc.dependencies or []:
                if getattr(d, "blocking", True) is False or d.service_id in stack:
                    continue
                dn = placement.get(d.service_id)
                if dn not in idx:
                    continue
                edge = float(latency[idx[dn]][idx[nid]]) + (d.size_mb / 200.0 if getattr(d, "size_mb", 0) > 0 else 0.0)
                total += edge + ls(d.service_id, stack + (sid,))
        memo[sid] = total
        return total

    return {s.service_id: ls(s.service_id) for s in services}


def blocking_view(services):
    """Shallow copies of the services keeping only blocking dependency edges."""
    view = []
    for svc in services:
        c = copy.copy(svc)
        c.dependencies = [d for d in (svc.dependencies or []) if getattr(d, "blocking", True)]
        view.append(c)
    return view


class ScaledLatency:
    """Read-only view multiplying every link latency (phase IV congestion)."""

    def __init__(self, inner, factor: float):
        self.inner, self.factor = inner, factor

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return float(self.inner[key[0]][key[1]]) * self.factor
        row = self.inner[key]
        return _ScaledRow(row, self.factor)

    def __len__(self):
        return len(self.inner) if hasattr(self.inner, "__len__") else self.inner.shape[0]


class _ScaledRow:
    def __init__(self, row, factor):
        self.row, self.factor = row, factor

    def __getitem__(self, j):
        return float(self.row[j]) * self.factor


def build_policies(names, services, nodes, latency, metrics, args):
    policies = {}
    for name in names:
        if name == "NSGA-II":
            policies[name] = NSGA2Orchestrator(services, nodes, latency, population_size=args.nsga_pop,
                                               generations=args.nsga_gen, metrics=metrics)
        elif name == "DQN":
            p = DQNOrchestrator(services, nodes, latency, metrics=metrics, episodes=0)
            p.load_checkpoint(os.path.join(args.checkpoint_dir, "DQN_checkpoint.pt"))
            policies[name] = p
        elif name == "GreedyWeighted":
            policies[name] = GreedyWeightedOrchestrator(services, nodes, latency, metrics=metrics)
        elif name == "Autoencoder":
            p = AutoEncoderOrchestrator(services, nodes, latency, metrics=metrics)
            p.load_checkpoint(os.path.join(args.checkpoint_dir, "AutoEncoder_checkpoint.pt"))
            policies[name] = p
        elif name == "Static-initial":
            policies[name] = StaticInitialBaseline(services, nodes)
        elif name == "OneShot-NSGA-II":
            inner = NSGA2Orchestrator(services, nodes, latency, population_size=args.nsga_pop,
                                      generations=args.nsga_gen, metrics=metrics)
            policies[name] = OneShotBaseline(inner, name)
        elif name == "K8s-default":
            policies[name] = KubeDefaultBaseline(services, nodes)
        elif name == "Kinitos-inspired":
            policies[name] = KinitosInspiredBaseline(services, nodes)
        elif name == "All-edge":
            policies[name] = LayerPinnedBaseline(services, nodes, "edge")
        elif name == "All-fog":
            policies[name] = LayerPinnedBaseline(services, nodes, "fog")
        else:
            raise ValueError(f"unknown policy {name}")
    return policies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="src/config/platform_sim.xml")
    ap.add_argument("--services-xml", default="src/config/services_sim.xml")
    ap.add_argument("--checkpoint-dir", default="out/final_backup")
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hysteresis", type=float, default=0.0)
    ap.add_argument("--hysteresis-mode", choices=["legacy", "uniform"], default="uniform",
                    help="uniform: the hysteresis filter is applied to NSGA-II too, as the live loop does "
                         "(OrchestrationContext); legacy: NSGA2Orchestrator.optimize skips it")
    ap.add_argument("--nsga-pop", type=int, default=20)
    ap.add_argument("--nsga-gen", type=int, default=40)
    ap.add_argument("--step-seconds", type=float, default=10.0)
    ap.add_argument("--jitter-percent", type=float, default=10.0)
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--no-wifi", action="store_true")
    ap.add_argument("--robot-speed", type=float, default=0.05,
                    help="m/s along x; 0.05 moves frank-2 from AP edge-gpu1 (0 m) to edge-gpu2 (30 m) "
                         "over the 600 s run. At 0.2 m/s the robot leaves coverage after ~250 s.")
    ap.add_argument("--migration-budget", type=int, default=-1,
                    help="max services moved per cycle while the running placement is feasible (-1 = no limit)")
    ap.add_argument("--trace-dir", default=None,
                    help="replay the measured latencies of a real run (e.g. out/20260319_080017); "
                         "service rates stay at their measured baseline, no WiFi model, no jitter")
    ap.add_argument("--num-steps", type=int, default=0, help="subsample the 60 decisions (0 = all)")
    ap.add_argument("--scenario", choices=["table6", "disruption", "robot_pressure"], default="table6")
    ap.add_argument("--latency-source", choices=["platform", "legacy"], default="platform",
                    help="platform: platform-XML link latencies + one jitter draw per step, shared by all "
                         "policies; legacy: facade fallback used by compare_orchestrators.py")
    ap.add_argument("--congestion-factor", type=float, default=1.0,
                    help="multiplier on link latencies during phase IV (1.0 = legacy behaviour)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--quiet", action="store_true", help="silence orchestrator prints")
    args = ap.parse_args()

    seed_everything(args.seed)
    env = simpy.Environment()
    _, all_nodes = PlatformTopology(env, args.platform).build()
    services = ServiceDefinitionLoader(args.services_xml, simpy_env=None).load_services()
    TrainUtils.validate_locked_nodes(services, all_nodes)
    locked = {str(s.locked_node_id) for s in services if getattr(s, "locked_node_id", None) is not None}
    nodes = [n for n in all_nodes if getattr(n, "schedulable", True) or str(n.node_id) in locked]
    for svc in services:
        svc._baseline_rate = svc.execution_rate_hz
        for c in svc.containers:
            c._baseline_exec_time = c.exec_time

    latency = NetworkUtil.build_latency_matrix_with_jitter(
        args.platform, nodes, enable_jitter=not args.no_jitter, jitter_percent=args.jitter_percent,
        enable_packet_loss=False, packet_loss_rate=0.0, seed=args.seed)

    wifi = None
    if not args.no_wifi:
        wifi = WiFiMobilityController()
        wifi.register_access_point("edge-gpu1", 0.0, 0.0, 20.0)
        wifi.register_access_point("edge-gpu2", 30.0, 0.0, 20.0)
        wifi.register_robot("frank-2", 0.0, 0.0, args.robot_speed, 0.0)
        wifi.register_wifi_link("frank-2-wifi6", "frank-2")
    if args.latency_source == "platform":
        # The facade only accepts ndarray/list matrices; given the jitter wrapper it silently
        # falls back to a synthetic layer-distance matrix redrawn at every call (legacy path).
        jitter_rng = np.random.default_rng(args.seed)
        metrics = SimulatorMetricsFacade(nodes, latency_matrix=np.array(getattr(latency, "base_matrix", latency), dtype=float),
                                         wifi_controller=wifi)
    else:
        metrics = SimulatorMetricsFacade(nodes, latency_matrix=latency, wifi_controller=wifi)

    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    sink = io.StringIO()
    mute = (lambda: contextlib.redirect_stdout(sink)) if args.quiet else contextlib.nullcontext
    with mute():
        policies = build_policies(names, services, nodes, latency, metrics, args)

    initial_deployment = StaticInitialBaseline(services, nodes).optimize(services)
    # Services the XML marks exclude_from_orchestration (or DaemonSets) stay where they were
    # measured, as OrchestrationLoop._restore_non_optimizable_allocations does live.
    node_ids = {n.node_id for n in nodes}
    pinned = {}
    for svc in services:
        if getattr(svc, "exclude_from_orchestration", False) or getattr(svc, "is_daemonset", False):
            seen = [nid for nid in (getattr(svc, "instance_usage_by_node", {}) or {}) if nid in node_ids]
            pinned[svc.service_id] = seen[0] if seen else initial_deployment.get(svc.service_id)
    pinned = {k: v for k, v in pinned.items() if v}
    initial_deployment.update(pinned)
    for pol in policies.values():
        if isinstance(pol, NSGA2Orchestrator) and not hasattr(pol, "_placement_history"):
            pol._placement_history, pol._history_window = {}, 3  # BaseNSGA2 skips super().__init__

    base_latency_fn = metrics.get_latency_matrix  # platform matrix + current WiFi state

    # Node outage = node reported with zero capacity (NotReady) to every feasibility check,
    # in addition to unreachable links. down_now is updated at each decision step.
    down_now = set()
    zero = {"cpu": 0.0, "ram_gb": 0.0, "gpu_gb": 0.0}

    def _guard(fn, nid_of):
        def wrapped(*a, **k):
            return dict(zero) if nid_of(*a, **k) in down_now else fn(*a, **k)
        return wrapped

    for _name in ("get_node_capacity", "get_node_available"):
        setattr(metrics, _name, _guard(getattr(metrics, _name), lambda nid, *a, **k: nid))
    for n in nodes:
        nm = getattr(n, "metrics", None)
        for _name in ("get_capacity", "get_available"):
            if nm is not None and hasattr(nm, _name):
                setattr(nm, _name, _guard(getattr(nm, _name), lambda *a, _id=n.node_id, **k: _id))

    background = DynamicScenario()
    random.seed(sub_seed(args.seed, "background"))
    background.initialize_baseline_loads(nodes)
    step_minutes = args.step_seconds / 60.0

    results = {n: [] for n in names}
    current = {n: None for n in names}
    realloc = {n: 0 for n in names}
    battery = {n: defaultdict(float) for n in names}
    sim_t = 0.0
    phases = PHASES
    trace = load_trace(args.trace_dir, nodes) if args.trace_dir else None
    if trace is not None:
        phases = ["replay"] * len(trace)
    if args.num_steps and args.num_steps < len(PHASES):
        # Evenly spaced subsample that keeps every phase (scalability runs).
        phases = [PHASES[int(i * len(PHASES) / args.num_steps)] for i in range(args.num_steps)]
    battery_ids = {n.node_id for n in nodes if getattr(n, "battery_powered", False)}
    locked_ids = {s.service_id for s in services if getattr(s, "locked_node_id", None) is not None}
    for step, phase in enumerate(phases):
        if wifi is not None:
            wifi.update(sim_t)
            sim_t += args.step_seconds
        down = set()
        if args.latency_source == "platform":
            m = np.asarray(base_latency_fn(nodes), dtype=float)
            if trace is not None:
                for (a, b), v in trace[step].items():
                    m[a, b] = v
            elif not args.no_jitter:
                m = m * np.clip(1.0 + jitter_rng.normal(0.0, args.jitter_percent / 100.0, m.shape), 0.5, None)
            down, robot_factor = scenario_events(args.scenario, step)
            down_now.clear()
            down_now.update(down)
            idx = {n.node_id: i for i, n in enumerate(nodes)}
            if robot_factor != 1.0 and "frank-2" in idx:
                r = idx["frank-2"]
                m[r, :] *= robot_factor
                m[:, r] *= robot_factor
                m[r, r] /= robot_factor ** 2
            for nid in down:
                k = idx[nid]
                m[k, :] = UNREACHABLE_S
                m[:, k] = UNREACHABLE_S
                m[k, k] = 0.0
            latency = m.tolist()
            # Every policy and the evaluator see the same frozen matrix for this decision step.
            metrics.get_latency_matrix = lambda nodes=None, _m=m: _m.copy()
        elif wifi is not None:
            latency = metrics.get_latency_matrix(nodes)
        scenario_rng = random.Random(sub_seed(args.seed, step, "scenario"))
        step_latency = apply_phase(services, phase, args.congestion_factor, None, latency, rng=scenario_rng)
        for name, pol in policies.items():
            pol.latency_matrix = step_latency
            cur = current[name]
            failed = False
            random.seed(sub_seed(args.seed, step, name))
            np.random.seed(sub_seed(args.seed, step, name))
            t0 = time.perf_counter()
            try:
                with mute():
                    new = pol.optimize(services, current_assignment=cur, hysteresis_threshold=args.hysteresis)
                    if (args.hysteresis_mode == "uniform" and name == "NSGA-II" and cur is not None):
                        new = pol._apply_hysteresis(new, cur, args.hysteresis)
            except RuntimeError:
                new, failed = None, True
            dt = time.perf_counter() - t0
            if new:
                new = {**new, **pinned}
            if new and cur is not None and args.migration_budget >= 0:
                # Migration budget per cycle (R(x, x_prev) as a constraint, R1-8/R2-12):
                # a feasible running placement is kept when the proposal would move more
                # than k services; an infeasible one is always replaced.
                n_moves = sum(1 for sid in new if new.get(sid) != cur.get(sid))
                if n_moves > args.migration_budget:
                    with mute():
                        cur_ok = not OrchestratorUtils(services, nodes, step_latency).validate_placement(cur, log_warnings=False)
                    if cur_ok:
                        new = dict(cur)
            if not new:
                # The live loop keeps the running deployment when a decision fails; before the
                # first successful decision that is the initial deployment.
                new, failed = (cur or initial_deployment), True
            moved = sum(1 for sid in new if cur is not None and new.get(sid) != cur.get(sid))
            if moved:
                realloc[name] += 1
            current[name] = copy.deepcopy(new)
            random.seed(sub_seed(args.seed, step, "metrics"))  # same evaluation draws for every policy
            with mute():
                lat, energy, viol, bat, usage = TrainUtils.calculate_metrics(
                    new, services, nodes, step_latency, time_step_minutes=step_minutes,
                    scenario=background, step=step)
                strict = TrainUtils.strict_validate_placement(new, services, nodes, step_latency)
                # L_tot as defined in the paper (R2-3): end-to-end latency over blocking
                # dependencies only. The legacy metric above also walks non-blocking edges.
                random.seed(sub_seed(args.seed, step, "metrics"))
                lat_blocking = TrainUtils.calculate_metrics(
                    new, blocking_view(services), nodes, step_latency, time_step_minutes=step_minutes,
                    scenario=background, step=step)[0]
            per_service = latency_per_service(new, services, nodes, step_latency)
            l_tot = sum(per_service.values())          # paper L_tot(x) = sum_s L_s(x)
            l_max = max(per_service.values()) if per_service else 0.0  # min-max variant (R1-10)
            if strict:
                viol = strict
            if failed:
                viol = list(viol or []) + ["Inference failure"]
            batt_energy = sum(b["total_energy_j"] for b in bat.values())
            for nid, b in bat.items():
                battery[name][nid] += b["energy_ah"]
            results[name].append({
                "time": f"T={int(step * args.step_seconds)}s", "step": step, "scenario": phase,
                "latency": l_tot, "latency_max_service": l_max,
                "latency_e2e_path": lat_blocking, "latency_all_edges": lat, "energy": batt_energy, "service_energy": energy,
                "violations": len(viol or []), "violation_details": list(viol or [])[:5],
                "assignments": new, "battery_usage": bat, "reassigned": moved > 0,
                "services_moved": moved, "decision_time_s": dt,
                "services_on_robot": sum(1 for sid, nid in new.items()
                                         if nid in battery_ids and sid not in locked_ids),
            })

    out = {
        "config": {**vars(args), "duration_seconds": len(phases) * args.step_seconds,
                   "time_step_seconds": args.step_seconds, "scenarios": phases},
        "results": results, "reassignment_counts": realloc,
        "cumulative_battery_usage": {k: dict(v) for k, v in battery.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    for name in names:
        rs = results[name]
        print(f"{name:16s} lat={np.mean([r['latency'] for r in rs]):.4f}s "
              f"E={sum(r['energy'] for r in rs):.2f}J realloc={realloc[name]:2d} "
              f"viol={sum(r['violations'] for r in rs):3d} "
              f"t_dec={1000 * np.mean([r['decision_time_s'] for r in rs]):.1f}ms")


if __name__ == "__main__":
    main()
