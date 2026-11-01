"""Discrete-event execution of a placement with SimPy over one control period.

Every placed service releases a job each period T_s (random initial phase). A job waits
for its upstream data (L_s: link latency + transfer time over blocking dependencies),
then acquires its CPU cores and, if any, its GPU memory on the node (SimPy Containers,
so co-located jobs contend and queue), runs for its execution time on that node and
releases them. The node's background load permanently holds its share of the cores.
A job on an unreachable node never completes.

Returned per service: jobs released, jobs completed, deadline misses (response time
above T_s * (1 - margin), or not completed) and mean queueing wait; per battery node,
the service energy computed with the evaluator's power model applied to the simulated
busy time instead of t_exec * rate * duration.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Sequence

import simpy

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.orchestrator_utils import OrchestratorUtils


def _power(node, cpu_needed: float, gpu_needed: float) -> float:
    base = getattr(node, "power_jetson_base", getattr(node, "power_base", 5.0))
    active = getattr(node, "power_jetson_active", getattr(node, "power_active", 30.0))
    gpu_cap = node.gpu.capacity if node.gpu else 0.0
    util = cpu_needed / max(node.cpu, 1.0) + (gpu_needed / max(gpu_cap, 1.0) if gpu_cap > 0 else 0.0)
    return base + (active - base) * min(1.0, util)


def simulate_step(placement: Dict[str, str], services: Sequence[Any], nodes: Sequence[Any],
                  l_s: Dict[str, float], background_cpu: Dict[str, float], horizon_s: float,
                  rng: random.Random, down: Iterable[str] = (), margin: float = 0.0) -> Dict[str, Any]:
    env = simpy.Environment()
    node_by_id = {n.node_id: n for n in nodes}
    by_id = {s.service_id: s for s in services}
    down = set(down)

    cpu, gpu, free = {}, {}, {}
    for n in nodes:
        free[n.node_id] = max(0.0, float(n.cpu) - float(background_cpu.get(n.node_id, 0.0)))
        cpu[n.node_id] = simpy.Container(env, capacity=max(float(n.cpu), 1e-9), init=free[n.node_id])
        g = float(n.gpu.capacity) if n.gpu else 0.0
        gpu[n.node_id] = simpy.Container(env, capacity=max(g, 1e-9), init=g)

    stats = {}

    def job(st, nid, cores, gpu_gb, exec_time, dep_lat, period):
        t0 = env.now
        yield env.timeout(dep_lat)
        ready = env.now
        yield cpu[nid].get(cores)
        if gpu_gb > 0:
            yield gpu[nid].get(gpu_gb)
        start = env.now
        yield env.timeout(exec_time)
        st["busy"] += max(0.0, min(env.now, horizon_s) - min(start, horizon_s))
        yield cpu[nid].put(cores)
        if gpu_gb > 0:
            yield gpu[nid].put(gpu_gb)
        st["done"] += 1
        st["wait"] += start - ready
        if env.now - t0 > period * (1.0 - margin):
            st["late"] += 1

    def releaser(sid, nid):
        svc, node = by_id[sid], node_by_id[nid]
        _, _, _, exec_time, period = OrchestratorUtils.calculate_service_demands(svc, nid)
        cores = sum(c.cpu for c in svc.containers)
        gpu_gb = sum(c.gpu_gb for c in svc.containers)
        st = stats[sid] = {"jobs": 0, "done": 0, "late": 0, "wait": 0.0, "busy": 0.0,
                           "power": _power(node, cores, gpu_gb), "node": nid}
        if not (0 < period < float("inf")):
            return
        cores = min(cores, max(free[nid], 1e-3))  # a job never asks more than the free cores
        gpu_gb = min(gpu_gb, gpu[nid].capacity) if gpu_gb > 0 else 0.0
        yield env.timeout(rng.uniform(0.0, period))
        while env.now < horizon_s:
            st["jobs"] += 1
            if nid not in down:
                env.process(job(st, nid, cores, gpu_gb, exec_time, l_s.get(sid, 0.0), period))
            yield env.timeout(period)

    for sid, nid in placement.items():
        if sid in by_id and nid in node_by_id:
            env.process(releaser(sid, nid))
    env.run(until=horizon_s * 3)  # let the jobs released before the horizon finish

    per_service, energy = {}, {}
    for sid, st in stats.items():
        misses = st["late"] + (st["jobs"] - st["done"])
        per_service[sid] = {"jobs": st["jobs"], "done": st["done"], "misses": misses,
                            "wait_mean": st["wait"] / st["done"] if st["done"] else 0.0}
        node = node_by_id[st["node"]]
        if getattr(node, "battery_powered", False):
            energy[st["node"]] = energy.get(st["node"], 0.0) + st["power"] * st["busy"]
    return {"per_service": per_service, "service_energy_j": energy}
