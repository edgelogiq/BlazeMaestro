"""Reference placement policies for the revision campaign (reviewer comment R2-13).

All baselines expose the same ``optimize(services, current_assignment,
hysteresis_threshold)`` interface as the four BlazeMaestro orchestrators so the
campaign harness can drive them unchanged. None of them is latency- or
energy-aware; their purpose is to quantify what dynamic orchestration adds.

- StaticInitialBaseline: placement computed once at t=0 by a K8s-like fit, never revised.
- OneShotBaseline: placement optimized once at t=0 by a wrapped orchestrator, then frozen.
- KubeDefaultBaseline: emulation of the default kube-scheduler (NodeResourcesFit filter
  on CPU/RAM/GPU requests + LeastAllocated scoring), re-run every cycle.
- LayerPinnedBaseline: every allocable service pinned to one layer (edge or fog),
  spread over that layer's schedulable nodes with LeastAllocated scoring.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence


def _requests(svc: Any) -> tuple:
    """Resource *requests* as Kubernetes sees them (no period/exec-time awareness)."""
    cpu = sum(float(getattr(c, "cpu", 0.0) or 0.0) for c in svc.containers)
    ram = sum(float(getattr(c, "ram_gb", 0.0) or 0.0) for c in svc.containers)
    gpu = sum(float(getattr(c, "gpu_gb", 0.0) or 0.0) for c in svc.containers)
    return cpu, ram, gpu


def _orchestrated(services: Sequence[Any]) -> list:
    return [s for s in services
            if not (getattr(s, "is_daemonset", False) or getattr(s, "exclude_from_orchestration", False))]


def _requires_gpu(svc: Any, gpu_nodes: set) -> bool:
    """A pod whose only measured instances ran on GPU nodes would carry a GPU request
    (nvidia.com/gpu) in its manifest; the default scheduler honours resource requests."""
    seen = set(getattr(svc, "instance_usage_by_node", {}) or {})
    return bool(seen) and seen <= gpu_nodes


class _FitScheduler:
    """Shared filter/score loop for the K8s-like baselines."""

    name = "fit"

    def __init__(self, services: Sequence[Any], nodes: Sequence[Any], layer: Optional[str] = None):
        self.services = _orchestrated(services)
        self.nodes = [n for n in nodes if getattr(n, "schedulable", True)]
        self.all_nodes = list(nodes)
        self.layer = layer
        self.latency_matrix = None  # interface compatibility with the harness

    def _candidates(self):
        if self.layer is None:
            return self.nodes
        return [n for n in self.nodes if getattr(n, "layer_id", None) == self.layer]

    def _place(self, services: Sequence[Any]) -> Dict[str, str]:
        free = {n.node_id: [float(n.cpu), float(n.ram), float(n.gpu.capacity if n.gpu else 0.0)]
                for n in self.all_nodes}
        caps = {n.node_id: list(v) for n, v in zip(self.all_nodes, free.values())}
        gpu_nodes = {n.node_id for n in self.all_nodes if n.gpu and n.gpu.capacity > 0}
        placement: Dict[str, str] = {}
        pending = []
        # Same service set the orchestrators receive in optimize(services).
        # Locked services first: they consume capacity wherever they are pinned.
        for svc in services:
            locked = getattr(svc, "locked_node_id", None)
            if locked is not None:
                placement[svc.service_id] = str(locked)
                cpu, ram, gpu = _requests(svc)
                if str(locked) in free:
                    f = free[str(locked)]
                    f[0] -= cpu; f[1] -= ram; f[2] -= gpu
            else:
                pending.append(svc)
        candidates = self._candidates()
        for svc in pending:
            cpu, ram, gpu = _requests(svc)
            locked_layer = getattr(svc, "locked_layer_id", None)
            pool = [n for n in candidates if locked_layer is None or getattr(n, "layer_id", None) == locked_layer]
            if _requires_gpu(svc, gpu_nodes):
                pool = [n for n in pool if n.node_id in gpu_nodes] or pool
            fits = [n for n in pool
                    if free[n.node_id][0] >= cpu and free[n.node_id][1] >= ram and free[n.node_id][2] >= gpu]
            if not fits:
                # Nothing fits: K8s would leave the pod Pending; we place it on the least
                # allocated node of the pool so the violation is counted, not hidden.
                fits = pool or candidates

            def score(n):
                f, c = free[n.node_id], caps[n.node_id]
                cpu_frac = f[0] / c[0] if c[0] > 0 else 0.0
                ram_frac = f[1] / c[1] if c[1] > 0 else 0.0
                return (cpu_frac + ram_frac) / 2.0

            best = max(fits, key=score)
            placement[svc.service_id] = best.node_id
            f = free[best.node_id]
            f[0] -= cpu; f[1] -= ram; f[2] -= gpu
        return placement


class KubeDefaultBaseline(_FitScheduler):
    name = "K8s-default"

    def optimize(self, services=None, current_assignment=None, hysteresis_threshold=0.0):
        proposal = self._place(services if services is not None else self.services)
        # The default scheduler never moves running pods; only unscheduled ones are placed.
        if current_assignment:
            return {sid: current_assignment.get(sid, nid) for sid, nid in proposal.items()}
        return proposal


class LayerPinnedBaseline(_FitScheduler):
    def __init__(self, services, nodes, layer: str):
        super().__init__(services, nodes, layer=layer)
        self.name = f"All-{layer}"

    def optimize(self, services=None, current_assignment=None, hysteresis_threshold=0.0):
        if current_assignment:
            return copy.deepcopy(current_assignment)
        return self._place(services if services is not None else self.services)


class StaticInitialBaseline(_FitScheduler):
    name = "Static-initial"

    def optimize(self, services=None, current_assignment=None, hysteresis_threshold=0.0):
        if current_assignment:
            return copy.deepcopy(current_assignment)
        return self._place(services if services is not None else self.services)


class OneShotBaseline:
    """Run a real orchestrator once at t=0, then keep its placement forever."""

    def __init__(self, inner: Any, name: str):
        self.inner = inner
        self.name = name

    @property
    def latency_matrix(self):
        return getattr(self.inner, "latency_matrix", None)

    @latency_matrix.setter
    def latency_matrix(self, value):
        self.inner.latency_matrix = value

    def optimize(self, services=None, current_assignment=None, hysteresis_threshold=0.0):
        if current_assignment:
            return copy.deepcopy(current_assignment)
        return self.inner.optimize(services, current_assignment=None, hysteresis_threshold=0.0)


class KinitosInspiredBaseline(_FitScheduler):
    """Network-aware scheduler inspired by Kinitos' NetworkOverhead plugin (Tsokov et al.,
    JNCA 2025). This is a re-implementation of the published rule, not the original code.

    Placement: among nodes with enough free resources (requests-based filter, GPU honoured),
    choose the one minimising the network cost to the service's already-placed dependencies.
    Migration: a running service is moved only if the dependencies violating their maximum
    latency outnumber those that satisfy it, and a strictly better node exists. The maximum
    latency of a dependency is derived from the service budget T_period - t_exec.
    """

    name = "Kinitos-inspired"

    def __init__(self, services, nodes):
        super().__init__(services, nodes)
        self._index = {n.node_id: i for i, n in enumerate(self.all_nodes)}

    def _lat(self, a, b):
        if a == b:
            return 0.0
        m = self.latency_matrix
        try:
            return float(m[self._index[a]][self._index[b]])
        except Exception:
            return 0.0

    def _budget(self, svc, nid):
        from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.orchestrator_utils import OrchestratorUtils
        _, _, _, exec_time, period = OrchestratorUtils.calculate_service_demands(svc, nid)
        if period == float("inf") or period <= 0:
            return float("inf")
        return max(0.0, period - (exec_time or 0.0))

    @staticmethod
    def _deps(svc):
        return [d.service_id for d in (svc.dependencies or []) if getattr(d, "blocking", True)]

    def _cost(self, svc, nid, placement):
        return sum(self._lat(placement[d], nid) for d in self._deps(svc) if d in placement)

    def _free(self, placement, services):
        free = {n.node_id: [float(n.cpu), float(n.ram), float(n.gpu.capacity if n.gpu else 0.0)]
                for n in self.all_nodes}
        for svc in services:
            nid = placement.get(svc.service_id)
            if nid in free:
                cpu, ram, gpu = _requests(svc)
                f = free[nid]; f[0] -= cpu; f[1] -= ram; f[2] -= gpu
        return free

    def _candidates_for(self, svc, free, gpu_nodes):
        cpu, ram, gpu = _requests(svc)
        pool = [n for n in self.nodes if free[n.node_id][0] >= cpu and free[n.node_id][1] >= ram
                and free[n.node_id][2] >= gpu]
        if _requires_gpu(svc, gpu_nodes):
            pool = [n for n in pool if n.node_id in gpu_nodes]
        return pool

    def optimize(self, services=None, current_assignment=None, hysteresis_threshold=0.0):
        services = list(services if services is not None else self.services)
        gpu_nodes = {n.node_id for n in self.all_nodes if n.gpu and n.gpu.capacity > 0}
        by_id = {s.service_id: s for s in services}
        if not current_assignment:
            placement = {}
            pending = []
            for svc in services:
                lock = getattr(svc, "locked_node_id", None)
                if lock is not None:
                    placement[svc.service_id] = str(lock)
                else:
                    pending.append(svc)
            # Dependencies first, so the network cost to them is known.
            done, order = set(placement), []
            while pending:
                ready = [s for s in pending if all(d in done or d not in by_id for d in self._deps(s))] or pending[:1]
                for s in ready:
                    order.append(s); done.add(s.service_id); pending.remove(s)
            for svc in order:
                free = self._free(placement, [by_id[k] for k in placement if k in by_id])
                pool = self._candidates_for(svc, free, gpu_nodes) or self.nodes
                best = min(pool, key=lambda n: (self._cost(svc, n.node_id, placement), -free[n.node_id][0]))
                placement[svc.service_id] = best.node_id
            return placement

        placement = dict(current_assignment)
        for svc in services:
            sid = svc.service_id
            if getattr(svc, "locked_node_id", None) is not None or sid not in placement:
                continue
            cur = placement[sid]
            budget = self._budget(svc, cur)
            deps = [d for d in self._deps(svc) if d in placement]
            if not deps:
                continue
            violating = sum(1 for d in deps if self._lat(placement[d], cur) > budget)
            if violating <= len(deps) - violating:
                continue
            others = [by_id[k] for k in placement if k in by_id and k != sid]
            free = self._free({k: v for k, v in placement.items() if k != sid}, others)
            pool = [n for n in self._candidates_for(svc, free, gpu_nodes) if n.node_id != cur]
            if not pool:
                continue
            best = min(pool, key=lambda n: self._cost(svc, n.node_id, placement))
            if self._cost(svc, best.node_id, placement) < self._cost(svc, cur, placement):
                placement[sid] = best.node_id
        return placement
