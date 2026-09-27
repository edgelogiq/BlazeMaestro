"""Topology utilities: helpers for applying loads to simulated nodes.

Provides a unified API that works with nodes implementing either:
- SimPyResourceMixin (node.acquire_cpu/ram with cpu_container/ram_container), or
- direct simpy resources/containers (node.cpu as a Resource, node.ram/gpu as Containers).
"""

from __future__ import annotations
import simpy
from typing import Any, cast


class TopologyUtil:
    @staticmethod
    def add_background_load(
        env: simpy.Environment,
        node,
        *,
        cpu_frac: float,
        ram_frac: float,
        gpu_frac: float,
        duration: float,
    ) -> None:
        """Apply background CPU, RAM, and GPU load to a node for a given duration.

        This function detects available node interfaces and adapts:
        - If node exposes acquire_cpu/acquire_ram and cpu_container/ram_container, use them.
        - Else, use simpy Resource/Container primitives on node.cpu/node.ram/node.gpu.
        """
        # Clamp fractions
        cpu_frac = max(0.0, min(1.0, float(cpu_frac)))
        ram_frac = max(0.0, min(1.0, float(ram_frac)))
        gpu_frac = max(0.0, min(1.0, float(gpu_frac)))

        # Prefer SimPyResourceMixin interface when available
        acquire_cpu = getattr(node, "acquire_cpu", None)
        acquire_ram = getattr(node, "acquire_ram", None)
        cpu_container = getattr(node, "cpu_container", None)
        ram_container = getattr(node, "ram_container", None)

        if callable(acquire_cpu) and callable(acquire_ram) and cpu_container is not None and ram_container is not None:
            try:
                cpu_capacity = float(getattr(cpu_container, "capacity", 0.0))
                ram_capacity = float(getattr(ram_container, "capacity", 0.0))
                cpu_to_get = float(max(0.0, cpu_frac * cpu_capacity))
                ram_to_get = float(max(0.0, ram_frac * ram_capacity))
                if cpu_to_get > 0:
                    cpu_proc = cast(Any, acquire_cpu(cpu_to_get, duration))
                    env.process(cpu_proc)
                if ram_to_get > 0:
                    ram_proc = cast(Any, acquire_ram(ram_to_get, duration))
                    env.process(ram_proc)
            except Exception:
                # Fall back to generic approach below if anything goes wrong
                TopologyUtil._fallback_load(env, node, cpu_frac, ram_frac, gpu_frac, duration)
        else:
            TopologyUtil._fallback_load(env, node, cpu_frac, ram_frac, gpu_frac, duration)

        # GPU handling (container-like)
        gpu = getattr(node, "gpu", None)
        if gpu is not None:
            try:
                gpu_cap = float(getattr(gpu, "capacity", 0.0))
                gpu_to_get = float(max(0.0, gpu_frac * gpu_cap))
                if gpu_to_get > 0:
                    def _gpu_reserver(amount: float):
                        if amount <= 0:
                            return
                        yield gpu.get(amount)
                        yield env.timeout(duration)
                        yield gpu.put(amount)
                    env.process(_gpu_reserver(gpu_to_get))
            except Exception:
                pass

    @staticmethod
    def _fallback_load(env: simpy.Environment, node, cpu_frac: float, ram_frac: float, gpu_frac: float, duration: float) -> None:
        """Fallback background load using simpy primitives on node.cpu/node.ram."""
        # CPU: treat node.cpu as a simpy.Resource with 'capacity' attribute (if provided)
        cpu_res = getattr(node, "cpu", None)
        cpu_capacity = float(getattr(cpu_res, "capacity", 0.0)) if cpu_res is not None else 0.0
        cpu_slots = int(round(max(0.0, cpu_frac) * cpu_capacity)) if cpu_capacity > 0 else 0

        if cpu_res is not None and hasattr(cpu_res, "request") and cpu_slots > 0:
            def _cpu_holder():
                with cpu_res.request() as req:
                    yield req
                    yield env.timeout(duration)
            for _ in range(cpu_slots):
                env.process(_cpu_holder())

        # RAM: treat node.ram as a simpy.Container
        ram_cont = getattr(node, "ram", None)
        ram_capacity = float(getattr(ram_cont, "capacity", 0.0)) if ram_cont is not None else 0.0
        ram_to_get = float(max(0.0, ram_frac * ram_capacity)) if ram_capacity > 0 else 0.0

        if ram_cont is not None and ram_to_get > 0:
            def _ram_reserver(amount: float):
                if amount <= 0:
                    return
                yield ram_cont.get(amount)
                yield env.timeout(duration)
                yield ram_cont.put(amount)
            env.process(_ram_reserver(ram_to_get))
