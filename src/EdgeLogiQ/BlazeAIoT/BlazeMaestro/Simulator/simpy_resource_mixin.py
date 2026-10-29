"""
SimPyResourceMixin: Provides simpy.Container-based resource management for CPU and RAM.
Uniform interface for resource acquisition/release in SimPy-based nodes.
"""
import simpy


class SimPyResourceMixin:
    def _init_simpy_resources(self, env, cpu_capacity, ram_capacity):
        self.env = env
        self.cpu_container = simpy.Container(env, capacity=float(cpu_capacity), init=float(cpu_capacity))
        self.ram_container = simpy.Container(env, capacity=float(ram_capacity), init=float(ram_capacity))

    def acquire_cpu(self, amount, duration):
        if amount <= 0:
            return
        yield self.cpu_container.get(amount)
        yield self.env.timeout(duration)
        yield self.cpu_container.put(amount)

    def acquire_ram(self, amount, duration):
        if amount <= 0:
            return
        yield self.ram_container.get(amount)
        yield self.env.timeout(duration)
        yield self.ram_container.put(amount)

