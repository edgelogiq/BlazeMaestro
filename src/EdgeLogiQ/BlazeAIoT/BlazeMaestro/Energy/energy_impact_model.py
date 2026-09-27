from __future__ import annotations

from typing import Any, Callable, List, Optional, cast
import numpy as np

from .base_energy_estimator import BaseEnergyEstimator
from ..Metrics.base import BaseMetricsFacade
from ..Metrics.simulator_facade import SimulatorMetricsFacade


def _lazy_sklearn():
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore

        return LinearRegression
    except Exception:
        return None


class EnergyImpactModel(BaseEnergyEstimator):
    """Rule-based energy model with optional LinearRegression backend (orchestrator-scoped)."""

    def __init__(
        self,
        nodes: Optional[List[Any]] = None,
        metrics: Optional[BaseMetricsFacade] = None,
        use_ml: bool = False,
        node_battery_level: Optional[float] = None,
        node_battery_rate: Optional[float] = None,
        min_viable_energy: float = 0.0,
        battery_remaining_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        self._nodes: List[Any] = list(nodes or [])
        self.metrics: BaseMetricsFacade = metrics or SimulatorMetricsFacade(self._nodes)
        self.use_ml = use_ml
        self._LinearRegression = _lazy_sklearn()
        self._linreg: Any | None = None
        self._X_train: List[List[float]] = []
        self._y_train: List[float] = []
        self._battery_level = float(node_battery_level) if node_battery_level is not None else 0.0
        self._battery_rate = float(node_battery_rate) if node_battery_rate is not None else 0.0
        self._min_viable_energy = float(min_viable_energy)
        self._history: List[float] = [self._battery_level]
        self._battery_remaining_fn = battery_remaining_fn

    # ---------- Feature engineering ----------
    def _svc_demands(self, svc: Any) -> tuple[float, float, float, float]:
        cpu = float(sum(getattr(c, "cpu", 0) for c in getattr(svc, "containers", [])))
        ram = float(sum(getattr(c, "ram_gb", 0.0) for c in getattr(svc, "containers", [])))
        gpu = float(sum(getattr(c, "gpu_gb", 0.0) for c in getattr(svc, "containers", [])))
        exec_time = float(max((getattr(c, "exec_time", 0.0) for c in getattr(svc, "containers", [])), default=0.0))
        return cpu, ram, gpu, exec_time

    def _node_key(self, node: Any) -> str:
        node_id = getattr(node, "node_id", None)
        if isinstance(node_id, str) and node_id:
            return node_id
        try:
            idx = next(i for i, n in enumerate(self._nodes) if n is node)
            return str(idx)
        except StopIteration:
            return ""

    def _node_caps(self, node: Any) -> tuple[float, float, float, float, float]:
        node_id = self._node_key(node)
        try:
            caps = node.metrics.get_capacity()
        except Exception:
            caps = {}
        cpu_cap = float(caps.get("cpu", 0.0))
        ram_cap = float(caps.get("ram_gb", 0.0))
        gpu_cap = float(caps.get("gpu_gb", 0.0))
        try:
            gpu_shared_flag = bool(node.metrics.get_property("gpu_shared"))
        except Exception:
            gpu_shared_flag = False
        gpu_shared = 1.0 if bool(gpu_shared_flag or getattr(node, "gpu_shared", False)) else 0.0
        battery_remain = 0.0
        if self._battery_remaining_fn is not None and node_id:
            try:
                battery_remain = float(self._battery_remaining_fn(node_id))
            except Exception:
                battery_remain = 0.0
        return cpu_cap, ram_cap, gpu_cap, gpu_shared, battery_remain

    def _features(self, svc: Any, node: Any) -> List[float]:
        cpu, ram, gpu, exec_time = self._svc_demands(svc)
        cpu_cap, ram_cap, gpu_cap, gpu_shared, batt = self._node_caps(node)
        return [cpu, ram, gpu, exec_time, cpu_cap, ram_cap, gpu_cap, gpu_shared, batt]

    # ---------- Estimation ----------
    def estimate_service_energy(self, service: Any, node: Any) -> float:
        if self.use_ml and self._linreg is not None and self._X_train:
            return self.estimate_service_energy_ml(service, node)
        cpu, _, gpu, exec_time = self._svc_demands(service)
        try:
            caps = node.metrics.get_capacity()
        except Exception:
            caps = {}
        has_gpu = float(caps.get("gpu_gb", 0.0)) > 0.0
        factor = 0.7 if has_gpu and gpu > 0.0 else 1.0
        return float(cpu) * float(exec_time) * factor

    def estimate_service_energy_ml(self, service: Any, node: Any) -> float:
        if self._linreg is None:
            return self.estimate_service_energy(service, node)
        model = cast(Any, self._linreg)
        X = np.array(self._features(service, node), dtype=np.float32).reshape(1, -1)
        try:
            return float(model.predict(X)[0])
        except Exception:
            return self.estimate_service_energy(service, node)

    # ---------- Training ----------
    def add_training_sample(self, service: Any, node: Any, observed_energy: float) -> None:
        self._X_train.append(self._features(service, node))
        self._y_train.append(float(observed_energy))

    def fit_ml_model(self) -> None:
        if not self.use_ml:
            return
        if self._LinearRegression is None:
            self._linreg = None
            return
        if len(self._X_train) < 5:
            self._linreg = None
            return
        LinearRegression = self._LinearRegression
        X = np.array(self._X_train, dtype=np.float32)
        y = np.array(self._y_train, dtype=np.float32)
        try:
            model = LinearRegression()
            model.fit(X, y)
            self._linreg = model
        except Exception:
            self._linreg = None

    # ---------- BaseEnergyEstimator abstract API ----------
    def apply_service(self, service: Any, node: Any):
        energy_used = self.estimate_service_energy(service, node)
        if self._battery_rate > 0.0:
            self._battery_level -= float(energy_used) * self._battery_rate
            self._history.append(self._battery_level)
        return None

    def get_battery_level(self):
        return float(self._battery_level)

    def get_battery_history(self):
        return list(self._history)

    def reset(self, battery_level=None):
        if battery_level is not None:
            self._battery_level = float(battery_level)
        self._history = [self._battery_level]
