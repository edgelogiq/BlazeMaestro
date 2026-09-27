from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Callable, cast

import numpy as np

from .base_energy_estimator import BaseEnergyEstimator
from ..Metrics.base import BaseMetricsFacade
from ..Metrics.simulator_facade import SimulatorMetricsFacade


def _lazy_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        return torch, nn, optim
    except Exception:
        return None, None, None


@dataclass
class UEESettings:
    latent_dim: int = 4
    epochs: int = 50
    lr: float = 1e-3
    device: str = "auto"  # "cpu", "cuda", or "auto"


class UnsupervisedEnergyEstimator(BaseEnergyEstimator):
    def save_checkpoint(self, filepath: str) -> None:
        if self._torch is None or self._model is None or not self._trained:
            raise RuntimeError("Model is not trained or torch is unavailable.")
        torch = cast(Any, self._torch)
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'settings': self.settings,
            'feat_names': self._feat_names,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        if self._torch is None:
            raise RuntimeError("Torch is unavailable.")
        torch = cast(Any, self._torch)
        checkpoint = torch.load(filepath, map_location=self._device, weights_only=False)
        
        # Restore settings and feature names BEFORE rebuilding model
        self.settings = checkpoint.get('settings', self.settings)
        self._feat_names = checkpoint.get('feat_names', self._feat_names)
        
        # Rebuild the model with correct input dimension from loaded feature names
        input_dim = len(self._feat_names)
        if self._model is None or self._model is None:
            # Need to rebuild model with loaded dimensions
            self._init_torch()  # Will rebuild model
        
        # Now load the state dict
        if self._model is not None:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise RuntimeError("Failed to build model for checkpoint loading.")
        
        self._trained = True
    """Unsupervised energy estimator (autoencoder) adapted for simulator Service/Node.

    - Features combine service demands and node capacities (+ optional battery).
    - Torch is imported lazily; without torch, falls back to a heuristic.
    - API: add_training_sample(service, node) -> fit() -> estimate_service_energy(service, node)
    """

    def __init__(
        self,
        nodes: List[Any],
        metrics: Optional[BaseMetricsFacade] = None,
        battery_remaining_fn: Optional[Callable[[str], float]] = None,  # fn(node_id) -> remaining Joules (optional)
        settings: UEESettings | None = None,
    ) -> None:
        self._nodes: List[Any] = list(nodes)
        self.metrics: BaseMetricsFacade = metrics or SimulatorMetricsFacade(self._nodes)
        self.battery_remaining_fn = battery_remaining_fn
        self.settings = settings or UEESettings()
        self._torch, self._nn, self._optim = _lazy_torch()
        self._model: Any | None = None
        self._device: Any | None = None
        self._trained = False
        self._X: List[List[float]] = []

        # Define feature schema
        self._feat_names = [
            "svc_cpu",
            "svc_ram",
            "svc_gpu",
            "svc_exec_time",
            "svc_rate",
            "node_cpu_cap",
            "node_ram_cap",
            "node_gpu_cap",
            "gpu_shared",
            "battery_remain",
            "est_exec_time",
        ]

        if self._torch is not None:
            self._init_torch()

    # ---------- Torch setup ----------
    def _init_torch(self):
        assert self._torch is not None and self._nn is not None
        torch = cast(Any, self._torch)
        nn = cast(Any, self._nn)
        dev = self.settings.device
        if dev == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(dev)

        input_dim = len(self._feat_names)

        class AE(nn.Module):
            def __init__(self, in_dim: int, latent: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, 16), nn.ReLU(),
                    nn.Linear(16, latent), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent, 16), nn.ReLU(),
                    nn.Linear(16, in_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                x_hat = self.decoder(z)
                return x_hat, z

        self._model = AE(input_dim, self.settings.latent_dim).to(self._device)

    # ---------- Features ----------
    def _svc_demands(self, svc: Any) -> tuple[float, float, float, float, float]:
        cpu = float(sum(getattr(c, "cpu", 0) for c in getattr(svc, "containers", [])))
        ram = float(sum(getattr(c, "ram_gb", 0.0) for c in getattr(svc, "containers", [])))
        gpu = float(sum(getattr(c, "gpu_gb", 0.0) for c in getattr(svc, "containers", [])))
        exec_time = float(max(((getattr(c, "exec_time", 0.0) or 0.0) for c in getattr(svc, "containers", [])), default=0.0))
        period = float(getattr(svc, "period", float('inf')))
        rate = 1.0 / period if period > 0 and period < float('inf') else 0.0
        return cpu, ram, gpu, exec_time, rate

    def _node_caps(self, node: Any) -> tuple[float, float, float, float, float]:
        node_id = self._node_key(node)
        caps = node.metrics.get_capacity()
        cpu_cap = float(caps.get("cpu", 0.0))
        ram_cap = float(caps.get("ram_gb", 0.0))
        gpu_cap = float(caps.get("gpu_gb", 0.0))
        gpu_shared = 1.0 if bool(node.metrics.get_property("gpu_shared") or getattr(node, "gpu_shared", False)) else 0.0
        battery_remain = 0.0
        if self.battery_remaining_fn is not None and node_id:
            try:
                battery_remain = float(self.battery_remaining_fn(node_id))
            except Exception:
                battery_remain = 0.0
        return cpu_cap, ram_cap, gpu_cap, gpu_shared, battery_remain

    def _features(self, svc: Any, node: Any) -> List[float]:
        cpu, ram, gpu, exec_time, rate = self._svc_demands(svc)
        cpu_cap, ram_cap, gpu_cap, gpu_shared, batt = self._node_caps(node)
        
        # Estimate execution time on this node
        # Assume max_cpu is roughly 8.0 (typical high-end node) for normalization
        # Ideally we'd know the cluster max, but for local estimation we use relative capacity
        if cpu_cap > 0:
            # Simple linear scaling: faster CPU = faster execution
            # Normalize against a "standard" 1.0 CPU core
            speed_factor = cpu_cap / 1.0 
            est_exec_time = exec_time / max(0.1, speed_factor)
        else:
            est_exec_time = exec_time * 10.0 # Very slow if no CPU
            
        return [cpu, ram, gpu, exec_time, rate, cpu_cap, ram_cap, gpu_cap, gpu_shared, batt, est_exec_time]

    # ---------- Public API ----------
    def add_training_sample(self, service: Any, node: Any) -> None:
        self._X.append(self._features(service, node))

    def fit_unsupervised_model(self) -> None:
        if self._torch is None:
            # No torch installed
            self._trained = False
            return
        if len(self._X) < 10:
            # Not enough samples to learn meaningful reconstruction
            self._trained = False
            return
        assert self._torch is not None and self._nn is not None and self._optim is not None and self._model is not None
        torch = cast(Any, self._torch)
        nn = cast(Any, self._nn)
        optim = cast(Any, self._optim)
        X = torch.tensor(np.array(self._X, dtype=np.float32), dtype=torch.float32).to(self._device)
        opt = optim.Adam(self._model.parameters(), lr=self.settings.lr)
        loss_fn = nn.MSELoss()
        self._model.train()
        for _ in range(self.settings.epochs):
            opt.zero_grad()
            x_hat, _ = self._model(X)
            loss = loss_fn(x_hat, X)
            loss.backward()
            opt.step()
        self._trained = True

    def estimate_service_energy(self, service: Any, node: Any) -> float:
        # If trained, use reconstruction error as energy proxy; else fallback heuristic
        if self._torch is not None and self._model is not None and self._trained:
            torch = cast(Any, self._torch)
            x = torch.tensor(np.array(self._features(service, node), dtype=np.float32), dtype=torch.float32).unsqueeze(0).to(self._device)
            self._model.eval()
            with torch.no_grad():
                x_hat, _ = self._model(x)
                err = torch.mean((x - x_hat) ** 2).item()
            return float(err)
        # Heuristic baseline: proportional to cpu * exec_time with GPU discount
        cpu, _, gpu, exec_time = self._svc_demands(service)
        caps = node.metrics.get_capacity()
        has_gpu = float(caps.get("gpu_gb", 0.0)) > 0.0
        factor = 0.7 if has_gpu and gpu > 0.0 else 1.0
        return float(cpu) * float(exec_time) * factor

    # ---------- BaseEnergyEstimator abstract API ----------
    def apply_service(self, service: Any, node: Any):
        # This estimator is stateless during estimation; optionally could log usage.
        _ = self.estimate_service_energy(service, node)
        return None

    def get_battery_level(self):
        # No internal battery model; return 0 to satisfy interface.
        return 0.0

    def get_battery_history(self):
        # No internal history maintained.
        return []

    def reset(self, battery_level=None):
        # Clear collected samples and training flag; keep model to avoid retraining if desired.
        self._X.clear()
        self._trained = False

    # ---------- Helpers ----------
    def _node_key(self, node: Any) -> str:
        node_id = getattr(node, "node_id", None)
        if isinstance(node_id, str) and node_id:
            return node_id
        # fallback to index by identity
        try:
            idx = next(i for i, n in enumerate(self._nodes) if n is node)
            return str(idx)
        except StopIteration:
            return ""
