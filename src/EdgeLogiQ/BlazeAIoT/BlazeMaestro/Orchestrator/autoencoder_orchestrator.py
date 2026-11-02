from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import random
import os
import numpy as np

from ..Metrics.base import BaseMetricsFacade
from ..Metrics.simulator_facade import SimulatorMetricsFacade
from .base_orchestrator import BaseOrchestrator
from .autoencoder_helpers import (
    OrchestratorWeights,
    build_autoencoder_model,
    build_state_features,
    lazy_torch,
    train_single_batch,
)


class AutoEncoderOrchestrator(BaseOrchestrator):
    """
    Autoencoder-based unsupervised orchestrator that learns placement patterns.
    
    Unlike DQN (reinforcement learning), this uses an autoencoder to:
    1. Encode state (services, nodes, resources) into a latent representation
    2. Decode latent representation into placement probabilities
    3. Learn through unsupervised reconstruction of good placements
    
    The autoencoder learns patterns of valid/optimal placements by minimizing
    reconstruction error weighted by placement quality.
    """

    def __init__(
        self,
        services: Sequence[Any],
        nodes: Sequence[Any],
        latency_matrix: Optional[Any] = None,
        *,
        energy_estimator=None,
        weights: Optional[OrchestratorWeights] = None,
        metrics: Optional[BaseMetricsFacade] = None,
        random_seed: Optional[int] = None,
        verbose: bool = False,
        epochs: int = 300,
        latent_dim: int = 32,
        lr: float = 6e-4,
        batch_size: int = 192,  # Larger batch to use more GPU and stabilize loss
        temperature: float = 0.5,
    ) -> None:
        metrics_facade = metrics or SimulatorMetricsFacade(nodes)
        
        # Initialize base class
        super().__init__(services, nodes, metrics_facade)
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.energy_estimator = energy_estimator
        self.weights = weights or OrchestratorWeights()
        self.verbose = verbose
        self.debug = False
        
        # Autoencoder hyperparameters
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        self.temperature = temperature
        self.backtrack_budget = 5000  # max placement trials per decision (bounds decision time)
        
        # Map node_id -> index for quick lookups
        self._node_index: Dict[str, int] = {str(getattr(n, "node_id", i)): i for i, n in enumerate(self.nodes)}
        
        # Latency matrix (use provided or get from metrics).
        # Unwrap LatencyMatrixWithJitter wrappers so downstream code always
        # gets a plain numpy float32 array.
        _lm = latency_matrix if latency_matrix is not None else self.metrics.get_latency_matrix(self.nodes)
        if hasattr(_lm, 'base_matrix'):
            _lm = _lm.base_matrix
        self.latency_matrix = np.asarray(_lm, dtype=np.float32) if _lm is not None else None
        
        # Torch components
        self._torch, self._nn, self._optim, self._device = lazy_torch()
        self._model: Any = None
        self._optimizer: Any = None
        self._scheduler: Any = None
        self.use_amp = bool(self._torch and self._device is not None and getattr(self._device, "type", "cpu") == "cuda")
        self._scaler = self._torch.amp.GradScaler("cuda") if self.use_amp else None
        if self.use_amp:
            try:
                self._torch.set_float32_matmul_precision("medium")
            except Exception:
                pass
        
        self._training_losses: List[float] = []

    def save_checkpoint(self, filepath: str) -> None:
        """Save trained autoencoder model."""
        if self._torch is None:
            raise ImportError("PyTorch is required")
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        # Get current input dimension from the model's first layer
        input_dim = self._model.encoder[0].in_features
        
        torch = self._torch
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'latent_dim': self.latent_dim,
            'input_dim': input_dim, # Save input dim explicitly
            'num_services': len(self.services),
            'num_nodes': len(self.nodes),
            'training_losses': self._training_losses,
        }, filepath)
        
        if self.verbose:
            print(f"[Unsupervised] Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> None:
        """Load trained autoencoder model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        if self._torch is None:
            raise ImportError("PyTorch is required")
        
        torch = self._torch
        checkpoint = torch.load(filepath, map_location=self._device, weights_only=False)
        
        # Rebuild model with saved dimensions
        self.latent_dim = checkpoint['latent_dim']
        
        # Use input_dim from the checkpoint when present (older checkpoints lack it), else infer it.
        saved_input_dim = checkpoint.get('input_dim')
        
        if saved_input_dim is not None:
            input_dim = saved_input_dim
        else:
            # Fallback: infer from current code
            state_features = build_state_features(list(self.services), self.nodes, self.latency_matrix, self._service_demands)
            input_dim = state_features.shape[-1]
        
        saved_services = int(checkpoint.get('num_services', len(self.services)))
        saved_nodes = int(checkpoint.get('num_nodes', len(self.nodes)))
        self._model = build_autoencoder_model(
            torch,
            self._nn,
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            num_services=saved_services,
            num_nodes=saved_nodes,
            device=self._device,
        )
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.eval()
        
        self._training_losses = checkpoint.get('training_losses', [])
        
        if self.verbose:
            print(f"[Unsupervised] Checkpoint loaded from {filepath}")

    # ---------- Constraints and helpers ----------

    def _apply_allocation(self, svc: Any, avail: Dict[str, float], gpu_shared: bool) -> None:
        cpu, ram, gpu, exec_time, period = self._service_demands(svc)
        # Use period-based CPU if applicable
        if period > 0 and period < float('inf'):
            cpu_utilization_factor = min(1.0, exec_time / period)
            effective_cpu = cpu * cpu_utilization_factor
        else:
            effective_cpu = cpu
        avail["cpu"] = max(0.0, avail.get("cpu", 0.0) - effective_cpu)
        used_ram = ram + (gpu if gpu_shared else 0.0)
        avail["ram_gb"] = max(0.0, avail.get("ram_gb", 0.0) - used_ram)
        # GPU accounting uses capacity feasibility only; leave 'available' as-is (simulator tracks runtime levels)

    def _latency_cost(self, svc: Any, candidate_idx: int, placement: Dict[str, str]) -> float:
        """Calculate latency cost using base class helper."""
        return self._latency_cost_full(svc, candidate_idx, placement, self.latency_matrix, self._node_index, self.services)

    def train(self, num_batches: int = 10) -> None:
        """Train the autoencoder for N batches.
        
        Args:
            num_batches: Number of training batches to perform
        """
        self._last_train_loss = 0.0
        self._last_train_violations = 0
        
        for batch_idx in range(num_batches):
            loss, violations = self._train_single_batch(self.nodes, self.services, self.latency_matrix)
            self._last_train_loss = loss
            self._last_train_violations = violations
            
            if self.verbose and (batch_idx + 1) % 5 == 0:
                print(f"[AutoEncoder Training] Batch {batch_idx + 1}/{num_batches}, Loss: {loss:.4f}, Violations: {violations:.1f}")

    # ---------- Optimize ----------
    def _ensure_model_ready(self, state_features: np.ndarray, expected_services: int, expected_nodes: int) -> None:
        """Ensure autoencoder model is initialized with correct dimensions."""
        model_services = getattr(self._model, "num_services", None) if self._model is not None else None
        model_nodes = getattr(self._model, "num_nodes", None) if self._model is not None else None
        
        if self._model is None or model_services != expected_services or model_nodes != expected_nodes:
            if self.verbose:
                print(f"[AutoEncoder] Rebuilding model: ({model_services},{model_nodes}) -> ({expected_services},{expected_nodes})")
            input_dim = state_features.shape[-1]
            self._model = build_autoencoder_model(
                self._torch,
                self._nn,
                input_dim=input_dim,
                latent_dim=self.latent_dim,
                num_services=expected_services,
                num_nodes=expected_nodes,
                device=self._device,
            )

    def _decode_to_placement(self, placement_probs: Any, svcs: List[Any]) -> Dict[int, int]:
        """Decode placement probabilities to node assignments.
        
        Returns:
            Dict mapping service index -> node index
        """
        torch = self._torch
        placement: Dict[int, int] = {}
        node_usage = {i: {"cpu": 0.0, "ram": 0.0, "gpu": 0.0} for i in range(len(self.nodes))}
        
        # Build in topological order to include dependency latency in period checks
        topo_svcs = self._topological_sort(svcs)
        svc_idx_map = {id(s): i for i, s in enumerate(svcs)}
        
        # Seed with the current assignment so services outside svcs keep their node
        placement_map: Dict[str, str] = {}
        if hasattr(self, '_current_assignment') and self._current_assignment:
            placement_map = dict(self._current_assignment)

        # Seed node usage with current load from non-optimizable services
        node_index_by_id = {str(getattr(n, 'node_id', i)): i for i, n in enumerate(self.nodes)}
        baseline_usage = self._current_load_usage(placement_map, exclude_services=svcs)
        for nid, usage in baseline_usage.items():
            node_idx = node_index_by_id.get(str(nid))
            if node_idx is None:
                continue
            node_usage[node_idx]["cpu"] += float(usage.get("cpu", 0.0))
            node_usage[node_idx]["ram"] += float(usage.get("ram", 0.0))
            node_usage[node_idx]["gpu"] += float(usage.get("gpu", 0.0))
        
        # Depth-first decoding in probability order with backtracking: when a service has no
        # feasible node, revisit earlier choices instead of failing the whole decision.
        import copy as _copy
        budget = [self.backtrack_budget]

        def place(i: int) -> bool:
            if i == len(topo_svcs):
                return True
            svc = topo_svcs[i]
            svc_idx = svc_idx_map[id(svc)]
            sid = str(getattr(svc, 'service_id', svc_idx))
            feasible = self._find_feasible_nodes(svc, node_usage, placement_map)
            probs = placement_probs[svc_idx].cpu().numpy()
            for node_idx in sorted(feasible, key=lambda k: float(probs[k]), reverse=True):
                if budget[0] <= 0:
                    return False
                budget[0] -= 1
                saved_usage = _copy.deepcopy(node_usage)
                prev = placement_map.get(sid)
                placement[svc_idx] = int(node_idx)
                placement_map[sid] = str(getattr(self.nodes[node_idx], 'node_id', node_idx))
                self._update_node_usage(int(node_idx), svc, node_usage)
                if place(i + 1):
                    return True
                node_usage.clear(); node_usage.update(saved_usage)
                placement.pop(svc_idx, None)
                if prev is None:
                    placement_map.pop(sid, None)
                else:
                    placement_map[sid] = prev
            return False

        if not place(0):
            raise RuntimeError("No feasible placement found by autoencoder decoding (backtracking exhausted)")

        return placement

    def _find_feasible_nodes(self, svc: Any, node_usage: Dict[int, Dict[str, float]], 
                            placement_map: Dict[str, str]) -> List[int]:
        """Find all feasible nodes for a service.
        
        Returns:
            List of feasible node indices
        """
        feasible_nodes = []
        
        from .orchestrator_utils import OrchestratorUtils
        period_utils = OrchestratorUtils(self.services, self.nodes, self.latency_matrix)
        for node_idx, node in enumerate(self.nodes):
            # Check node lock
            locked_nid = getattr(svc, 'locked_node_id', None)
            if locked_nid is not None and str(getattr(node, 'node_id', '')) != str(locked_nid):
                continue

            # Check layer lock
            locked_layer = getattr(svc, 'locked_layer_id', None)
            if locked_layer is not None:
                node_layer = getattr(node, 'layer_id', None)
                if node_layer != locked_layer:
                    continue

            # Multi-layer services may use any of their allowed layers (no layer preservation,
            # which is not part of V(x))
            node_layer = getattr(node, 'layer_id', None)
            if getattr(svc, 'multi_layer', False):
                allowed_layers = getattr(svc, 'layers', [])
                if allowed_layers and node_layer not in allowed_layers:
                    continue

            # Check resource feasibility
            nid = str(getattr(node, 'node_id', ''))
            avail = {"cpu": node.metrics.get_capacity().get("cpu", 0.0) - node_usage[node_idx]["cpu"],
                     "ram_gb": node.metrics.get_capacity().get("ram_gb", 0.0) - node_usage[node_idx]["ram"]}
            caps = node.metrics.get_capacity()
            gpu_shared = bool(node.metrics.get_property("gpu_shared") or getattr(node, "gpu_shared", False))

            if not self._is_feasible(svc, nid, avail, caps, gpu_shared):
                continue
            
            # Period part of V(x), shared with every other orchestrator
            if not period_utils.service_period_ok(svc, nid, placement_map):
                continue
            
            feasible_nodes.append(node_idx)
        
        return feasible_nodes

    def _update_node_usage(self, node_idx: int, svc: Any, node_usage: Dict[int, Dict[str, float]]) -> None:
        """Update node usage after placing a service."""
        node_id = str(getattr(self.nodes[node_idx], 'node_id', ''))
        cpu, ram, gpu, exec_time, period = self._service_demands(svc, node_id)
        if period > 0 and period < float('inf'):
            cpu_utilization_factor = min(1.0, exec_time / period)
            effective_cpu = cpu * cpu_utilization_factor
        else:
            effective_cpu = cpu
        
        node = self.nodes[node_idx]
        gpu_shared = bool(node.metrics.get_property("gpu_shared") or getattr(node, "gpu_shared", False))
        ram_needed = ram + (gpu if gpu_shared else 0.0)
        
        node_usage[node_idx]["cpu"] += effective_cpu
        node_usage[node_idx]["ram"] += ram_needed
        node_usage[node_idx]["gpu"] += gpu

    def _placement_to_result_dict(self, placement: Dict[str, int], svcs: List[Any]) -> Dict[str, str]:
        """Convert placement dict {service_idx: node_idx} to {service_id: node_id}."""
        result = {}
        for svc_idx, node_idx in placement.items():
            if node_idx >= 0 and node_idx < len(self.nodes):
                svc = svcs[svc_idx]
                node = self.nodes[node_idx]
                service_id = str(getattr(svc, 'service_id', svc_idx))
                node_id = str(getattr(node, 'node_id', node_idx))
                result[service_id] = node_id
        return result

    def _propose_assignment(self, services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Generate a service-to-node assignment (implements BaseOrchestrator abstract method).
        
        Args:
            services: Optional services list. If provided, uses these instead of self.services.
        
        Returns:
            Dict mapping service_id (str) to node_id (str)
        """
        if self._torch is None:
            raise ImportError("PyTorch is required for AutoencoderOrchestrator")
        
        torch = self._torch
        
        svcs: List[Any] = list(services) if services is not None else list(self.services)
                
        # Build state features
        state_features = build_state_features(svcs, self.nodes, self.latency_matrix, self._service_demands)
        state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self._device)
        
        # Ensure model is ready
        expected_services = len(svcs)
        expected_nodes = len(self.nodes)
        self._ensure_model_ready(state_features, expected_services, expected_nodes)
        
        # Generate placement
        self._model.eval()
        with torch.no_grad():
            placement_logits, _ = self._model(state_tensor)
            placement_logits = placement_logits.squeeze(0)
            placement_probs = torch.softmax(placement_logits / self.temperature, dim=-1)
            placement = self._decode_to_placement(placement_probs, svcs)
        
        result = self._placement_to_result_dict(placement, svcs)

        from .orchestrator_utils import OrchestratorUtils
        strict_violations = OrchestratorUtils(self.services, self.nodes, self.latency_matrix).validate_placement(result)
        if strict_violations:
            raise RuntimeError(f"Autoencoder placement violates constraints: {strict_violations[0]}")

        if self.debug:
            print(f"[Autoencoder] Generated placement: {result}")
        
        return result

    def optimize(self, services: Optional[Sequence[Any]] = None,
                 current_assignment: Optional[Dict[str, str]] = None,
                 hysteresis_threshold: float = 0.0) -> Dict[str, str]:
        """Generate optimized placement with hysteresis support.
        
        Delegates to base class which handles hysteresis logic.
        """
        # Read by the placement search (seed placement and baseline load)
        self._current_assignment = current_assignment if current_assignment else {}
        return super().optimize(services, current_assignment, hysteresis_threshold)

    def run_single_episode_training(self, all_nodes, services, latency):
        """Perform a single training batch for the autoencoder and return (loss, violation_count)."""
        # Update orchestrator with new environment
        self.services = services
        self.num_services = len(services)
        
        # Filter non-schedulable nodes
        locked_node_ids = set()
        for s in services:
            locked_nid = getattr(s, 'locked_node_id', None)
            if locked_nid is not None:
                locked_node_ids.add(str(locked_nid))
        self.nodes = [
            n for n in all_nodes
            if getattr(n, 'schedulable', True) or str(getattr(n, 'node_id', '')) in locked_node_ids
        ]
        self.num_nodes = len(self.nodes)
        _lat = latency
        if hasattr(_lat, 'base_matrix'):
            _lat = _lat.base_matrix
        self.latency_matrix = np.asarray(_lat, dtype=np.float32) if _lat is not None else None

        # Train for one batch
        self.train(num_batches=1)
        
        # Get placement after training (a scenario may have no feasible placement)
        try:
            self.optimize(services)
        except RuntimeError:
            pass
    
    def _train_single_batch(self, all_nodes, services, latency):
        """Perform a single training batch for the autoencoder and return (loss, violation_count)"""
        # Note: all_nodes, services, latency are already updated in train() or run_single_episode_training()
        # This method is called by train() which updates self.nodes/services/latency first

        if self._torch is None:
            raise ImportError("PyTorch is required for AutoEncoderOrchestrator")

        loss, violations, model, optimizer, scheduler = train_single_batch(
            torch=self._torch,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            device=self._device,
            use_amp=self.use_amp,
            scaler=self._scaler,
            services=self.services,
            nodes=self.nodes,
            latency_matrix=self.latency_matrix,
            batch_size=self.batch_size,
            lr=self.lr,
            latent_dim=self.latent_dim,
            verbose=self.verbose,
            service_demands=self._service_demands,
            is_feasible=self._is_feasible,
            latency_cost=self._latency_cost,
            topological_sort=self._topological_sort,
        )

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        return float(loss), float(violations)
