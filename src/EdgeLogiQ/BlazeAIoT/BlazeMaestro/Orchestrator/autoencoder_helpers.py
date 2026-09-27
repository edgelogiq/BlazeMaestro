from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext
import random
import numpy as np


def lazy_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.optim as optim  # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch, nn, optim, device
    except Exception:
        return None, None, None, None


@dataclass
class OrchestratorWeights:
    """Weights for scoring each candidate node. Higher weight = more important to minimize."""
    energy: float = 10.0  # Increased: prioritize energy minimization
    latency: float = 1.0  # Reduced: secondary importance
    balance: float = 0.5  # Reduced: tertiary importance


def build_state_features(
    services: List[Any],
    nodes: Sequence[Any],
    latency_matrix: np.ndarray,
    service_demands: Callable[[Any], Tuple[float, float, float, float, float]],
) -> np.ndarray:
    """Build feature matrix for the current state with enhanced latency-awareness."""
    if latency_matrix is None:
        latency_matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    else:
        # Unwrap LatencyMatrixWithJitter (or any wrapper exposing .base_matrix)
        if hasattr(latency_matrix, 'base_matrix'):
            latency_matrix = latency_matrix.base_matrix
        latency_matrix = np.asarray(latency_matrix, dtype=np.float32)

    service_features = []

    max_cpu = max((float(n.metrics.get_capacity().get("cpu", 0.0)) for n in nodes), default=1.0)
    
    # Build service_id to index map for dependency latency calculation
    svc_to_idx = {str(getattr(s, 'service_id', i)): i for i, s in enumerate(services)}
    node_to_idx = {str(getattr(n, 'node_id', i)): i for i, n in enumerate(nodes)}

    for svc in services:
        cpu, ram, gpu, exec_time, period = service_demands(svc)
        rate = 1.0 / period if period > 0 and period < float("inf") else 0.0

        # Calculate estimated execution time on each node
        node_est_times = []
        for n in nodes:
            caps = n.metrics.get_capacity()
            cap_cpu = float(caps.get("cpu", 0.0))
            if cap_cpu > 0:
                speed_factor = cap_cpu / max_cpu
                est_time = exec_time / speed_factor
            else:
                est_time = exec_time * 100.0
            node_est_times.append(est_time)
        
        # Calculate per-node dependency latency statistics (ENHANCEMENT)
        # For each candidate node, what's the min/max/avg latency to this service's dependencies?
        per_node_dep_latency = []
        has_dependencies = False
        
        dependencies = getattr(svc, 'dependencies', [])
        if dependencies:
            has_dependencies = True
            for target_node_idx in range(len(nodes)):
                dep_latencies = []
                for dep in dependencies:
                    dep_svc_id = str(getattr(dep, 'service_id', getattr(dep, 'service', '')))
                    # Find where dependency is currently placed (use first node as default)
                    dep_svc_idx = svc_to_idx.get(dep_svc_id)
                    if dep_svc_idx is not None:
                        dep_svc = services[dep_svc_idx]
                        dep_node_id = getattr(dep_svc, 'locked_node_id', None)
                        if dep_node_id:
                            dep_node_idx = node_to_idx.get(str(dep_node_id))
                            if dep_node_idx is not None:
                                # Latency from target_node to dependency's node
                                dep_latencies.append(latency_matrix[target_node_idx, dep_node_idx])
                
                if dep_latencies:
                    per_node_dep_latency.extend([
                        min(dep_latencies),
                        max(dep_latencies),
                        sum(dep_latencies) / len(dep_latencies)
                    ])
                else:
                    # No dependencies or can't determine - use zeros
                    per_node_dep_latency.extend([0.0, 0.0, 0.0])
        else:
            # No dependencies - add zeros for all nodes
            per_node_dep_latency = [0.0, 0.0, 0.0] * len(nodes)
        
        # Calculate period slack/margin (ENHANCEMENT)
        # How much time is left after exec_time if period exists?
        if period > 0 and period < float("inf"):
            period_slack = period - exec_time  # Time available for latency
            slack_ratio = period_slack / period  # Normalized slack (0-1)
        else:
            period_slack = float("inf")
            slack_ratio = 1.0
        
        # Criticality flag: is this a tight real-time constraint? (ENHANCEMENT)
        is_critical = float(period < float("inf") and period_slack < 0.1)  # Less than 100ms slack

        service_features.append([
            cpu, ram, gpu, exec_time, period, rate, 
            period_slack, slack_ratio, is_critical, float(has_dependencies)
        ] + node_est_times + per_node_dep_latency)

    node_cpu = []
    node_ram = []
    node_gpu = []
    for node in nodes:
        caps = node.metrics.get_capacity()
        node_cpu.append(float(caps.get("cpu", 0.0)))
        node_ram.append(float(caps.get("ram_gb", 0.0)))
        node_gpu.append(float(caps.get("gpu_gb", 0.0)))

    total_cpu = sum(node_cpu)
    total_ram = sum(node_ram)
    total_gpu = sum(node_gpu)
    avg_cpu = total_cpu / max(len(nodes), 1)
    avg_ram = total_ram / max(len(nodes), 1)
    avg_gpu = total_gpu / max(len(nodes), 1)

    latency_flat = latency_matrix.flatten()
    avg_latency = float(np.mean(latency_flat))
    max_latency = float(np.max(latency_flat))

    # Per-node energy features so the model can learn energy-aware placement.
    # node_is_battery: 1.0 for battery-powered nodes, 0.0 for grid-powered.
    # node_power_norm: active power draw normalised by the max across all nodes.
    node_is_battery = [1.0 if getattr(node, 'battery_powered', False) else 0.0 for node in nodes]
    max_power = max((float(getattr(node, 'power_active', 0.0)) for node in nodes), default=1.0)
    node_power_norm = [float(getattr(node, 'power_active', 0.0)) / max(max_power, 1e-6) for node in nodes]

    state_features = []
    for svc_feat in service_features:
        combined = svc_feat + [
            total_cpu, total_ram, total_gpu,
            avg_cpu, avg_ram, avg_gpu,
            avg_latency, max_latency,
            len(nodes),
        ] + node_is_battery + node_power_norm
        state_features.append(combined)

    return np.array(state_features, dtype=np.float32)


def build_autoencoder_model(
    torch: Any,
    nn: Any,
    *,
    input_dim: int,
    latent_dim: int,
    num_services: int,
    num_nodes: int,
    device: Any,
):
    """Build the autoencoder architecture (decoder outputs num_services * num_nodes logits)."""
    class PlacementAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim, num_services, num_nodes):
            super().__init__()
            self.num_services = num_services
            self.num_nodes = num_nodes
            output_dim = num_nodes
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, latent_dim),
                nn.Tanh(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, output_dim),
            )

        def forward(self, x):
            """Forward pass: [batch, num_services, input_dim] -> [batch, num_services, num_nodes]"""
            batch_size = x.size(0)
            x_flat = x.view(-1, x.size(-1))
            latent = self.encoder(x_flat)
            placement_logits = self.decoder(latent)
            placement_logits = placement_logits.view(batch_size, self.num_services, self.num_nodes)
            latent = latent.view(batch_size, self.num_services, -1)
            return placement_logits, latent

    model = PlacementAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_services=num_services,
        num_nodes=num_nodes,
    )
    return model.to(device)


def generate_training_samples(
    *,
    services: List[Any],
    nodes: Sequence[Any],
    num_samples: int,
    service_demands: Callable[[Any], Tuple[float, float, float, float, float]],
    is_feasible: Callable[[Any, str, Dict[str, float], Dict[str, float], bool], bool],
    latency_cost: Callable[[Any, int, Dict[str, str]], float],
    topological_sort: Callable[[List[Any]], List[Any]],
    build_state_features_fn: Callable[[List[Any], Sequence[Any], np.ndarray, Callable[[Any], Tuple[float, float, float, float, float]]], np.ndarray],
    latency_matrix: np.ndarray,
) -> List[Tuple[np.ndarray, List[int]]]:
    """Generate training samples by exploring different placements."""
    samples: List[Tuple[np.ndarray, List[int]]] = []

    for _ in range(num_samples):
        placement: List[Optional[int]] = [None] * len(services)
        node_usage = {i: {"cpu": 0.0, "ram": 0.0, "gpu": 0.0} for i in range(len(nodes))}
        placement_map: Dict[str, str] = {}
        topo_svcs = topological_sort(services)
        svc_idx_map = {id(s): i for i, s in enumerate(services)}
        for svc in topo_svcs:
            feasible_nodes: List[int] = []
            for node_idx, node in enumerate(nodes):
                locked_nid = getattr(svc, "locked_node_id", None)
                if locked_nid is not None and str(getattr(node, "node_id", "")) != str(locked_nid):
                    continue
                locked_layer = getattr(svc, "locked_layer_id", None)
                if locked_layer is not None:
                    node_layer = getattr(node, "layer_id", None)
                    if node_layer != locked_layer:
                        continue

                nid = str(getattr(node, "node_id", ""))
                avail = {
                    "cpu": node.metrics.get_capacity().get("cpu", 0.0) - node_usage[node_idx]["cpu"],
                    "ram_gb": node.metrics.get_capacity().get("ram_gb", 0.0) - node_usage[node_idx]["ram"],
                }
                caps = node.metrics.get_capacity()
                gpu_shared = bool(node.metrics.get_property("gpu_shared") or getattr(node, "gpu_shared", False))
                if not is_feasible(svc, nid, avail, caps, gpu_shared):
                    continue
                cpu, ram, gpu, exec_time, period = service_demands(svc)
                if period < float("inf"):
                    lat = latency_cost(svc, node_idx, placement_map)
                    if exec_time + lat > period:
                        continue
                feasible_nodes.append(node_idx)

            if feasible_nodes:
                # Enhanced sampling: prefer nodes with more period slack margin
                # This helps the model learn to favor safe placements over risky ones
                cpu, ram, gpu, exec_time, period = service_demands(svc)
                if period < float("inf") and len(feasible_nodes) > 1:
                    # Calculate slack for each feasible node
                    node_slacks = []
                    for node_idx in feasible_nodes:
                        lat = latency_cost(svc, node_idx, placement_map)
                        total_time = exec_time + lat
                        slack = period - total_time  # How much margin
                        node_slacks.append(max(0.001, slack))  # Avoid zero weights
                    
                    # Convert slack to probability weights (more slack = higher weight)
                    # Use softmax-like weighting: prefer high slack but still some randomness
                    slack_array = np.array(node_slacks)
                    slack_mean = slack_array.mean()
                    slack_std = slack_array.std() + 1e-6
                    normalized_slacks = (slack_array - slack_mean) / slack_std
                    # Apply softmax with temperature to control randomness
                    # Higher temperature = more random, lower = more deterministic
                    temperature = 2.0  # Moderate preference for high-slack nodes
                    weights = np.exp(normalized_slacks / temperature)
                    weights = weights / weights.sum()
                    
                    # Sample with weighted probability
                    node_idx = np.random.choice(feasible_nodes, p=weights)
                else:
                    # No period constraint or only one option: uniform random
                    node_idx = random.choice(feasible_nodes)
            else:
                # Fallback: relax period constraint to keep training moving on hard cases.
                # Prefer resource-feasible nodes; if none, allow any node to generate a sample.
                fallback_nodes: List[int] = []
                for node_idx, node in enumerate(nodes):
                    locked_nid = getattr(svc, "locked_node_id", None)
                    if locked_nid is not None and str(getattr(node, "node_id", "")) != str(locked_nid):
                        continue
                    locked_layer = getattr(svc, "locked_layer_id", None)
                    if locked_layer is not None:
                        node_layer = getattr(node, "layer_id", None)
                        if node_layer != locked_layer:
                            continue

                    nid = str(getattr(node, "node_id", ""))
                    avail = {
                        "cpu": node.metrics.get_capacity().get("cpu", 0.0) - node_usage[node_idx]["cpu"],
                        "ram_gb": node.metrics.get_capacity().get("ram_gb", 0.0) - node_usage[node_idx]["ram"],
                    }
                    caps = node.metrics.get_capacity()
                    gpu_shared = bool(node.metrics.get_property("gpu_shared") or getattr(node, "gpu_shared", False))
                    if is_feasible(svc, nid, avail, caps, gpu_shared):
                        fallback_nodes.append(node_idx)

                if not fallback_nodes:
                    fallback_nodes = list(range(len(nodes)))

                node_idx = random.choice(fallback_nodes)

            svc_idx = svc_idx_map[id(svc)]
            placement[svc_idx] = node_idx
            placement_map[str(getattr(svc, "service_id", str(svc_idx)))] = str(
                getattr(nodes[node_idx], "node_id", node_idx)
            )

            cpu, ram, gpu, exec_time, period = service_demands(svc)
            if period > 0 and period < float("inf"):
                cpu_utilization_factor = min(1.0, exec_time / period)
                effective_cpu = cpu * cpu_utilization_factor
            else:
                effective_cpu = cpu
            node = nodes[node_idx]
            gpu_shared = bool(node.metrics.get_property("gpu_shared") or getattr(node, "gpu_shared", False))
            ram_needed = ram + (gpu if gpu_shared else 0.0)
            node_usage[node_idx]["cpu"] += effective_cpu
            node_usage[node_idx]["ram"] += ram_needed
            node_usage[node_idx]["gpu"] += gpu

        final_placement: List[int] = [int(p) if p is not None else 0 for p in placement]
        state_features = build_state_features_fn(services, nodes, latency_matrix, service_demands)
        samples.append((state_features, final_placement))

    return samples


def train_single_batch(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    use_amp: bool,
    scaler: Any,
    services: List[Any],
    nodes: Sequence[Any],
    latency_matrix: np.ndarray,
    batch_size: int,
    lr: float,
    latent_dim: int,
    verbose: bool,
    service_demands: Callable[[Any], Tuple[float, float, float, float, float]],
    is_feasible: Callable[[Any, str, Dict[str, float], Dict[str, float], bool], bool],
    latency_cost: Callable[[Any, int, Dict[str, str]], float],
    topological_sort: Callable[[List[Any]], List[Any]],
) -> Tuple[float, float, Any, Any, Any]:
    """Train the autoencoder for a single batch and return loss, violations, and updated state."""
    samples = generate_training_samples(
        services=services,
        nodes=nodes,
        num_samples=batch_size,
        service_demands=service_demands,
        is_feasible=is_feasible,
        latency_cost=latency_cost,
        topological_sort=topological_sort,
        build_state_features_fn=build_state_features,
        latency_matrix=latency_matrix,
    )

    states = []
    placements = []
    sample_weights = []
    node_ids = [str(getattr(n, "node_id", i)) for i, n in enumerate(nodes)]
    for state, placement in samples:
        states.append(state)
        placements.append(placement)

        # Weight samples by feasibility to discourage period-violating targets.
        placement_map: Dict[str, str] = {}
        for svc_idx, node_idx in enumerate(placement):
            if node_idx is None:
                continue
            if node_idx < 0 or node_idx >= len(node_ids):
                continue
            sid = str(getattr(services[svc_idx], "service_id", svc_idx))
            placement_map[sid] = node_ids[node_idx]

        violations = 0
        for svc_idx, svc in enumerate(services):
            node_idx = placement[svc_idx] if svc_idx < len(placement) else None
            if node_idx is None or node_idx < 0 or node_idx >= len(nodes):
                continue
            _, _, _, exec_time, period = service_demands(svc)
            if period < float("inf"):
                dep_latency = latency_cost(svc, node_idx, placement_map)
                if exec_time + dep_latency > period:
                    violations += 1

        # Energy soft objective: weight samples by energy efficiency so the model
        # learns to prefer placements that avoid draining battery-powered nodes.
        # OrchestratorUtils.estimate_energy returns 0 for grid-powered nodes, so
        # all-grid deployments get energy_factor == 1.0 (no change in behaviour).
        # Lazy import avoids circular import at module load time.
        from .orchestrator_utils import OrchestratorUtils  # noqa: PLC0415
        energy_total = 0.0
        for svc_idx_e, node_idx_e in enumerate(placement):
            if node_idx_e is None or node_idx_e < 0 or node_idx_e >= len(nodes):
                continue
            energy_total += OrchestratorUtils.estimate_energy(services[svc_idx_e], nodes[node_idx_e])
        energy_factor = 1.0 / (1.0 + energy_total * 0.1)

        if violations == 0:
            sample_weights.append(max(energy_factor, 0.1))
        else:
            # Heavily downweight infeasible samples (keep some signal for edge cases).
            sample_weights.append(max(0.05, energy_factor / (1.0 + violations * 5.0)))

    states = torch.FloatTensor(np.array(states)).to(device)
    placements = torch.LongTensor(np.array(placements)).to(device)
    input_dim = states.size(-1)

    model_services = getattr(model, "num_services", None) if model is not None else None
    model_nodes = getattr(model, "num_nodes", None) if model is not None else None
    # Detect input-dimension change (e.g. feature additions like energy features)
    model_input_dim: Optional[int] = None
    if model is not None:
        try:
            model_input_dim = model.encoder[0].in_features
        except (AttributeError, IndexError):
            pass

    model_rebuilt = False
    if model is None or model_services != len(services) or model_nodes != len(nodes) or (model_input_dim is not None and model_input_dim != input_dim):
        if verbose:
            print(f"[AutoEncoder Training] Rebuilding: ({model_services},{model_nodes}) -> ({len(services)},{len(nodes)})")
        model = build_autoencoder_model(
            torch,
            torch.nn,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_services=len(services),
            num_nodes=len(nodes),
            device=device,
        )
        model_rebuilt = True

    if optimizer is None or model_rebuilt:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=60, min_lr=lr * 0.02
        )

    model.train()
    batch_states = states
    batch_placements = placements
    batch_weights = torch.FloatTensor(np.array(sample_weights, dtype=np.float32)).to(device)

    cast_ctx = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext
    with cast_ctx():
        placement_logits, latent = model(batch_states)
        loss = torch.nn.functional.cross_entropy(
            placement_logits.view(-1, len(nodes)),
            batch_placements.view(-1),
            reduction="none",
        )
        batch_weights_expanded = batch_weights.unsqueeze(1).expand(-1, len(services)).reshape(-1)
        loss = (loss * batch_weights_expanded).mean()
        latent_reg = 0.01 * torch.mean(latent ** 2)
        loss = loss + latent_reg

    violations_per_batch = float(sum(1 for w in sample_weights if w < 1.0))

    optimizer.zero_grad()
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if scheduler is not None:
        scheduler.step(float(loss.detach().item()))

    return float(loss.item()), float(violations_per_batch), model, optimizer, scheduler