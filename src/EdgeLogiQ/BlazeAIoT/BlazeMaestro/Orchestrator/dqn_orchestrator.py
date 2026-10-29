from __future__ import annotations

import copy
import os
from typing import Any, List, Sequence, Optional, Callable, Dict

import torch

from ..Metrics.base import BaseMetricsFacade
from ..Metrics.simulator_facade import SimulatorMetricsFacade
from .orchestrator_utils import OrchestratorUtils
from .dqn_environment import DQNEnvironment
from .base_orchestrator import BaseOrchestrator


class DQNOrchestrator(BaseOrchestrator):
    """Torch-based DQN orchestrator adapted to simulator types with reward logging."""

    def __init__(
        self,
        services: Sequence[Any],
        nodes: Sequence[Any],
        energy_estimator=None,
        metrics: Optional[BaseMetricsFacade] = None,
        episodes: int = 300,
        gamma: float = 0.99,
        lr: float = 3e-4,  # Better learning rate for DQN
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,  # Keep more exploration
        epsilon_decay: float = 0.9995,  # Slower decay
        batch_size: int = 128,  # Larger batch for better gradient estimates
        max_memory: int = 10000,
        update_target_steps: int = 100,
        use_post_validation_repairs: bool = False,
    ) -> None:
        metrics_facade = metrics or SimulatorMetricsFacade(nodes)
        
        # Initialize base class
        super().__init__(services, nodes, metrics_facade)
        
        self.energy_estimator = energy_estimator
        # Allow passing a latency matrix as the third positional arg (legacy usage in compare script)
        self.latency_matrix = None
        if isinstance(energy_estimator, (list, tuple)):
            if energy_estimator and isinstance(energy_estimator[0], (list, tuple)):
                self.latency_matrix = energy_estimator
        self.episodes = episodes
        # Logging/tracking
        self._episode_logger: Optional[Callable[[int, float], None]] = None
        self._last_rewards: List[float] = []
        # DQN hyperparams
        self._gamma = gamma
        self._lr = lr
        self._eps_start = epsilon_start
        self._eps_end = epsilon_end
        self._eps_decay = epsilon_decay
        self._batch_size = batch_size
        self._max_memory = max_memory
        self._update_target_steps = update_target_steps
        self._use_post_validation_repairs = use_post_validation_repairs
        # Agent state - persisted across calls for continual learning
        # Calculate dimensions based on num_nodes (now available after super().__init__)
        # Enhanced state: 9 service features + (10 per-node features × num_nodes) + 1 global = 9 + num_nodes*10 + 1
        self._state_dim = 9 + self.num_nodes * 10 + 1
        self._action_dim = self.num_nodes
        # Initialize agent
        try:
            torch, nn, optim, device = self._lazy_torch()
            from .dqn_agent import DQNAgent
            self._agent = DQNAgent(
                self._state_dim,
                self._action_dim,
                lr=self._lr,
                gamma=self._gamma,
                eps_start=self._eps_start,
                eps_end=self._eps_end,
                eps_decay=self._eps_decay,
                batch_size=self._batch_size,
                max_memory=self._max_memory,
                update_steps=self._update_target_steps,
                device=device,
            )
        except ImportError:
            # PyTorch not available, will handle gracefully in _propose_assignment
            self._agent = None
        # Logging/training state
        self._previous_assignment: Optional[List[int]] = None
        self._trained_agent: Optional[Any] = self._agent
        self.debug = False
        self.backtrack_budget = 5000  # max placement trials per decision (bounds decision time)


    # ---------- DQN infra (lazy torch import) ----------
    def _lazy_torch(self):
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore
            import torch.optim as optim  # type: ignore
            # Detect CUDA availability
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch, nn, optim, device
        except Exception as e:
            raise ImportError("PyTorch is required for DQNOrchestrator. Install torch or use another orchestrator.") from e

    def _get_latency_matrix(self) -> List[List[float]]:
        """Get fresh latency matrix from metrics provider."""
        if getattr(self, "latency_matrix", None) is not None:
            matrix = self.latency_matrix
            # Ensure it's a list of lists
            if isinstance(matrix, (list, tuple)) and len(matrix) > 0 and isinstance(matrix[0], (list, tuple)):
                return [[float(x) for x in row] for row in matrix]
        if self.metrics and hasattr(self.metrics, 'get_latency_matrix'):
            try:
                matrix = self.metrics.get_latency_matrix()
                if isinstance(matrix, (list, tuple)) and len(matrix) > 0:
                    return [[float(x) for x in row] for row in matrix]
            except:
                pass
        return [[0.0] * self.num_nodes for _ in range(self.num_nodes)]

    def _dependency_wait(
        self,
        s_idx: int,
        n_idx: int,
        assignment: List[int],
        fallback_assignment: Optional[List[int]] = None,
        hard_only: bool = False,
    ) -> float:
        """Calculate latency waiting time due to service dependencies.
        
        Now traverses the full dependency tree to accumulate all upstream latencies.
        
        Args:
            s_idx: Service index
            n_idx: Node index being assigned
            assignment: Current assignment (list of node indices)
            fallback_assignment: Fallback assignment to use for unplaced services
            hard_only: If True, only include dependencies with constraint_level='hard'
            
        Returns:
            Wait time in seconds
        """
        svc = self.services[s_idx]
        deps = getattr(svc, "dependencies", []) or []
        if not deps:
            return 0.0

        throughput_mbps = 200.0  # Assumed network throughput
        latency_matrix = self._get_latency_matrix()
        visiting = set()

        def dfs(cur_svc_idx: int, cur_node_idx: int) -> float:
            total = 0.0
            cur_svc = self.services[cur_svc_idx]
            for dep in getattr(cur_svc, "dependencies", []) or []:
                # Skip if hard_only=True and constraint_level is not 'hard'
                if hard_only:
                    constraint_level = getattr(dep, "constraint_level", "hard")
                    if constraint_level != "hard":
                        continue
                
                dep_id = getattr(dep, "service_id", dep)
                # Find upstream service index
                up_idx = None
                for i, s in enumerate(self.services):
                    if getattr(s, "service_id", None) == dep_id:
                        up_idx = i
                        break
                if up_idx is None:
                    continue

                up_node_idx = None
                if up_idx < len(assignment):
                    try:
                        up_node_idx = int(assignment[up_idx])
                    except (ValueError, TypeError):
                        up_node_idx = None

                if up_node_idx is None and fallback_assignment is not None and up_idx < len(fallback_assignment):
                    try:
                        up_node_idx = int(fallback_assignment[up_idx])
                    except (ValueError, TypeError):
                        up_node_idx = None

                if up_node_idx is None:
                    continue

                try:
                    net_lat = float(latency_matrix[up_node_idx][cur_node_idx]) if latency_matrix is not None else 0.0
                except (IndexError, ValueError, TypeError):
                    net_lat = 0.0
                size_mb = float(getattr(dep, "size_mb", 0.0))
                transfer = (size_mb / max(1.0, throughput_mbps)) if size_mb > 0 else 0.0
                edge_cost = max(0.0, net_lat + transfer)
                # Avoid cycles
                if up_idx in visiting:
                    total += edge_cost
                    continue
                visiting.add(up_idx)
                total += edge_cost + dfs(up_idx, up_node_idx)
                visiting.remove(up_idx)
            return total

        return dfs(s_idx, int(n_idx))

    def _build_fallback_assignment(self, services: Sequence[Any]) -> List[Optional[int]]:
        """Build a fallback assignment list for locked services.

        This lets dependency latency checks account for locked upstream services
        even before they are placed in the current assignment.
        """
        node_index_map = {
            str(getattr(node, "node_id", idx)): idx for idx, node in enumerate(self.nodes)
        }
        fallback: List[Optional[int]] = [None] * len(services)
        for i, svc in enumerate(services):
            locked_nid = getattr(svc, "locked_node_id", None)
            if locked_nid is None:
                continue
            idx = node_index_map.get(str(locked_nid))
            if idx is not None:
                fallback[i] = idx
        return fallback

    def train(self, episodes: Optional[int] = None) -> None:
        """Train the DQN agent for N episodes.
        
        Args:
            episodes: Number of episodes to train for. If None, uses self.episodes.
        """
        if episodes is None:
            episodes = self.episodes
        
        if self._agent is None:
            raise RuntimeError("Agent not initialized. PyTorch may not be available.")
        
        # Ensure latency matrix is in correct format
        latency_matrix = self._get_latency_matrix()
        
        agent = self._agent
        best_assign = None
        best_reward = float('-inf')
        self._last_rewards = []
        
        for ep_idx in range(episodes):
            # Create fresh environment for this episode
            fallback_assignment = self._build_fallback_assignment(self.services)
            env = DQNEnvironment(
                self.services,
                self.nodes,
                latency_matrix,
                debug=self.debug,
                fallback_assignment=fallback_assignment,
            )
            state = env.reset()
            ep_reward = 0.0
            step_count = 0

            while env.state["s_idx"] < self.num_services and step_count < 1000:
                # Get valid actions (NEVER allow period violations - this must be a hard constraint)
                valid_actions = env.get_valid_actions(
                    allow_violations=False,
                    allow_period_violations=False,  # Hard constraint: never relax this
                    dependency_wait_func=self._dependency_wait,
                )

                relaxed_step = False
                if not valid_actions:
                    # Only relax RESOURCE constraints if absolutely necessary, but NEVER period constraints
                    # This teaches the model that period deadlines are non-negotiable
                    valid_actions = env.get_valid_actions(
                        allow_violations=True,  # Only relax resource violations
                        allow_period_violations=False,  # Keep period constraint firm
                        dependency_wait_func=self._dependency_wait,
                    )
                    relaxed_step = True

                if not valid_actions:
                    # No solution even with relaxed resources = training failure (architecture issue)
                    # Penalize heavily to encourage the agent to find period-safe placements earlier
                    ep_reward -= 10000.0
                    break

                # Select action with exploration
                temperature = 0.1
                action = agent.select(state, valid_actions, temperature=temperature)
                
                # Step environment
                next_state, reward, done = env.step(
                    action,
                    dependency_wait_func=self._dependency_wait,
                    nodes=self.nodes,
                    allow_violations=relaxed_step,
                    allow_period_violations=False,
                )

                if relaxed_step:
                    reward -= 2000.0  # Increased penalty for using relaxed resources
                
                # Store experience and train
                agent.store((state, action, reward, next_state, done))
                if len(agent.memory) >= agent.batch:
                    for _ in range(4):
                        agent.train()
                
                state = next_state
                ep_reward += reward
                step_count += 1
                
                if done:
                    break

            # Track episode reward and best assignment
            self._last_rewards.append(float(ep_reward))
            if ep_reward > best_reward and len(env.state["assign"]) == self.num_services:
                best_reward = ep_reward
                best_assign = list(env.state["assign"])
        
        # Store best assignment from training for later reference
        if best_assign is not None:
            self._previous_assignment = list(best_assign)

    def _propose_assignment(self, services: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Propose service assignment using trained DQN agent (inference only, no training)."""
        if services is not None:
            svcs = list(services)
        else:
            svcs = self.services
        orig_services = self.services
        self.services = svcs
        self.num_services = len(self.services)
        
        try:
            torch, nn, optim, device = self._lazy_torch()
        except ImportError as exc:
            raise ImportError("PyTorch is required for DQN inference") from exc

        # Refresh latency matrix for this decision cycle
        self.latency_matrix = self._get_latency_matrix()
        
        # Use pre-created agent (initialized in __init__)
        if self._agent is None:
            raise RuntimeError("DQN agent is not initialized; cannot run inference")
        
        agent = self._agent
        self._apply_checkpoint_weights(agent)

        # Single inference pass (no training, just greedy action selection)
        fallback_assignment = self._build_fallback_assignment(self.services)
        env = DQNEnvironment(
            self.services,
            self.nodes,
            self.latency_matrix,
            debug=self.debug,
            fallback_assignment=fallback_assignment,
        )
        state = env.reset()

        # Depth-first placement in Q-value order with backtracking: when a service has no
        # feasible node, revisit the previous choices instead of failing the whole decision.
        # Capacity bound = 1.0, the bound of V(x) shared by every orchestrator.
        budget = [self.backtrack_budget]

        def ranked_actions(state_vec, valid):
            agent.model.eval()
            with torch.no_grad():
                q = agent.model(torch.FloatTensor(state_vec).unsqueeze(0).to(agent.device)).squeeze(0)
            return sorted(valid, key=lambda a: float(q[a]) if a < q.size(0) else float("-inf"), reverse=True)

        def place(state_vec) -> bool:
            if env.state["s_idx"] >= self.num_services:
                return True
            valid = env.get_valid_actions(
                allow_violations=False,
                dependency_wait_func=self._dependency_wait,
                utilization_limit=1.0,
            )
            for action in ranked_actions(state_vec, valid):
                if budget[0] <= 0:
                    return False
                budget[0] -= 1
                saved = copy.deepcopy(env.state)
                placed = len(env.state["assign"])
                next_state, _, _ = env.step(
                    action,
                    dependency_wait_func=self._dependency_wait,
                    nodes=self.nodes,
                    allow_violations=False,
                    allow_period_violations=False,
                )
                if len(env.state["assign"]) == placed + 1 and place(next_state):
                    return True
                env.state = saved
            return False

        if not place(state):
            raise RuntimeError("No feasible placement found by DQN inference (backtracking exhausted)")

        # Extract best assignment from environment
        best_assign = list(env.state["assign"]) if len(env.state["assign"]) == self.num_services else None
        if best_assign is None:
            raise RuntimeError("Incomplete assignment produced by DQN inference")

        # Store trained agent reference for checkpoint saving
        self._trained_agent = agent

        # Convert assignment to dict
        placement = {}
        for i, svc in enumerate(svcs):
            sid = str(getattr(svc, 'service_id', i))
            nid = str(self.nodes[best_assign[i]].node_id)
            placement[sid] = nid

        # Validate placement
        valid = True
        for i, svc in enumerate(svcs):
            sid = str(getattr(svc, 'service_id', i))
            nid = placement.get(sid)
            if nid is None or not any(str(getattr(n, 'node_id', idx)) == nid for idx, n in enumerate(self.nodes)):
                valid = False
                break
            locked_nid = getattr(svc, 'locked_node_id', None)
            if locked_nid is not None and str(locked_nid) != nid:
                valid = False
                break
            locked_layer = getattr(svc, 'locked_layer_id', None)
            if locked_layer is not None:
                node = next((n for n in self.nodes if str(getattr(n, 'node_id', '')) == nid), None)
                if node and getattr(node, 'layer_id', None) != locked_layer:
                    valid = False
                    break

        self.services = orig_services
        self.num_services = len(self.services)
        
        if not valid:
            return {}

        strict_violations = OrchestratorUtils(self.services, self.nodes, self._get_latency_matrix()).validate_placement(placement)
        if strict_violations:
            raise RuntimeError(f"DQN placement violates constraints: {strict_violations[0]}")

        return placement
    
    # DEPRECATED: Old incomplete optimize method - see the correct one below
    # def optimize_old(self, services: Optional[Sequence[Any]] = None,
    #                  current_assignment: Optional[Dict[str, str]] = None,
    #                  hysteresis_threshold: float = 0.0) -> Dict[str, str]:
    #     \"\"\"
    #     Generate a service-to-node assignment using the DQN agent.
    #     \"\"\"
    #     pass
    
    def optimize(self, services: Optional[Sequence[Any]] = None, 
                 current_assignment: Optional[Dict[str, str]] = None,
                 hysteresis_threshold: float = 0.0) -> Dict[str, str]:
        """
        Generate a service-to-node assignment using the DQN agent.
        
        Args:
            services: Optional services list. If provided, uses these instead of self.services.
            current_assignment: Current assignment to compare against (for hysteresis)
            hysteresis_threshold: Only reassign if new placement is significantly better
        
        Returns:
            Dict mapping service_id (str) to node_id (str)
        """
        # Update previous assignment for stability bonus during inference
        if current_assignment and hasattr(self, '_trained_agent'):
            # Convert dict to list format for comparison
            svcs = services if services is not None else self.services
            prev_list = []
            for i, svc in enumerate(svcs):
                sid = str(getattr(svc, 'service_id', i))
                if sid in current_assignment:
                    # Find node index
                    nid = current_assignment[sid]
                    for ni, node in enumerate(self.nodes):
                        if str(getattr(node, 'node_id', ni)) == nid:
                            prev_list.append(ni)
                            break
                    else:
                        prev_list.append(0)  # Default to first node if not found
                else:
                    prev_list.append(0)
            self._previous_assignment = prev_list
        
        # Delegate to base class for hysteresis handling
        return super().optimize(services, current_assignment, hysteresis_threshold)

    # --- Checkpoint save/load ---
    def save_checkpoint(self, path: str):
        """Save the trained DQN model to a checkpoint file."""
        if not hasattr(self, '_trained_agent') or self._trained_agent is None:
            raise RuntimeError("No trained agent found. Call optimize() first.")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'policy_net_state': self._trained_agent.model.state_dict(),
            'target_net_state': self._trained_agent.target.state_dict(),
            'optimizer_state': self._trained_agent.opt.state_dict(),
            'epsilon': self._trained_agent.epsilon,
            'state_dim': self._state_dim,
            'action_dim': self._action_dim,
        }, path)
        
    def _apply_checkpoint_weights(self, agent) -> None:
        """Copy the weights loaded by load_checkpoint() into the live agent (once).

        The weight-loading step was dropped in a refactor (Feb 2026), so inference ran with
        an untrained network; this restores it.
        """
        data = getattr(self, "_checkpoint_data", None)
        if data is None or agent is None:
            return
        if data.get("state_dim") != self._state_dim or data.get("action_dim") != self._action_dim:
            raise RuntimeError(
                f"DQN checkpoint dimensions (state_dim={data.get('state_dim')}, action_dim={data.get('action_dim')}) "
                f"do not match the current platform (state_dim={self._state_dim}, action_dim={self._action_dim})")
        agent.model.load_state_dict(data["policy_net_state"])
        agent.target.load_state_dict(data["target_net_state"])
        agent.epsilon = data.get("epsilon", agent.epsilon)
        agent.model.eval()
        agent.target.eval()
        self._checkpoint_data = None

    def load_checkpoint(self, path: str):
        """Load a trained DQN model from a checkpoint file and prepare for inference."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, map_location="cpu" if not torch.cuda.is_available() else None)
        
        # Store dimensions for later use
        self._state_dim = checkpoint['state_dim']
        self._action_dim = checkpoint['action_dim']
        self._checkpoint_data = checkpoint
        # Agent will be reconstructed on next optimize() call with episodes=0

    def run_single_episode_training(self, all_nodes, services, latency):
        """Perform a single DQN training step and return (loss, violation_count)"""
        # Update orchestrator with new environment
        self.services = services
        self.num_services = len(services)
        self.nodes = all_nodes
        self.num_nodes = len(all_nodes)
        self.latency_matrix = latency

        # Train for one episode
        self.train(episodes=1)

        # Perform a single training step (if enough memory)
        loss = None
        if self._trained_agent is not None and hasattr(self._trained_agent, 'train'):
            loss = self._trained_agent.train()
            if loss is not None and hasattr(loss, 'item'):
                loss = float(loss.item())

        # Assignment after training step
        try:
            dqn_assignment = self.optimize(services)
        except RuntimeError:
            return (loss if loss is not None else 0.0), 1

        # Validate assignment: all node IDs must exist
        dqn_violations = []
        for svc in services:
            if svc.service_id in dqn_assignment:
                node_id = dqn_assignment[svc.service_id]
                node = next((n for n in all_nodes if n.node_id == node_id), None)
                if node:
                    cpu_req, ram_req, gpu_req, _, _ = self._service_demands(svc)
                    caps = node.metrics.get_capacity()
                    if cpu_req > caps.get("cpu", 0):
                        dqn_violations.append(f"CPU: {cpu_req} > {caps.get('cpu', 0)}")
                    if ram_req > caps.get("ram_gb", 0):
                        dqn_violations.append(f"RAM: {ram_req} > {caps.get('ram_gb', 0)}")
                    if gpu_req > caps.get("gpu_gb", 0):
                        dqn_violations.append(f"GPU: {gpu_req} > {caps.get('gpu_gb', 0)}")

        return (loss if loss is not None else 0.0), len(dqn_violations)
