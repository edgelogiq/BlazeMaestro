from __future__ import annotations

from typing import List, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, device: torch.device):
        super().__init__()
        # Deeper, wider network with residual-like skip for large action spaces (21 nodes)
        # Increased capacity: 256→512→256→256 with better feature extraction
        self.net = nn.Sequential(
            nn.Linear(s_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, a_dim),
        )
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        lr: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        batch_size: int,
        max_memory: int,
        update_steps: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.model = DQNNetwork(s_dim, a_dim, device)
        self.target = DQNNetwork(s_dim, a_dim, device)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = eps_start
        self.batch = batch_size
        self.memory: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.max_memory = max_memory
        self.update_steps = update_steps
        self.step = 0
        self._eps_end = eps_end
        self._eps_decay = eps_decay

    def select(self, state: np.ndarray, valid_actions: List[int], temperature: float = 0.1) -> int:
        """
        Select action using temperature-scaled softmax for better exploration
        even during inference.
        """
        # Handle empty valid_actions (infeasible state)
        if not valid_actions:
            return 0  # Fallback, will trigger penalty in step_env

        self.model.eval()  # Set to evaluation mode for BatchNorm
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)
            num_actions = q_values.size(0)

            # Convert valid_actions list to boolean mask
            valid_mask = torch.zeros(num_actions, dtype=torch.bool, device=self.device)
            for action_idx in valid_actions:
                if action_idx < num_actions:
                    valid_mask[action_idx] = True

            # Apply valid action mask
            masked_q_values = q_values.clone()
            masked_q_values[~valid_mask] = float('-inf')

            # Use temperature-based softmax for action selection
            # Higher temperature = more exploration, lower = more exploitation
            if temperature > 0:
                probabilities = torch.softmax(masked_q_values / temperature, dim=0)
                # Sample from the probability distribution
                action = torch.multinomial(probabilities, 1).item()
            else:
                # Pure greedy selection (original behavior)
                action = masked_q_values.argmax().item()

        return action

    def store(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.memory.append(transition)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch:
            return
        self.model.train()  # Ensure training mode for BatchNorm
        batch = random.sample(self.memory, self.batch)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        q_vals = self.model(states).gather(1, actions)
        with torch.no_grad():
            self.target.eval()
            next_q = self.target(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_vals, target)
        self.opt.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.step += 1
        if self.step % self.update_steps == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.epsilon = max(self._eps_end, self.epsilon * self._eps_decay)
        return loss
