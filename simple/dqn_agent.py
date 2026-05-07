from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    target_update_interval: int = 100
    tau: float = 0.005  # polyak averaging coefficient for soft target updates
    hidden_size: int = 64


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: DQNConfig | None = None,
        seed: int = 0,
        device: str | None = None,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.config = config or DQNConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.policy_net = QNetwork(observation_size, action_size, self.config.hidden_size).to(self.device)
        self.target_net = QNetwork(observation_size, action_size, self.config.hidden_size).to(self.device)
        self.update_target_network()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.training_steps = 0

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(torch.argmax(q_values, dim=1).item())

    def optimize(self, replay_buffer) -> float | None:
        if len(replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(self.config.batch_size)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double DQN: policy net selects action, target net evaluates it
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + self.config.gamma * next_q_values * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.training_steps += 1
        self.update_target_network()

        return float(loss.item())

    def update_target_network(self) -> None:
        for p_target, p_policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
            p_target.data.copy_(self.config.tau * p_policy.data + (1.0 - self.config.tau) * p_target.data)
