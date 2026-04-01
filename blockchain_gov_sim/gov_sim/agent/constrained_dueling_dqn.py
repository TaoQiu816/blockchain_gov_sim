"""Constrained Double Dueling DQN。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gov_sim.agent.replay_buffer import ReplayBuffer, ReplayBatch


class DuelingQHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.value_mlp = nn.Linear(hidden_dim, 1)
        self.adv_mlp = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.value_mlp(hidden)
        advantage = self.adv_mlp(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DoubleDuelingNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.q_reward_head = DuelingQHead(hidden_dim=hidden_dim, action_dim=action_dim)
        self.q_cost_head = DuelingQHead(hidden_dim=hidden_dim, action_dim=action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.q_reward_head(hidden), self.q_cost_head(hidden)


class ConstrainedDoubleDuelingDQN:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        replay_buffer: ReplayBuffer | None = None,
        cost_limit: float = 0.10,
        lr: float | None = None,
        target_update_period: int | None = None,
        warmup_steps: int | None = None,
        train_freq: int | None = None,
        epsilon_decay_steps: int | None = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.lr = float(1.0e-4 if lr is None else lr)
        self.batch_size = 128
        self.buffer_size = 200000
        self.gamma = 0.99
        self.beta_c = 1.0
        self.lambda_init = 0.1
        self.eta_lambda = 1.0e-3
        self.target_update_period = int(1000 if target_update_period is None else target_update_period)
        self.warmup_steps = int(5000 if warmup_steps is None else warmup_steps)
        self.train_freq = int(4 if train_freq is None else train_freq)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = int(50000 if epsilon_decay_steps is None else epsilon_decay_steps)
        self.gradient_clip = 10.0
        self.cost_window_len = 200
        self.cost_limit = float(cost_limit)

        self.online_net = DoubleDuelingNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DoubleDuelingNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.replay_buffer = replay_buffer or ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)
        self.lambda_value = float(self.lambda_init)
        self.cost_window: deque[float] = deque(maxlen=self.cost_window_len)
        self.global_step = 0
        self.epsilon = float(self.epsilon_start)

    def _epsilon_by_step(self, step: int) -> float:
        progress = min(max(int(step), 0), self.epsilon_decay_steps) / max(self.epsilon_decay_steps, 1)
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress)

    def _masked_score(self, q_r: torch.Tensor, q_c: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        score = q_r - float(self.lambda_value) * q_c
        return score.masked_fill(legal_mask <= 0, float("-inf"))

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray, deterministic: bool = False) -> int:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(np.asarray(legal_mask, dtype=np.float32), device=self.device).unsqueeze(0)
        q_r, q_c = self.online_net(obs_tensor)
        score = self._masked_score(q_r, q_c, mask_tensor).squeeze(0)
        legal_indices = torch.nonzero(mask_tensor.squeeze(0) > 0, as_tuple=False).flatten()
        if legal_indices.numel() == 0:
            return 0
        if (not deterministic) and np.random.random() < self.epsilon:
            choice = int(legal_indices[torch.randint(0, legal_indices.numel(), (1,), device=self.device)].item())
            return choice
        return int(torch.argmax(score).item())

    def store_transition(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        reward: float,
        cost: float,
        next_state: np.ndarray,
        next_legal_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(
            state=state,
            legal_mask=legal_mask,
            action_idx=int(action),
            reward=float(reward),
            cost=float(cost),
            next_state=next_state,
            next_legal_mask=next_legal_mask,
            done=bool(done),
        )
        self.cost_window.append(float(cost))
        self.global_step += 1
        self.epsilon = self._epsilon_by_step(self.global_step)

    @torch.no_grad()
    def _compute_targets(self, batch: ReplayBatch) -> tuple[torch.Tensor, torch.Tensor]:
        next_q_r_online, next_q_c_online = self.online_net(batch.next_state)
        next_score = self._masked_score(next_q_r_online, next_q_c_online, batch.next_legal_mask)
        a_next = torch.argmax(next_score, dim=1, keepdim=True)
        next_q_r_target, next_q_c_target = self.target_net(batch.next_state)
        next_r = next_q_r_target.gather(1, a_next).squeeze(1)
        next_c = next_q_c_target.gather(1, a_next).squeeze(1)
        y_r = batch.reward + self.gamma * (1.0 - batch.done) * next_r
        y_c = batch.cost + self.gamma * (1.0 - batch.done) * next_c
        return y_r, y_c

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _update_lambda(self) -> None:
        if not self.cost_window:
            return
        mean_cost = float(np.mean(self.cost_window))
        self.lambda_value = float(max(0.0, self.lambda_value + self.eta_lambda * (mean_cost - self.cost_limit)))

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        y_r, y_c = self._compute_targets(batch)
        q_r, q_c = self.online_net(batch.state)
        chosen_q_r = q_r.gather(1, batch.action_idx.unsqueeze(1)).squeeze(1)
        chosen_q_c = q_c.gather(1, batch.action_idx.unsqueeze(1)).squeeze(1)
        loss_r = torch.mean((chosen_q_r - y_r) ** 2)
        loss_c = torch.mean((chosen_q_c - y_c) ** 2)
        loss = loss_r + self.beta_c * loss_c
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimizer.step()
        if self.global_step % self.target_update_period == 0:
            self._sync_target()
        self._update_lambda()
        return {
            "loss_r": float(loss_r.item()),
            "loss_c": float(loss_c.item()),
            "loss": float(loss.item()),
            "lambda": float(self.lambda_value),
            "epsilon": float(self.epsilon),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lambda_value": self.lambda_value,
                "global_step": self.global_step,
                "epsilon": self.epsilon,
                "cost_limit": self.cost_limit,
                "lr": self.lr,
                "target_update_period": self.target_update_period,
                "warmup_steps": self.warmup_steps,
                "train_freq": self.train_freq,
                "epsilon_decay_steps": self.epsilon_decay_steps,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ConstrainedDoubleDuelingDQN":
        payload = torch.load(Path(path), map_location=device)
        agent = cls(
            state_dim=int(payload["state_dim"]),
            action_dim=int(payload["action_dim"]),
            device=device,
            cost_limit=float(payload.get("cost_limit", 0.10)),
            lr=float(payload.get("lr", 1.0e-4)),
            target_update_period=int(payload.get("target_update_period", 1000)),
            warmup_steps=int(payload.get("warmup_steps", 5000)),
            train_freq=int(payload.get("train_freq", 4)),
            epsilon_decay_steps=int(payload.get("epsilon_decay_steps", 50000)),
        )
        agent.online_net.load_state_dict(payload["online_net"])
        agent.target_net.load_state_dict(payload["target_net"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.lambda_value = float(payload.get("lambda_value", agent.lambda_init))
        agent.global_step = int(payload.get("global_step", 0))
        agent.epsilon = float(payload.get("epsilon", agent._epsilon_by_step(agent.global_step)))
        return agent
