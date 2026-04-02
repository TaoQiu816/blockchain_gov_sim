"""支持主方法与 reward-only 变体的 DQN agent。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gov_sim.agent.replay_buffer import ReplayBatch, ReplayBuffer


AGENT_VARIANT_PROPOSED = "proposed"
AGENT_VARIANT_VANILLA = "vanilla_dqn"
AGENT_VARIANT_NO_CONSTRAINT = "no_constraint_dueling_double_dqn"

AGENT_VARIANTS = (
    AGENT_VARIANT_PROPOSED,
    AGENT_VARIANT_VANILLA,
    AGENT_VARIANT_NO_CONSTRAINT,
)


class DuelingQHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.value_mlp = nn.Linear(hidden_dim, 1)
        self.adv_mlp = nn.Linear(hidden_dim, action_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.value_mlp(hidden)
        advantage = self.adv_mlp(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for _ in range(int(num_hidden_layers)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.model = nn.Sequential(*layers)
        self.output_dim = int(prev_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class DoubleDuelingNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 2) -> None:
        super().__init__()
        self.backbone = MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
        self.q_reward_head = DuelingQHead(hidden_dim=self.backbone.output_dim, action_dim=action_dim)
        self.q_cost_head = DuelingQHead(hidden_dim=self.backbone.output_dim, action_dim=action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.q_reward_head(hidden), self.q_cost_head(hidden)


class RewardOnlyDoubleDuelingNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 2) -> None:
        super().__init__()
        self.backbone = MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
        self.q_head = DuelingQHead(hidden_dim=self.backbone.output_dim, action_dim=action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(obs)
        return self.q_head(hidden)


class VanillaQNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 2) -> None:
        super().__init__()
        self.backbone = MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
        self.q_head = nn.Linear(self.backbone.output_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(obs)
        return self.q_head(hidden)


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
        agent_variant: str = AGENT_VARIANT_PROPOSED,
    ) -> None:
        if agent_variant not in AGENT_VARIANTS:
            raise ValueError(f"Unknown agent_variant: {agent_variant}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.agent_variant = str(agent_variant)
        self.uses_cost_head = self.agent_variant == AGENT_VARIANT_PROPOSED
        self.uses_dueling = self.agent_variant != AGENT_VARIANT_VANILLA
        self.uses_double = self.agent_variant != AGENT_VARIANT_VANILLA

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

        self.online_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.replay_buffer = replay_buffer or ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)
        self.lambda_value = float(self.lambda_init if self.uses_cost_head else 0.0)
        self.cost_window: deque[float] = deque(maxlen=self.cost_window_len)
        self.global_step = 0
        self.epsilon = float(self.epsilon_start)

    def _build_network(self) -> nn.Module:
        if self.agent_variant == AGENT_VARIANT_PROPOSED:
            return DoubleDuelingNetwork(self.state_dim, self.action_dim)
        if self.agent_variant == AGENT_VARIANT_NO_CONSTRAINT:
            return RewardOnlyDoubleDuelingNetwork(self.state_dim, self.action_dim)
        return VanillaQNetwork(self.state_dim, self.action_dim)

    def _epsilon_by_step(self, step: int) -> float:
        progress = min(max(int(step), 0), self.epsilon_decay_steps) / max(self.epsilon_decay_steps, 1)
        return float(self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress)

    def _mask_q(self, q_value: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        return q_value.masked_fill(legal_mask <= 0, float("-inf"))

    def _split_output(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.uses_cost_head:
            reward_q, cost_q = output
            return reward_q, cost_q
        return output, None

    def _masked_score(self, q_r: torch.Tensor, q_c: torch.Tensor | None, legal_mask: torch.Tensor) -> torch.Tensor:
        if self.uses_cost_head:
            score = q_r - float(self.lambda_value) * q_c
        else:
            score = q_r
        return self._mask_q(score, legal_mask)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray, deterministic: bool = False) -> int:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(np.asarray(legal_mask, dtype=np.float32), device=self.device).unsqueeze(0)
        q_r, q_c = self._split_output(self.online_net(obs_tensor))
        score = self._masked_score(q_r, q_c, mask_tensor).squeeze(0)
        legal_indices = torch.nonzero(mask_tensor.squeeze(0) > 0, as_tuple=False).flatten()
        if legal_indices.numel() == 0:
            return 0
        if (not deterministic) and np.random.random() < self.epsilon:
            sampled = torch.randint(0, legal_indices.numel(), (1,), device=self.device)
            return int(legal_indices[sampled].item())
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
    def _compute_targets(self, batch: ReplayBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.uses_cost_head:
            next_q_r_online, next_q_c_online = self.online_net(batch.next_state)
            next_score = self._masked_score(next_q_r_online, next_q_c_online, batch.next_legal_mask)
            a_next = torch.argmax(next_score, dim=1, keepdim=True)
            next_q_r_target, next_q_c_target = self.target_net(batch.next_state)
            next_r = next_q_r_target.gather(1, a_next).squeeze(1)
            next_c = next_q_c_target.gather(1, a_next).squeeze(1)
            y_r = batch.reward + self.gamma * (1.0 - batch.done) * next_r
            y_c = batch.cost + self.gamma * (1.0 - batch.done) * next_c
            return y_r, y_c

        if self.uses_double:
            next_q_online = self.online_net(batch.next_state)
            next_q_online = self._mask_q(next_q_online, batch.next_legal_mask)
            a_next = torch.argmax(next_q_online, dim=1, keepdim=True)
            next_q_target = self.target_net(batch.next_state)
            next_r = next_q_target.gather(1, a_next).squeeze(1)
        else:
            next_q_target = self.target_net(batch.next_state)
            next_q_target = self._mask_q(next_q_target, batch.next_legal_mask)
            next_r = torch.max(next_q_target, dim=1).values

        y_r = batch.reward + self.gamma * (1.0 - batch.done) * next_r
        return y_r, None

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _update_lambda(self) -> None:
        if (not self.uses_cost_head) or (not self.cost_window):
            self.lambda_value = 0.0
            return
        mean_cost = float(np.mean(self.cost_window))
        self.lambda_value = float(max(0.0, self.lambda_value + self.eta_lambda * (mean_cost - self.cost_limit)))

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size, device=self.device)
        y_r, y_c = self._compute_targets(batch)
        output = self.online_net(batch.state)
        q_r, q_c = self._split_output(output)
        chosen_q_r = q_r.gather(1, batch.action_idx.unsqueeze(1)).squeeze(1)
        loss_r = torch.mean((chosen_q_r - y_r) ** 2)

        if self.uses_cost_head:
            chosen_q_c = q_c.gather(1, batch.action_idx.unsqueeze(1)).squeeze(1)
            loss_c = torch.mean((chosen_q_c - y_c) ** 2)
            loss = loss_r + self.beta_c * loss_c
        else:
            loss_c = torch.zeros((), device=self.device)
            loss = loss_r

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
                "agent_variant": self.agent_variant,
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
            agent_variant=str(payload.get("agent_variant", AGENT_VARIANT_PROPOSED)),
        )
        agent.online_net.load_state_dict(payload["online_net"])
        agent.target_net.load_state_dict(payload["target_net"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.lambda_value = float(payload.get("lambda_value", agent.lambda_init if agent.uses_cost_head else 0.0))
        agent.global_step = int(payload.get("global_step", 0))
        agent.epsilon = float(payload.get("epsilon", agent._epsilon_by_step(agent.global_step)))
        return agent


__all__ = [
    "AGENT_VARIANT_NO_CONSTRAINT",
    "AGENT_VARIANT_PROPOSED",
    "AGENT_VARIANT_VANILLA",
    "AGENT_VARIANTS",
    "ConstrainedDoubleDuelingDQN",
]
