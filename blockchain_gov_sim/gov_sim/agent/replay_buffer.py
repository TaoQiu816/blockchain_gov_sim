"""DQN replay buffer。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ReplayBatch:
    state: torch.Tensor
    legal_mask: torch.Tensor
    action_idx: torch.Tensor
    reward: torch.Tensor
    cost: torch.Tensor
    next_state: torch.Tensor
    next_legal_mask: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    """固定容量循环经验池。"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.legal_mask = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.action_idx = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.cost = np.zeros(self.capacity, dtype=np.float32)
        self.next_state = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_legal_mask = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        state: np.ndarray,
        legal_mask: np.ndarray,
        action_idx: int,
        reward: float,
        cost: float,
        next_state: np.ndarray,
        next_legal_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.state[self.ptr] = np.asarray(state, dtype=np.float32)
        self.legal_mask[self.ptr] = np.asarray(legal_mask, dtype=np.float32)
        self.action_idx[self.ptr] = int(action_idx)
        self.reward[self.ptr] = float(reward)
        self.cost[self.ptr] = float(cost)
        self.next_state[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.next_legal_mask[self.ptr] = np.asarray(next_legal_mask, dtype=np.float32)
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> ReplayBatch:
        if self.size < int(batch_size):
            raise ValueError(f"Not enough samples: size={self.size}, requested={batch_size}")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        return ReplayBatch(
            state=torch.as_tensor(self.state[indices], device=target_device),
            legal_mask=torch.as_tensor(self.legal_mask[indices], device=target_device),
            action_idx=torch.as_tensor(self.action_idx[indices], dtype=torch.long, device=target_device),
            reward=torch.as_tensor(self.reward[indices], device=target_device),
            cost=torch.as_tensor(self.cost[indices], device=target_device),
            next_state=torch.as_tensor(self.next_state[indices], device=target_device),
            next_legal_mask=torch.as_tensor(self.next_legal_mask[indices], device=target_device),
            done=torch.as_tensor(self.done[indices], device=target_device),
        )
