"""全局常量定义。

集中管理动作集合、信誉维度和全局数值常量，避免“魔法常数”散落在代码中。
"""

from __future__ import annotations

from typing import Final

REPUTATION_DIMS: Final[tuple[str, ...]] = ("svc", "con", "rec", "stab")
M_CHOICES: Final[tuple[int, ...]] = (7, 9, 11, 13, 15)
B_CHOICES: Final[tuple[int, ...]] = (128, 256, 384, 512)
TAU_CHOICES: Final[tuple[int, ...]] = (20, 40, 60, 80)
THETA_CHOICES: Final[tuple[float, ...]] = (0.40, 0.50, 0.60, 0.70, 0.80)
ACTION_DIM: Final[int] = len(M_CHOICES) * len(B_CHOICES) * len(TAU_CHOICES) * len(THETA_CHOICES)
EPS: Final[float] = 1.0e-8
