"""全局常量定义。"""

from __future__ import annotations

from typing import Final

REPUTATION_DIMS: Final[tuple[str, ...]] = ("svc", "con", "rec", "stab")
M_CHOICES: Final[tuple[int, ...]] = (5, 7, 9)
RHO_M_CHOICES: Final[tuple[float, ...]] = tuple(choice / 27.0 for choice in M_CHOICES)
B_CHOICES: Final[tuple[int, ...]] = (256, 320, 384, 448, 512)
TAU_CHOICES: Final[tuple[int, ...]] = (40, 60, 80, 100)
THETA_CHOICES: Final[tuple[float, ...]] = (0.45, 0.50, 0.55, 0.60)
ACTION_DIM: Final[int] = len(RHO_M_CHOICES) * len(B_CHOICES) * len(TAU_CHOICES) * len(THETA_CHOICES)
M_MIN: Final[int] = 3
TR_MIN: Final[float] = 0.4
H_MIN: Final[float] = 0.65
H_WARN: Final[float] = 0.75
EPS: Final[float] = 1.0e-8
