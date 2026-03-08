"""训练/推理设备选择工具。

目标：
- 有 NVIDIA CUDA 时优先使用 CUDA；
- 在 Apple Silicon 上若支持 MPS，则使用 MPS；
- 否则回退到 CPU。

这样可以避免仅依赖第三方库对 `device="auto"` 的默认解释，
保证本项目在本地和服务器上都遵循同一套设备选择规则。
"""

from __future__ import annotations

from typing import Any


def resolve_device(requested: str | None) -> str:
    """把配置中的设备字段解析成最终可用设备。

    规则：
    - 若显式指定为 `cpu/cuda/mps`，则尊重用户配置；
    - 若配置为 `auto` 或为空，则按 `cuda -> mps -> cpu` 解析。
    """

    if requested is None:
        requested = "auto"
    requested = str(requested).lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def device_runtime_info() -> dict[str, Any]:
    """返回当前 PyTorch 设备探测结果，便于日志与审计。"""

    info: dict[str, Any] = {
        "requested": "auto",
        "resolved": "cpu",
        "cuda_available": False,
        "cuda_device_count": 0,
        "mps_available": False,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        info["resolved"] = resolve_device("auto")
    except ImportError:
        pass
    return info
