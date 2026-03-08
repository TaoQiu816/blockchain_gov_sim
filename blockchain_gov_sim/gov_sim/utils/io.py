"""配置与结果文件 I/O 工具。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 Path 对象。"""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def read_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，要求根节点为 mapping。"""
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归覆盖字典配置。"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(*paths: str | Path) -> dict[str, Any]:
    """按顺序加载并合并多个 YAML 配置。"""
    config: dict[str, Any] = {}
    for path in paths:
        if path is None:
            continue
        config = deep_update(config, read_yaml(path))
    return config


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """写出带缩进的 UTF-8 JSON。"""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
