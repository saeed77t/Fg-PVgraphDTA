"""Configuration loading and path resolution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")
    return data


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_required(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for key in dotted_key.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Required configuration key is missing: {dotted_key}")
        current = current[key]
    return current


def resolve_path(value: str | Path, project_root: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = Path.cwd() if project_root is None else Path(project_root)
    return (root / path).resolve()
