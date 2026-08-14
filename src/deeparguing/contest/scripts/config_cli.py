"""Shared CLI/YAML-config resolution helpers for the ``contest/scripts``
entry points (``contest_all.py``, ``contest_all_irrelevance.py``,
``run_finetune.py``): each takes a ``--config`` YAML file of hyperparameters
and paths, with any CLI flag overriding the corresponding config value.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        print(
            f"Warning: config file {config_path} not found -- proceeding with "
            "CLI flags and library defaults only."
        )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolved(cli_value: Any, config: dict[str, Any], key: str, fallback: Any) -> Any:
    """CLI flag (if given) overrides the config file's value, which
    overrides ``fallback``. A ``null``/absent key in the config file both
    fall through to ``fallback``."""
    if cli_value is not None:
        return cli_value
    if config.get(key) is not None:
        return config[key]
    return fallback


def required(cli_value: Any, config: dict[str, Any], key: str, config_path: str) -> Any:
    value = cli_value if cli_value is not None else config.get(key)
    if value is None:
        raise ValueError(
            f"'{key}' was not given on the command line and is not set in "
            f"{config_path} -- add it there or pass --{key.replace('_', '-')}."
        )
    return value
