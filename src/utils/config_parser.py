"""YAML Configuration parser with dictionary-like access and path resolution."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If parsing fails.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    return data


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges multiple configuration dictionaries.

    Args:
        *configs: Variable number of config dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {}
    for cfg in configs:
        for key, value in cfg.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged
