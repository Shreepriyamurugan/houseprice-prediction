from pathlib import Path
from typing import Any, Union
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "params.yaml").exists():
            return parent
    return PROJECT_ROOT


def load_params(path: Union[str, Path] = "params.yaml") -> dict[str, Any]:
    """Load project parameters from a YAML file, resolved relative to the project root.

    Parameters
    ----------
    path : str or Path, default="params.yaml"
        Path to the YAML parameters file. If relative, resolved against the project root.

    Returns
    -------
    dict
        Parsed parameter dictionary.
    """
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = get_project_root() / target_path

    with open(target_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return params if isinstance(params, dict) else {}
