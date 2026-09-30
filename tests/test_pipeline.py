"""Tests for DVC pipeline configuration (dvc.yaml)."""

from pathlib import Path
import yaml

from lifinity.config import get_project_root


def test_dvc_yaml_structure():
    root = get_project_root()
    dvc_yaml_path = root / "dvc.yaml"
    assert dvc_yaml_path.exists(), "dvc.yaml does not exist"

    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "stages" in data, "dvc.yaml missing 'stages' key"
    stages = data["stages"]

    # Check stages exist
    expected_stages = ["split", "compare", "tune", "evaluate", "report"]
    for stage_name in expected_stages:
        assert stage_name in stages, f"Stage {stage_name} missing from dvc.yaml"

    # compare and tune have frozen: true
    assert stages["compare"].get("frozen") is True, "compare stage must have frozen: true"
    assert stages["tune"].get("frozen") is True, "tune stage must have frozen: true"

    # evaluate outs include models/model.joblib and metrics include models/metrics.json
    evaluate_stage = stages["evaluate"]

    # Extract outs list (handle both simple string list and dict options)
    outs = evaluate_stage.get("outs", [])
    out_paths = []
    for item in outs:
        if isinstance(item, str):
            out_paths.append(item)
        elif isinstance(item, dict):
            out_paths.extend(item.keys())

    assert "models/model.joblib" in out_paths, "models/model.joblib missing from evaluate outs"

    # Extract metrics list
    metrics = evaluate_stage.get("metrics", [])
    metric_paths = []
    for item in metrics:
        if isinstance(item, str):
            metric_paths.append(item)
        elif isinstance(item, dict):
            metric_paths.extend(item.keys())

    assert "models/metrics.json" in metric_paths, "models/metrics.json missing from evaluate metrics"
