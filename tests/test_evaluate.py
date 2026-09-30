"""Tests for model evaluation, persistence, and report structure."""

import json
import joblib
import pandas as pd
import pytest

pytestmark = [pytest.mark.requires_data, pytest.mark.requires_model]

from lifinity.config import get_project_root


def test_model_joblib_loads_and_predicts():
    root = get_project_root()
    model_path = root / "models" / "model.joblib"
    if not model_path.exists():
        pytest.skip("models/model.joblib does not exist yet.")

    model = joblib.load(model_path)
    val_df = pd.read_parquet(root / "data" / "processed" / "val.parquet")
    X_val = val_df.drop(columns=["SalePrice"], errors="ignore").head(5)

    preds = model.predict(X_val)
    assert len(preds) == 5
    assert (preds > 0).all()


def test_metrics_json_keys():
    root = get_project_root()
    metrics_path = root / "models" / "metrics.json"
    assert metrics_path.exists(), "models/metrics.json does not exist."

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required_keys = {"rmse_log", "mae", "mape", "r2_log", "bands", "reference_lasso"}
    for k in required_keys:
        assert k in metrics, f"Missing key '{k}' in metrics.json"


def test_input_defaults_coverage():
    root = get_project_root()
    defaults_path = root / "models" / "input_defaults.json"
    assert defaults_path.exists(), "models/input_defaults.json does not exist."

    with open(defaults_path, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    raw_train = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    expected_cols = set(c for c in raw_train.columns if c not in ("SalePrice", "Id"))

    assert set(defaults.keys()) == expected_cols, "input_defaults.json does not match expected raw input columns."


def test_only_split_and_evaluate_contain_test_parquet():
    root = get_project_root()
    src_dir = root / "src"

    matching_files = []
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if "test.parquet" in content:
            rel_path = py_file.relative_to(src_dir).as_posix()
            matching_files.append(rel_path)

    expected = {"lifinity/data/split.py", "lifinity/models/evaluate.py"}
    assert set(matching_files) == expected, f"Unexpected files containing test.parquet: {matching_files}"


def test_report_final_model_heading_single_occurrence():
    root = get_project_root()
    report_path = root / "reports" / "model_selection.md"
    assert report_path.exists(), "reports/model_selection.md does not exist."

    content = report_path.read_text(encoding="utf-8")
    count = content.count("## Final model")
    assert count == 1, f"Expected exactly one '## Final model' heading, found {count}"
