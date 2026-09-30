"""Tests for tuning, LogBlendRegressor, and model selection report generator."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Lasso, Ridge

pytestmark = pytest.mark.requires_data

from lifinity.config import get_project_root
from lifinity.features.preprocessor import build_full_pipeline
from lifinity.models.ensemble import LogBlendRegressor


def test_log_blend_regressor_unit(tmp_path):
    df = pd.DataFrame(
        {
            "OverallQual": [7, 6, 8],
            "GrLivArea": [1500, 1200, 1800],
            "LotArea": [8000, 9000, 10000],
            "SalePrice": [200000, 180000, 250000],
        }
    )
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    p1 = build_full_pipeline(Ridge(alpha=10.0), kind="linear")
    p2 = build_full_pipeline(Lasso(alpha=0.001), kind="linear")

    p1.fit(X, y)
    p2.fit(X, y)

    pred1 = p1.predict(X)

    # Weights [1.0, 0.0] should equal p1 predictions
    blend_p1 = LogBlendRegressor(estimators=[("p1", p1), ("p2", p2)], weights=[1.0, 0.0])
    blend_p1.fit(X, y)
    pred_blend = blend_p1.predict(X)

    np.testing.assert_allclose(pred1, pred_blend, rtol=1e-5)

    # Test pickling with joblib
    dump_file = tmp_path / "blend.joblib"
    joblib.dump(blend_p1, dump_file)
    loaded_blend = joblib.load(dump_file)
    pred_loaded = loaded_blend.predict(X)

    np.testing.assert_allclose(pred1, pred_loaded, rtol=1e-5)


def test_optuna_smoke_tune(tmp_path):
    import mlflow
    import optuna
    from sklearn.model_selection import KFold, cross_validate
    from lifinity.features.preprocessor import RMSE_LOG_SCORER

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    root = get_project_root()
    train_df = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    sample_df = train_df.head(100).copy()

    X = sample_df.drop(columns=["SalePrice"], errors="ignore")
    y = sample_df["SalePrice"]

    db_file = tmp_path / "test_tune_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_file}")
    mlflow.set_experiment("test_tune_exp")

    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    def obj(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e-1, log=True)
        pipe = build_full_pipeline(Lasso(alpha=alpha, max_iter=5000), kind="linear")
        res = cross_validate(pipe, X, y, cv=cv, scoring=RMSE_LOG_SCORER)
        return float(-res["test_score"].mean())

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=3)

    assert study.best_params["alpha"] > 0
    assert study.best_value < 0.30


def test_no_test_parquet_in_code():
    root = get_project_root()
    files_to_check = [
        root / "src" / "lifinity" / "models" / "tune.py",
        root / "src" / "lifinity" / "models" / "ensemble.py",
        root / "src" / "lifinity" / "models" / "report.py",
    ]

    for filepath in files_to_check:
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8")
            assert "test.parquet" not in content, f"Forbidden 'test.parquet' found in {filepath}"
