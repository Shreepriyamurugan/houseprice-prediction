"""Tests for model training pipeline, DropColumns transformer, and MLflow logging."""

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from lifinity.config import get_project_root
from lifinity.models.train import DropColumns, build_pipeline_with_ablation, rmse_log


def test_drop_columns_transformer():
    df = pd.DataFrame(
        {
            "LotRatio": [0.1, 0.2],
            "IsRemodeled": [1, 0],
            "KeepMe": [10, 20],
        }
    )

    dropper = DropColumns(cols=["LotRatio", "NonExistentCol"])
    dropper.fit(df)
    out = dropper.transform(df)

    assert "LotRatio" not in out.columns
    assert "IsRemodeled" in out.columns
    assert "KeepMe" in out.columns
    assert list(out.columns) == ["IsRemodeled", "KeepMe"]


def test_drop_columns_pipeline_pickling(tmp_path):
    df = pd.DataFrame(
        {
            "OverallQual": [7, 6],
            "GrLivArea": [1500, 1200],
            "LotArea": [8000, 9000],
            "LotRatio": [0.18, 0.13],
            "SalePrice": [200000, 180000],
        }
    )
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    pipe = build_pipeline_with_ablation(Ridge(alpha=10.0), kind="linear", cols=["LotRatio"])
    pipe.fit(X, y)

    dump_path = tmp_path / "ablation_pipe.joblib"
    joblib.dump(pipe, dump_path)

    loaded_pipe = joblib.load(dump_path)
    preds = loaded_pipe.predict(X)

    assert len(preds) == len(X)
    assert (preds > 0).all()


def test_smoke_train_and_mlflow(tmp_path):
    """Smoke test: 200-row sample of train.parquet, 2-fold KFold, Ridge + LightGBM(50), MLflow pointed at tmp_path."""
    import mlflow
    from sklearn.model_selection import KFold, cross_val_score
    from lifinity.features.preprocessor import RMSE_LOG_SCORER, build_full_pipeline

    root = get_project_root()
    train_path = root / "data" / "processed" / "train.parquet"
    full_df = pd.read_parquet(train_path)

    sample_df = full_df.head(200).copy()
    X = sample_df.drop(columns=["SalePrice"], errors="ignore")
    y = sample_df["SalePrice"]

    # Point MLflow at tmp_path to avoid modifying real mlflow.db
    db_file = tmp_path / "test_smoke_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_file}")
    mlflow.set_experiment("smoke_test_exp")

    cv = KFold(n_splits=2, shuffle=True, random_state=42)

    # 1. Ridge
    pipe_ridge = build_full_pipeline(Ridge(alpha=10.0, random_state=42), kind="tree")
    scores_ridge = cross_val_score(pipe_ridge, X, y, cv=cv, scoring=RMSE_LOG_SCORER)
    rmse_ridge = -scores_ridge.mean()
    assert rmse_ridge < 0.25, f"Ridge smoke test RMSE {rmse_ridge:.4f} >= 0.25"

    # 2. LightGBM (50 trees)
    pipe_lgb = build_full_pipeline(LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), kind="tree")
    scores_lgb = cross_val_score(pipe_lgb, X, y, cv=cv, scoring=RMSE_LOG_SCORER)
    rmse_lgb = -scores_lgb.mean()
    assert rmse_lgb < 0.25, f"LightGBM smoke test RMSE {rmse_lgb:.4f} >= 0.25"
