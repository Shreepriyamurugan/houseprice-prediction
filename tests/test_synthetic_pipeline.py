"""Synthetic dataset pipeline unit and integration tests (no markers)."""

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lifinity.features.engineering import build_cleaning_pipeline
from lifinity.features.preprocessor import build_full_pipeline, rmse_log
from lifinity.models.ensemble import LogBlendRegressor
from tests.synthetic import make_synthetic_ames


def test_synthetic_cleaning_chain_zero_nan():
    """Verify that domain cleaning and feature engineering produce zero NaNs."""
    df = make_synthetic_ames(n_samples=300, seed=42)
    X = df.drop(columns=["SalePrice"], errors="ignore")

    cleaning_pipe = build_cleaning_pipeline()
    X_clean = cleaning_pipe.fit_transform(X)

    assert isinstance(X_clean, pd.DataFrame)
    # Verify zero NaNs in engineered/cleaned DataFrame
    nan_count = X_clean.isna().sum().sum()
    assert nan_count == 0, f"Cleaned DataFrame contains {nan_count} NaNs"


def test_synthetic_ridge_and_lgbm_pipelines():
    """Verify that Ridge (linear) and LGBM (tree) pipelines fit 240 rows, predict 60, with RMSE(log) < 0.5."""
    df = make_synthetic_ames(n_samples=300, seed=42)
    train_df = df.iloc[:240].copy()
    test_df = df.iloc[240:].copy()

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]

    X_test = test_df.drop(columns=["SalePrice"], errors="ignore")
    y_test = test_df["SalePrice"]

    ridge_pipe = build_full_pipeline(Ridge(alpha=1.0), kind="linear")
    ridge_pipe.fit(X_train, y_train)
    ridge_preds = ridge_pipe.predict(X_test)

    assert np.all(np.isfinite(ridge_preds)), "Ridge predictions contain non-finite values"
    assert np.all(ridge_preds > 0), "Ridge predictions contain non-positive values"
    ridge_rmse = rmse_log(y_test, ridge_preds)
    assert ridge_rmse < 0.5, f"Ridge RMSE(log) {ridge_rmse:.4f} >= 0.5"

    lgbm_pipe = build_full_pipeline(LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), kind="tree")
    lgbm_pipe.fit(X_train, y_train)
    lgbm_preds = lgbm_pipe.predict(X_test)

    assert np.all(np.isfinite(lgbm_preds)), "LGBM predictions contain non-finite values"
    assert np.all(lgbm_preds > 0), "LGBM predictions contain non-positive values"
    lgbm_rmse = rmse_log(y_test, lgbm_preds)
    assert lgbm_rmse < 0.5, f"LGBM RMSE(log) {lgbm_rmse:.4f} >= 0.5"


def test_synthetic_log_blend_regressor(tmp_path):
    """Verify LogBlendRegressor fits with 0.5/0.5 weights, predicts, pickles, and reloads identically."""
    df = make_synthetic_ames(n_samples=300, seed=42)
    train_df = df.iloc[:240].copy()
    test_df = df.iloc[240:].copy()

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]
    X_test = test_df.drop(columns=["SalePrice"], errors="ignore")

    ridge_pipe = build_full_pipeline(Ridge(alpha=1.0), kind="linear")
    lgbm_pipe = build_full_pipeline(LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), kind="tree")

    estimators = [("Ridge", ridge_pipe), ("LGBM", lgbm_pipe)]
    blend = LogBlendRegressor(estimators=estimators, weights=[0.5, 0.5])
    blend.fit(X_train, y_train)

    preds_original = blend.predict(X_test)
    assert np.all(np.isfinite(preds_original))
    assert np.all(preds_original > 0)

    model_file = tmp_path / "blend.joblib"
    joblib.dump(blend, model_file)

    reloaded_blend = joblib.load(model_file)
    preds_reloaded = reloaded_blend.predict(X_test)

    np.testing.assert_allclose(
        preds_original,
        preds_reloaded,
        rtol=1e-5,
        err_msg="Reloaded LogBlendRegressor predictions differ from original",
    )


def test_synthetic_unseen_neighborhood_does_not_crash():
    """Verify that predicting on a record with an unseen Neighborhood does not crash."""
    df = make_synthetic_ames(n_samples=300, seed=42)
    train_df = df.iloc[:240].copy()
    test_df = df.iloc[240:].copy()

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]

    ridge_pipe = build_full_pipeline(Ridge(alpha=1.0), kind="linear")
    ridge_pipe.fit(X_train, y_train)

    lgbm_pipe = build_full_pipeline(LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), kind="tree")
    lgbm_pipe.fit(X_train, y_train)

    blend = LogBlendRegressor(estimators=[("Ridge", ridge_pipe), ("LGBM", lgbm_pipe)], weights=[0.5, 0.5])
    blend.fit(X_train, y_train)

    # Modify neighborhood of test row to an unseen value
    X_unseen = test_df.drop(columns=["SalePrice"], errors="ignore").copy()
    X_unseen.iloc[0, X_unseen.columns.get_loc("Neighborhood")] = "UnseenLoc"

    pred_ridge = ridge_pipe.predict(X_unseen.iloc[:1])
    assert np.isfinite(pred_ridge[0]) and pred_ridge[0] > 0

    pred_blend = blend.predict(X_unseen.iloc[:1])
    assert np.isfinite(pred_blend[0]) and pred_blend[0] > 0
