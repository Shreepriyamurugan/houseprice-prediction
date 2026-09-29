"""Tests for preprocessor module, SkewCorrector, build_preprocessor, and build_full_pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from lifinity.config import get_project_root
from lifinity.features.engineering import build_cleaning_pipeline
from lifinity.features.preprocessor import (
    ORDINAL_CATEGORIES,
    SkewCorrector,
    build_full_pipeline,
    build_preprocessor,
)


@pytest.fixture
def train_and_holdout_clean():
    root = get_project_root()
    train_path = root / "data" / "processed" / "train.parquet"
    holdout_path = root / "data" / "processed" / "holdout.parquet"

    train_df = pd.read_parquet(train_path)
    holdout_df = pd.read_parquet(holdout_path)

    cleaning_pipe = build_cleaning_pipeline()
    X_train_clean = cleaning_pipe.fit_transform(train_df.drop(columns=["SalePrice"], errors="ignore"))
    X_holdout_clean = cleaning_pipe.transform(holdout_df.drop(columns=["SalePrice"], errors="ignore"))
    
    y_train = train_df["SalePrice"]
    y_holdout = holdout_df["SalePrice"] if "SalePrice" in holdout_df.columns else None

    return X_train_clean, X_holdout_clean, y_train, y_holdout, train_df, holdout_df


@pytest.mark.parametrize("kind", ["tree", "linear"])
def test_preprocessor_kinds_zero_nan_inf_numeric_identical(kind, train_and_holdout_clean):
    X_train_clean, X_holdout_clean, _, _, _, _ = train_and_holdout_clean

    prep = build_preprocessor(kind=kind)
    X_train_prep = prep.fit_transform(X_train_clean)
    X_holdout_prep = prep.transform(X_holdout_clean)

    # zero NaN
    assert X_train_prep.isnull().sum().sum() == 0, f"{kind} train output has NaN"
    assert X_holdout_prep.isnull().sum().sum() == 0, f"{kind} holdout output has NaN"

    # zero inf
    assert not np.isinf(X_train_prep.values).any(), f"{kind} train output has inf"
    assert not np.isinf(X_holdout_prep.values).any(), f"{kind} holdout output has inf"

    # all numeric
    for col in X_train_prep.columns:
        assert pd.api.types.is_numeric_dtype(X_train_prep[col]), f"{col} in {kind} train is not numeric"
    for col in X_holdout_prep.columns:
        assert pd.api.types.is_numeric_dtype(X_holdout_prep[col]), f"{col} in {kind} holdout is not numeric"

    # identical columns
    assert list(X_train_prep.columns) == list(X_holdout_prep.columns), f"{kind} train and holdout columns differ"


def test_exterqual_ex_encodes_higher_than_gd(train_and_holdout_clean):
    X_train_clean, _, _, _, _, _ = train_and_holdout_clean

    prep = build_preprocessor(kind="tree")
    X_prep = prep.fit_transform(X_train_clean)

    # ExterQual is mapped to numbers where Ex > Gd
    ex_mask = X_train_clean["ExterQual"] == "Ex"
    gd_mask = X_train_clean["ExterQual"] == "Gd"

    ex_val = X_prep.loc[ex_mask, "ExterQual"].iloc[0]
    gd_val = X_prep.loc[gd_mask, "ExterQual"].iloc[0]

    assert ex_val > gd_val, f"Expected Ex ({ex_val}) > Gd ({gd_val})"


def test_unknown_category_does_not_crash(train_and_holdout_clean):
    X_train_clean, X_holdout_clean, _, _, _, _ = train_and_holdout_clean

    X_holdout_mod = X_holdout_clean.copy()
    X_holdout_mod.loc[X_holdout_mod.index[0], "Neighborhood"] = "XYZ_NEW"

    prep = build_preprocessor(kind="tree")
    prep.fit(X_train_clean)
    X_holdout_prep = prep.transform(X_holdout_mod)

    assert X_holdout_prep.isnull().sum().sum() == 0


def test_skew_corrector_reduces_mean_skewness(train_and_holdout_clean):
    X_train_clean, _, _, _, _, _ = train_and_holdout_clean

    num_cols = [c for c in X_train_clean.columns if pd.api.types.is_numeric_dtype(X_train_clean[c])]
    X_num = X_train_clean[num_cols].fillna(X_train_clean[num_cols].median())

    corrector = SkewCorrector(threshold=0.75)
    corrector.fit(X_num)

    assert hasattr(corrector, "skewed_cols_")
    assert len(corrector.skewed_cols_) > 0

    skew_before = X_num[corrector.skewed_cols_].skew().abs().mean()

    X_trans = corrector.transform(X_num)
    skew_after = X_trans[corrector.skewed_cols_].skew().abs().mean()

    assert skew_after < skew_before, f"Expected mean |skew| after ({skew_after:.4f}) < before ({skew_before:.4f})"


def test_build_full_pipeline_ridge_linear(train_and_holdout_clean):
    _, _, y_train, _, train_df, holdout_df = train_and_holdout_clean

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    X_holdout = holdout_df.drop(columns=["SalePrice"], errors="ignore")

    pipe = build_full_pipeline(Ridge(alpha=10.0), kind="linear")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_holdout)

    assert len(preds) == len(X_holdout)
    assert (preds > 0).all(), "All predicted prices should be positive"


def test_joblib_dump_and_load_full_pipeline(train_and_holdout_clean, tmp_path):
    _, _, y_train, _, train_df, holdout_df = train_and_holdout_clean

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    X_holdout = holdout_df.drop(columns=["SalePrice"], errors="ignore")

    pipe = build_full_pipeline(Ridge(alpha=10.0), kind="linear")
    pipe.fit(X_train, y_train)

    preds_original = pipe.predict(X_holdout)

    model_file = tmp_path / "fitted_pipeline.joblib"
    joblib.dump(pipe, model_file)

    pipe_loaded = joblib.load(model_file)
    preds_loaded = pipe_loaded.predict(X_holdout)

    np.testing.assert_allclose(preds_original, preds_loaded, rtol=1e-5)
