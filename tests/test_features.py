"""Tests for FeatureEngineer and build_cleaning_pipeline."""

import numpy as np
import pandas as pd
import pytest

from lifinity.config import get_project_root
from lifinity.features.engineering import FeatureEngineer, QUAL_MAP, build_cleaning_pipeline


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_minimal_df(n: int = 2) -> pd.DataFrame:
    """Two rows with every column the transformations need."""
    return pd.DataFrame(
        {
            "MSSubClass": [20, 60],
            "LotArea": [8450, 9600],
            "OverallQual": [7, 6],
            "OverallCond": [5, 8],
            "YearBuilt": [2003, 1976],
            "YearRemodAdd": [2003, 1976],
            "ExterQual": ["Gd", "TA"],
            "BsmtQual": ["Gd", "TA"],
            "TotalBsmtSF": [856.0, 1262.0],
            "1stFlrSF": [856, 1262],
            "2ndFlrSF": [854, 0],
            "GrLivArea": [1710, 1262],
            "BsmtFullBath": [1.0, 0.0],
            "BsmtHalfBath": [0.0, 1.0],
            "FullBath": [2, 2],
            "HalfBath": [1, 0],
            "Fireplaces": [0, 1],
            "GarageCars": [2.0, 2.0],
            "GarageArea": [548.0, 460.0],
            "PoolArea": [0, 0],
            "WoodDeckSF": [0, 298],
            "OpenPorchSF": [61, 0],
            "EnclosedPorch": [0, 0],
            "3SsnPorch": [0, 0],
            "ScreenPorch": [0, 0],
            "MoSold": [2, 5],
            "YrSold": [2008, 2007],
            "KitchenQual": ["Gd", "TA"],
            "GarageQual": ["TA", "TA"],
            "GarageYrBlt": [2003.0, 1976.0],
            "MiscVal": [0, 0],
        }
    )


@pytest.fixture
def df() -> pd.DataFrame:
    return make_minimal_df()


@pytest.fixture
def fe_default() -> FeatureEngineer:
    return FeatureEngineer(drop_after=["GarageArea", "GarageYrBlt", "PoolArea", "MiscVal"])


# ---------------------------------------------------------------------------
# 1. TotalSF and TotalBath correctness
# ---------------------------------------------------------------------------

def test_total_sf(df, fe_default):
    out = fe_default.fit_transform(df)
    expected = [856 + 856 + 854, 1262 + 1262 + 0]
    assert list(out["TotalSF"]) == expected


def test_total_bath(df, fe_default):
    out = fe_default.fit_transform(df)
    # row0: 2 + 0.5*1 + 1 + 0.5*0 = 3.5
    # row1: 2 + 0.5*0 + 0 + 0.5*1 = 2.5
    assert out["TotalBath"].tolist() == [3.5, 2.5]


# ---------------------------------------------------------------------------
# 2. HouseAge and RemodAge never negative
# ---------------------------------------------------------------------------

def test_house_age_non_negative():
    df = pd.DataFrame(
        {
            "YrSold": [2005, 2000],
            "YearBuilt": [2010, 1990],
            "YearRemodAdd": [2010, 1990],
        }
    )
    fe = FeatureEngineer(drop_after=[])
    out = fe.fit_transform(df)
    assert (out["HouseAge"] >= 0).all()


def test_remod_age_non_negative():
    df = pd.DataFrame(
        {
            "YrSold": [2000, 2010],
            "YearBuilt": [1980, 2000],
            "YearRemodAdd": [2005, 1990],
        }
    )
    fe = FeatureEngineer(drop_after=[])
    out = fe.fit_transform(df)
    assert (out["RemodAge"] >= 0).all()


# ---------------------------------------------------------------------------
# 3. Has* / Is* columns contain only 0 and 1
# ---------------------------------------------------------------------------

def test_binary_flag_values(df, fe_default):
    out = fe_default.fit_transform(df)
    flag_cols = [c for c in out.columns if c.startswith("Has") or c.startswith("Is")]
    assert flag_cols, "Expected at least one Has*/Is* column"
    for col in flag_cols:
        unique = set(out[col].unique())
        assert unique.issubset({0, 1}), f"{col} has non-binary values: {unique}"


# ---------------------------------------------------------------------------
# 4. MSSubClass, MoSold, YrSold are str after transform; HouseAge still correct
# ---------------------------------------------------------------------------

def test_code_cols_cast_to_str(df, fe_default):
    out = fe_default.fit_transform(df)
    for col in ["MSSubClass", "MoSold", "YrSold"]:
        if col in out.columns:
            assert out[col].dtype == object, f"{col} should be object/str dtype"
            for v in out[col]:
                assert isinstance(v, str), f"{col} value {v!r} is not str"


def test_house_age_correct_despite_str_cast(df, fe_default):
    """Ages must be computed before YrSold is cast to str."""
    out = fe_default.fit_transform(df)
    # row0: 2008 - 2003 = 5, row1: 2007 - 1976 = 31
    assert list(out["HouseAge"]) == [5.0, 31.0]


# ---------------------------------------------------------------------------
# 5. No crash when PoolArea or 3SsnPorch is missing
# ---------------------------------------------------------------------------

def test_missing_pool_area_no_crash():
    df = pd.DataFrame(
        {
            "OverallQual": [7],
            "GrLivArea": [1500],
            "LotArea": [8000],
        }
    )
    fe = FeatureEngineer(drop_after=[])
    out = fe.fit_transform(df)
    assert "HasPool" not in out.columns
    assert "TotalPorchSF" not in out.columns


def test_missing_3ssn_porch_no_crash():
    """With 3SsnPorch absent, TotalPorchSF should not be created."""
    df = pd.DataFrame(
        {
            "OpenPorchSF": [10],
            "EnclosedPorch": [0],
            "ScreenPorch": [0],
            "WoodDeckSF": [0],
            # 3SsnPorch intentionally missing
        }
    )
    fe = FeatureEngineer(drop_after=[])
    out = fe.fit_transform(df)
    assert "TotalPorchSF" not in out.columns


# ---------------------------------------------------------------------------
# 6. drop_after columns are removed
# ---------------------------------------------------------------------------

def test_drop_after_columns_gone(df, fe_default):
    out = fe_default.fit_transform(df)
    for col in ["GarageArea", "GarageYrBlt", "PoolArea", "MiscVal"]:
        assert col not in out.columns, f"{col} should have been dropped"


# ---------------------------------------------------------------------------
# 7. build_cleaning_pipeline: fit on train, transform holdout
#    -> zero NaN, zero inf, identical columns
# ---------------------------------------------------------------------------

def test_build_cleaning_pipeline_integration():
    root = get_project_root()
    train_path = root / "data" / "processed" / "train.parquet"
    holdout_path = root / "data" / "processed" / "holdout.parquet"

    train_df = pd.read_parquet(train_path)
    holdout_df = pd.read_parquet(holdout_path)

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    X_holdout = holdout_df.drop(columns=["SalePrice"], errors="ignore")

    pipe = build_cleaning_pipeline()
    X_train_out = pipe.fit_transform(X_train)
    X_holdout_out = pipe.transform(X_holdout)

    # zero NaN
    assert X_train_out.isnull().sum().sum() == 0, "train output has NaN"
    assert X_holdout_out.isnull().sum().sum() == 0, "holdout output has NaN"

    # zero inf
    num_train = X_train_out.select_dtypes(include=[np.number])
    num_holdout = X_holdout_out.select_dtypes(include=[np.number])
    assert not np.isinf(num_train.values).any(), "train output has inf"
    assert not np.isinf(num_holdout.values).any(), "holdout output has inf"

    # identical columns
    assert list(X_train_out.columns) == list(X_holdout_out.columns), (
        "train and holdout output columns differ"
    )
