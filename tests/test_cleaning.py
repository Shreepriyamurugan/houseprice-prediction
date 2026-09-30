import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.requires_data

from lifinity.config import get_project_root, load_params
from lifinity.features.cleaning import ColumnDropper, DomainImputer, remove_outliers


@pytest.fixture
def raw_train_df():
    """Load raw training dataset."""
    raw_path = get_project_root() / "data" / "raw" / "train.csv"
    return pd.read_csv(raw_path)


def test_domain_imputer_train_csv_zero_nan(raw_train_df):
    """Test DomainImputer.fit_transform on train.csv leaves zero NaN."""
    imputer = DomainImputer()
    cleaned = imputer.fit_transform(raw_train_df)

    assert isinstance(cleaned, pd.DataFrame)
    assert cleaned.isna().sum().sum() == 0


def test_poolqc_and_garage_area_imputation():
    """Test PoolQC NA -> 'None'; GarageArea NA -> 0."""
    df = pd.DataFrame(
        {
            "PoolQC": [np.nan, "Ex"],
            "GarageArea": [np.nan, 450.0],
        }
    )
    imputer = DomainImputer()
    result = imputer.fit_transform(df)

    assert result["PoolQC"].iloc[0] == "None"
    assert result["PoolQC"].iloc[1] == "Ex"
    assert result["GarageArea"].iloc[0] == 0
    assert result["GarageArea"].iloc[1] == 450.0


def test_garage_yr_blt_equals_year_built():
    """Test GarageYrBlt NA equals that row's YearBuilt."""
    df = pd.DataFrame(
        {
            "GarageYrBlt": [np.nan, 2005.0],
            "YearBuilt": [1995, 2005],
        }
    )
    imputer = DomainImputer()
    result = imputer.fit_transform(df)

    assert result["GarageYrBlt"].iloc[0] == 1995.0
    assert result["GarageYrBlt"].iloc[1] == 2005.0


def test_lot_frontage_medians_from_fit_only():
    """Test LotFrontage medians come only from fit data (subset A medians applied to subset B)."""
    # Subset A: CollgCr has frontages [60, 80] -> median 70. Veenker has frontage [100] -> median 100.
    # Overall LotFrontage median in Subset A is 80.
    subset_a = pd.DataFrame(
        {
            "Neighborhood": ["CollgCr", "CollgCr", "Veenker"],
            "LotFrontage": [60.0, 80.0, 100.0],
        }
    )

    # Subset B has NaNs and an unseen neighborhood
    subset_b = pd.DataFrame(
        {
            "Neighborhood": ["CollgCr", "Veenker", "Edwards"],
            "LotFrontage": [np.nan, np.nan, np.nan],
        }
    )

    imputer = DomainImputer()
    imputer.fit(subset_a)
    transformed_b = imputer.transform(subset_b)

    # CollgCr should take subset A median (70.0)
    assert transformed_b["LotFrontage"].iloc[0] == 70.0
    # Veenker should take subset A median (100.0)
    assert transformed_b["LotFrontage"].iloc[1] == 100.0
    # Edwards (unseen) should take subset A global median (80.0)
    assert transformed_b["LotFrontage"].iloc[2] == 80.0


def test_remove_outliers_on_full_train(raw_train_df):
    """Test remove_outliers on full train.csv removes exactly Ids 524 and 1299."""
    params = load_params()
    grlivarea_max = params.get("outliers", {}).get("grlivarea_max", 4000)
    saleprice_min = params.get("outliers", {}).get("saleprice_min", 300000)

    df_clean, n_removed = remove_outliers(
        raw_train_df,
        grlivarea_max=grlivarea_max,
        saleprice_min=saleprice_min,
    )

    assert n_removed == 2
    assert len(raw_train_df) - len(df_clean) == 2
    removed_ids = set(raw_train_df["Id"]) - set(df_clean["Id"])
    assert removed_ids == {524, 1299}


def test_column_dropper_ignores_missing_columns():
    """Test ColumnDropper ignores missing columns and drops present configured columns."""
    df = pd.DataFrame(
        {
            "Id": [1, 2],
            "SalePrice": [200000, 250000],
            "LotArea": [8000, 9000],
        }
    )

    # Id is in df, NonExistentCol and Utilities are not in df
    dropper = ColumnDropper(drop_cols=["Id", "Utilities", "NonExistentCol"])
    result = dropper.fit_transform(df)

    assert "Id" not in result.columns
    assert "SalePrice" in result.columns
    assert "LotArea" in result.columns
    assert list(result.columns) == ["SalePrice", "LotArea"]


def test_column_dropper_defaults_from_params(raw_train_df):
    """Test ColumnDropper uses clean.drop_cols from params.yaml by default."""
    params = load_params()
    configured_drop_cols = params["clean"]["drop_cols"]

    dropper = ColumnDropper()
    result = dropper.fit_transform(raw_train_df)

    for col in configured_drop_cols:
        assert col not in result.columns
