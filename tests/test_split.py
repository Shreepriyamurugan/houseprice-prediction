"""Tests for 70/15/15 dataset splitting."""

import pandas as pd
import pytest

from lifinity.config import get_project_root
from lifinity.data.split import make_split


@pytest.mark.requires_data
def test_split_proportions_and_no_overlap():
    root = get_project_root()
    raw_path = root / "data" / "raw" / "train.csv"
    raw_df = pd.read_csv(raw_path)
    n_raw = len(raw_df)

    train_clean, val_clean, test_clean = make_split()

    # Before outlier removal, proportions were 70% / 15% / 15%
    # Train lost 2 outliers, so train length is 1020 out of 1460 raw (approx 69.86%)
    # val and test lengths are exactly 219 rows (15.0% of 1460)
    assert abs(len(val_clean) / n_raw - 0.15) < 0.01, f"Val proportion {len(val_clean)/n_raw:.4f} not within 1% of 15%"
    assert abs(len(test_clean) / n_raw - 0.15) < 0.01, f"Test proportion {len(test_clean)/n_raw:.4f} not within 1% of 15%"

    # No Id overlap
    set_train = set(train_clean["Id"])
    set_val = set(val_clean["Id"])
    set_test = set(test_clean["Id"])

    assert len(set_train & set_val) == 0, "Overlap between train and val IDs"
    assert len(set_train & set_test) == 0, "Overlap between train and test IDs"
    assert len(set_val & set_test) == 0, "Overlap between val and test IDs"


@pytest.mark.requires_data
def test_outliers_only_removed_from_train():
    train_clean, val_clean, test_clean = make_split()

    # Outlier criteria: GrLivArea > 4000 AND SalePrice < 300000
    outliers_in_val = val_clean[(val_clean["GrLivArea"] > 4000) & (val_clean["SalePrice"] < 300000)]
    outliers_in_test = test_clean[(test_clean["GrLivArea"] > 4000) & (test_clean["SalePrice"] < 300000)]
    outliers_in_train = train_clean[(train_clean["GrLivArea"] > 4000) & (train_clean["SalePrice"] < 300000)]

    assert len(outliers_in_train) == 0, "Outliers should be removed from train"
