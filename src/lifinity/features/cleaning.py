from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from lifinity.config import load_params

NONE_COLS = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "FireplaceQu",
    "MasVnrType",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
]

ZERO_COLS = [
    "MasVnrArea",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
]


class DomainImputer(BaseEstimator, TransformerMixin):
    """Domain-specific imputer for the Ames housing dataset.

    Learns imputation statistics from training data and applies rules:
    a) BsmtExposure / BsmtFinType2: if TotalBsmtSF > 0 and value is NA -> learned mode
    b) NONE_COLS NA -> "None"; ZERO_COLS NA -> 0
    c) GarageYrBlt NA -> YearBuilt of that row
    d) LotFrontage NA -> neighborhood median, else global median
    e) Functional NA -> "Typ"
    f) remaining object NA -> learned mode; remaining numeric NA -> learned median
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DomainImputer":
        """Learn LotFrontage neighborhood and global medians, column modes and medians."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if "LotFrontage" in X.columns and "Neighborhood" in X.columns:
            self.lot_frontage_neighborhood_medians_ = (
                X.groupby("Neighborhood")["LotFrontage"].median().to_dict()
            )
        else:
            self.lot_frontage_neighborhood_medians_ = {}

        if "LotFrontage" in X.columns:
            lf_median = X["LotFrontage"].median(skipna=True)
            self.lot_frontage_global_median_ = float(lf_median) if pd.notna(lf_median) else 0.0
        else:
            self.lot_frontage_global_median_ = 0.0

        self.modes_: dict[str, Any] = {}
        self.medians_: dict[str, float] = {}

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                med = X[col].median(skipna=True)
                if pd.notna(med):
                    self.medians_[col] = float(med)
            else:
                modes = X[col].mode(dropna=True)
                if len(modes) > 0:
                    self.modes_[col] = modes.iloc[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X according to domain rules."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        # a) BsmtExposure / BsmtFinType2: if TotalBsmtSF > 0 and value is NA -> learned mode
        if "TotalBsmtSF" in X.columns:
            has_basement = (X["TotalBsmtSF"] > 0).fillna(False)
            for col in ["BsmtExposure", "BsmtFinType2"]:
                if col in X.columns and hasattr(self, "modes_") and col in self.modes_:
                    mask = has_basement & X[col].isna()
                    X.loc[mask, col] = self.modes_[col]

        # b) NONE_COLS NA -> "None"; ZERO_COLS NA -> 0
        for col in NONE_COLS:
            if col in X.columns:
                X[col] = X[col].fillna("None")

        for col in ZERO_COLS:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # c) GarageYrBlt NA -> YearBuilt of that row
        if "GarageYrBlt" in X.columns and "YearBuilt" in X.columns:
            X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])

        # d) LotFrontage NA -> neighborhood median, else global median
        if "LotFrontage" in X.columns:
            global_med = getattr(self, "lot_frontage_global_median_", 0.0)
            if "Neighborhood" in X.columns and hasattr(self, "lot_frontage_neighborhood_medians_"):
                nhood_med = X["Neighborhood"].map(self.lot_frontage_neighborhood_medians_)
                imputed_lf = nhood_med.fillna(global_med)
                X["LotFrontage"] = X["LotFrontage"].fillna(imputed_lf)
            else:
                X["LotFrontage"] = X["LotFrontage"].fillna(global_med)

        # e) Functional NA -> "Typ"
        if "Functional" in X.columns:
            X["Functional"] = X["Functional"].fillna("Typ")

        # f) remaining object NA -> learned mode; remaining numeric NA -> learned median
        for col in X.columns:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    if hasattr(self, "medians_") and col in self.medians_:
                        X[col] = X[col].fillna(self.medians_[col])
                else:
                    if hasattr(self, "modes_") and col in self.modes_:
                        X[col] = X[col].fillna(self.modes_[col])

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops specified columns, ignoring any absent ones."""

    def __init__(self, drop_cols: list[str] | None = None) -> None:
        self.drop_cols = drop_cols

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnDropper":
        if self.drop_cols is None:
            params = load_params()
            self.drop_cols_ = list(params.get("clean", {}).get("drop_cols", []))
        else:
            self.drop_cols_ = list(self.drop_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        cols_to_drop = getattr(self, "drop_cols_", None)
        if cols_to_drop is None:
            if self.drop_cols is None:
                params = load_params()
                cols_to_drop = list(params.get("clean", {}).get("drop_cols", []))
            else:
                cols_to_drop = list(self.drop_cols)

        existing_cols = [c for c in cols_to_drop if c in X.columns]
        return X.drop(columns=existing_cols).copy()


def remove_outliers(
    df: pd.DataFrame,
    grlivarea_max: float = 4000,
    saleprice_min: float = 300000,
) -> tuple[pd.DataFrame, int]:
    """Drop rows with GrLivArea > grlivarea_max AND SalePrice < saleprice_min.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Ames housing data.
    grlivarea_max : float
        Maximum allowable above-grade living area before outlier check.
    saleprice_min : float
        Minimum sale price threshold for outlier check.

    Returns
    -------
    tuple[pd.DataFrame, int]
        (df_clean, n_removed)
    """
    if "GrLivArea" not in df.columns or "SalePrice" not in df.columns:
        return df.copy(), 0

    outlier_mask = (df["GrLivArea"] > grlivarea_max) & (df["SalePrice"] < saleprice_min)
    n_removed = int(outlier_mask.sum())
    df_clean = df[~outlier_mask].copy()
    return df_clean, n_removed
