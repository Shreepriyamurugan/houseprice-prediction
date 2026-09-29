"""Feature engineering transformer for the Ames housing dataset."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from lifinity.config import load_params
from lifinity.features.cleaning import ColumnDropper, DomainImputer

QUAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless feature engineering transformer for the Ames housing dataset.

    Creates domain-specific features from raw columns.  Only creates a
    feature when *all* required input columns are present so the transformer
    never crashes on partial DataFrames (missing-column-safe).
    """

    def __init__(self, drop_after: list[str] | None = None) -> None:
        self.drop_after = drop_after

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        """Stateless: record column info for get_feature_names_out."""
        if self.drop_after is None:
            params = load_params()
            self.drop_after_ = list(params.get("features", {}).get("drop_after", []))
        else:
            self.drop_after_ = list(self.drop_after)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            dummy = self.transform(X.iloc[:0].copy())
            self.feature_names_out_ = np.array(dummy.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain features on a copy of X."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()

        # ---- area totals ------------------------------------------------
        if all(c in X.columns for c in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
            X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

        if all(c in X.columns for c in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
            X["TotalBath"] = (
                X["FullBath"]
                + 0.5 * X["HalfBath"]
                + X["BsmtFullBath"]
                + 0.5 * X["BsmtHalfBath"]
            )

        # ---- ages (BEFORE year columns are cast to str) -----------------
        if all(c in X.columns for c in ["YrSold", "YearBuilt"]):
            yr_sold = pd.to_numeric(X["YrSold"], errors="coerce")
            yr_built = pd.to_numeric(X["YearBuilt"], errors="coerce")
            X["HouseAge"] = (yr_sold - yr_built).clip(lower=0)

        if all(c in X.columns for c in ["YrSold", "YearRemodAdd"]):
            yr_sold = pd.to_numeric(X["YrSold"], errors="coerce")
            yr_remod = pd.to_numeric(X["YearRemodAdd"], errors="coerce")
            X["RemodAge"] = (yr_sold - yr_remod).clip(lower=0)

        if all(c in X.columns for c in ["YearRemodAdd", "YearBuilt"]):
            X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)

        if all(c in X.columns for c in ["YrSold", "YearBuilt"]):
            yr_sold2 = pd.to_numeric(X["YrSold"], errors="coerce")
            yr_built2 = pd.to_numeric(X["YearBuilt"], errors="coerce")
            X["IsNew"] = (yr_sold2 == yr_built2).astype(int)

        # ---- porch total ------------------------------------------------
        porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        if all(c in X.columns for c in porch_cols):
            X["TotalPorchSF"] = (
                X["OpenPorchSF"]
                + X["EnclosedPorch"]
                + X["3SsnPorch"]
                + X["ScreenPorch"]
                + X["WoodDeckSF"]
            )

        # ---- quality interactions ---------------------------------------
        if all(c in X.columns for c in ["OverallQual", "GrLivArea"]):
            X["QualxArea"] = X["OverallQual"] * X["GrLivArea"]

        if all(c in X.columns for c in ["OverallQual", "OverallCond"]):
            X["OverallScore"] = X["OverallQual"] * X["OverallCond"]

        qual_cols = ["ExterQual", "KitchenQual", "BsmtQual", "GarageQual"]
        if all(c in X.columns for c in qual_cols):
            mapped = [X[c].map(QUAL_MAP).fillna(0) for c in qual_cols]
            X["QualSum"] = sum(mapped).astype(int)

        # ---- binary presence flags -------------------------------------
        if "PoolArea" in X.columns:
            X["HasPool"] = (X["PoolArea"] > 0).astype(int)
        if "GarageCars" in X.columns:
            X["HasGarage"] = (X["GarageCars"] > 0).astype(int)
        if "TotalBsmtSF" in X.columns:
            X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
        if "2ndFlrSF" in X.columns:
            X["Has2ndFlr"] = (X["2ndFlrSF"] > 0).astype(int)
        if "Fireplaces" in X.columns:
            X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)

        # ---- lot ratio --------------------------------------------------
        if all(c in X.columns for c in ["GrLivArea", "LotArea"]):
            lot_area = X["LotArea"].replace({0: np.nan})
            X["LotRatio"] = (X["GrLivArea"] / lot_area).fillna(0.0).astype(float)

        # ---- cast code columns to str AFTER ages are computed ----------
        for col in ["MSSubClass", "MoSold", "YrSold"]:
            if col in X.columns:
                X[col] = X[col].astype(str)

        # ---- drop configured columns -----------------------------------
        cols_to_drop = getattr(self, "drop_after_", None)
        if cols_to_drop is None:
            if self.drop_after is None:
                params = load_params()
                cols_to_drop = list(params.get("features", {}).get("drop_after", []))
            else:
                cols_to_drop = list(self.drop_after)

        existing = [c for c in cols_to_drop if c in X.columns]
        if existing:
            X = X.drop(columns=existing)

        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names."""
        if input_features is not None:
            dummy_df = pd.DataFrame(columns=list(input_features))
            return np.array(self.transform(dummy_df).columns, dtype=object)
        if hasattr(self, "feature_names_out_"):
            return self.feature_names_out_
        if hasattr(self, "feature_names_in_"):
            dummy_df = pd.DataFrame(columns=list(self.feature_names_in_))
            return np.array(self.transform(dummy_df).columns, dtype=object)
        raise ValueError(
            "FeatureEngineer has not been fitted and input_features was not provided."
        )


def build_cleaning_pipeline() -> Pipeline:
    """Build the full cleaning + feature-engineering pipeline.

    Reads parameters from params.yaml via load_params().

    Returns
    -------
    Pipeline
        sklearn Pipeline with steps: impute -> drop -> features.
    """
    params = load_params()
    clean_drop_cols = params.get("clean", {}).get("drop_cols", None)
    features_drop_after = params.get("features", {}).get("drop_after", None)
    return Pipeline(
        [
            ("impute", DomainImputer()),
            ("drop", ColumnDropper(drop_cols=clean_drop_cols)),
            ("features", FeatureEngineer(drop_after=features_drop_after)),
        ]
    )
