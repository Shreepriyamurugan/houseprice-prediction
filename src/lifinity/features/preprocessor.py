"""Preprocessing module defining ORDINAL_CATEGORIES, SkewCorrector, ColumnTransformers, and full pipelines."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler

from lifinity.config import load_params
from lifinity.features.cleaning import ColumnDropper, DomainImputer
from lifinity.features.engineering import FeatureEngineer

ORDINAL_CATEGORIES: dict[str, list[str]] = {
    "ExterQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "PavedDrive": ["N", "P", "Y"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}


def select_ordinal(df: pd.DataFrame) -> list[str]:
    """Select ordinal columns present in df."""
    return [col for col in ORDINAL_CATEGORIES if col in df.columns]


def select_nominal(df: pd.DataFrame) -> list[str]:
    """Select object/string columns in df not in ORDINAL_CATEGORIES."""
    return [
        col for col in df.columns
        if (df[col].dtype == object or pd.api.types.is_string_dtype(df[col]))
        and col not in ORDINAL_CATEGORIES
    ]


def select_numeric(df: pd.DataFrame) -> list[str]:
    """Select numeric columns in df."""
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


class SkewCorrector(BaseEstimator, TransformerMixin):
    """Transformer that applies Yeo-Johnson PowerTransformer to numeric columns with |skewness| > threshold."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SkewCorrector":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        thresh = self.threshold
        if thresh is None:
            params = load_params()
            thresh = params.get("preprocess", {}).get("skew_threshold", 0.75)

        self.threshold_ = thresh
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

        if num_cols:
            skewness = X[num_cols].skew(skipna=True)
            self.skewed_cols_ = skewness[skewness.abs() > thresh].index.tolist()
        else:
            self.skewed_cols_ = []

        if self.skewed_cols_:
            self.transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
            self.transformer_.fit(X[self.skewed_cols_])

        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        if hasattr(self, "skewed_cols_") and self.skewed_cols_ and hasattr(self, "transformer_"):
            cols_to_transform = [c for c in self.skewed_cols_ if c in X.columns]
            if cols_to_transform:
                transformed_vals = self.transformer_.transform(X[cols_to_transform])
                X[cols_to_transform] = transformed_vals

        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        raise ValueError("SkewCorrector has not been fitted or input_features not provided.")


def build_preprocessor(kind: str = "tree") -> ColumnTransformer:
    """Build a ColumnTransformer for tree or linear models.

    Parameters
    ----------
    kind : str, default="tree"
        "tree" or "linear".

    Returns
    -------
    ColumnTransformer
        Pre-configured ColumnTransformer with pandas DataFrame output.
    """
    params = load_params()
    min_freq = params.get("preprocess", {}).get("min_frequency", 0.01)
    skew_thresh = params.get("preprocess", {}).get("skew_threshold", 0.75)

    # Categories list in ordinal order
    ord_cats = [ORDINAL_CATEGORIES[c] for c in ORDINAL_CATEGORIES]
    ordinal_encoder = OrdinalEncoder(
        categories=ord_cats,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    onehot_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_freq,
        sparse_output=False,
    )

    if kind == "tree":
        ord_branch = ordinal_encoder
        num_branch = SimpleImputer(strategy="median")
    elif kind == "linear":
        ord_branch = Pipeline(
            [
                ("encoder", ordinal_encoder),
                ("scaler", RobustScaler()),
            ]
        )
        num_branch = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("skew", SkewCorrector(threshold=skew_thresh)),
                ("scaler", RobustScaler()),
            ]
        )
    else:
        raise ValueError(f"Unknown kind '{kind}'. Expected 'tree' or 'linear'.")

    ct = ColumnTransformer(
        transformers=[
            ("ordinal", ord_branch, select_ordinal),
            ("nominal", onehot_encoder, select_nominal),
            ("numeric", num_branch, select_numeric),
        ],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def build_full_pipeline(model: Any, kind: str = "tree") -> Pipeline:
    """Build full pipeline combining domain cleaning, feature engineering, preprocessing and model.

    Parameters
    ----------
    model : Estimator
        Regression model.
    kind : str, default="tree"
        Preprocessing kind ("tree" or "linear").

    Returns
    -------
    Pipeline
        Full sklearn Pipeline.
    """
    preprocessor = build_preprocessor(kind=kind)
    target_regressor = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline(
        [
            ("impute", DomainImputer()),
            ("drop", ColumnDropper()),
            ("features", FeatureEngineer()),
            ("prep", preprocessor),
            ("model", target_regressor),
        ]
    )


def rmse_log(y_true: Any, y_pred: Any) -> float:
    """Compute RMSE between log1p(y_true) and log1p(y_pred)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    
    y_true_clean = np.clip(y_true_arr, a_min=0, a_max=None)
    y_pred_clean = np.clip(y_pred_arr, a_min=0, a_max=None)

    log_true = np.log1p(y_true_clean)
    log_pred = np.log1p(y_pred_clean)
    return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))


RMSE_LOG_SCORER = make_scorer(rmse_log, greater_is_better=False)
