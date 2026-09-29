"""Check preprocessing shapes, one-hot columns, skew corrector, and baseline CV scores."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from lifinity.config import get_project_root
from lifinity.features.engineering import build_cleaning_pipeline
from lifinity.features.preprocessor import (
    RMSE_LOG_SCORER,
    SkewCorrector,
    build_full_pipeline,
    build_preprocessor,
)


def main() -> None:
    root = get_project_root()
    train_path = root / "data" / "processed" / "train.parquet"
    val_path = root / "data" / "processed" / "val.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    cleaning_pipe = build_cleaning_pipeline()
    X_train_clean = cleaning_pipe.fit_transform(train_df.drop(columns=["SalePrice"], errors="ignore"))
    X_holdout_clean = cleaning_pipe.transform(val_df.drop(columns=["SalePrice"], errors="ignore"))

    # Preprocessors
    prep_tree = build_preprocessor(kind="tree")
    X_train_tree = prep_tree.fit_transform(X_train_clean)

    prep_linear = build_preprocessor(kind="linear")
    X_train_linear = prep_linear.fit_transform(X_train_clean)

    # Number of one-hot columns
    nom_transformer = prep_tree.named_transformers_["nominal"]
    if hasattr(nom_transformer, "get_feature_names_out"):
        ohe_cols_count = len(nom_transformer.get_feature_names_out())
    else:
        ohe_cols_count = 0

    # Skewed cols count in linear numeric branch
    num_pipeline = prep_linear.named_transformers_["numeric"]
    skew_step = num_pipeline.named_steps.get("skew")
    skewed_cols_count = len(skew_step.skewed_cols_) if skew_step and hasattr(skew_step, "skewed_cols_") else 0

    print("=" * 60)
    print("Preprocessing Summary")
    print(f"  Tree kind output shape   : {X_train_tree.shape}")
    print(f"  Linear kind output shape : {X_train_linear.shape}")
    print(f"  One-hot encoded columns  : {ohe_cols_count}")
    print(f"  Skew-corrected columns  : {skewed_cols_count}")

    # Baseline CV Evaluation
    X_train_raw = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print()
    print("Running 5-Fold Baseline Cross-Validation...")

    # a) Ridge (linear)
    pipe_ridge = build_full_pipeline(Ridge(alpha=10.0), kind="linear")
    scores_ridge = cross_val_score(pipe_ridge, X_train_raw, y_train, cv=cv, scoring=RMSE_LOG_SCORER)
    rmse_ridge = -scores_ridge

    # b) LightGBM (tree)
    pipe_lgb = build_full_pipeline(LGBMRegressor(random_state=42, verbose=-1), kind="tree")
    scores_lgb = cross_val_score(pipe_lgb, X_train_raw, y_train, cv=cv, scoring=RMSE_LOG_SCORER)
    rmse_lgb = -scores_lgb

    print()
    print("Baseline RMSE(log) Scores (mean ± std):")
    print(f"  a) Ridge(alpha=10) [linear] : {rmse_ridge.mean():.4f} ± {rmse_ridge.std():.4f}")
    print(f"  b) LightGBM        [tree]   : {rmse_lgb.mean():.4f} ± {rmse_lgb.std():.4f}")


if __name__ == "__main__":
    main()
