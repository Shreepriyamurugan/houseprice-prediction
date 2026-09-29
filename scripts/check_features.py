"""Quick sanity check for FeatureEngineer and build_cleaning_pipeline."""

import numpy as np
import pandas as pd

from lifinity.config import get_project_root
from lifinity.features.engineering import build_cleaning_pipeline


def main() -> None:
    root = get_project_root()
    train_path = root / "data" / "processed" / "train.parquet"
    val_path = root / "data" / "processed" / "val.parquet"

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    X_holdout = val_df.drop(columns=["SalePrice"], errors="ignore")

    pipe = build_cleaning_pipeline()
    X_train_out = pipe.fit_transform(X_train)
    X_holdout_out = pipe.transform(X_holdout)

    # --- shapes -----------------------------------------------------------
    print("=" * 60)
    print("Output shapes")
    print(f"  train  : {X_train_out.shape}")
    print(f"  holdout: {X_holdout_out.shape}")

    # --- NaN / inf --------------------------------------------------------
    num_train = X_train_out.select_dtypes(include=[np.number])
    num_holdout = X_holdout_out.select_dtypes(include=[np.number])

    print()
    print("NaN counts")
    print(f"  train  : {X_train_out.isnull().sum().sum()}")
    print(f"  holdout: {X_holdout_out.isnull().sum().sum()}")

    print()
    print("Inf counts")
    print(f"  train  : {np.isinf(num_train.values).sum()}")
    print(f"  holdout: {np.isinf(num_holdout.values).sum()}")

    # --- Correlation of NEW features with log1p(SalePrice) on train ------
    NEW_FEATURES = [
        "TotalSF", "TotalBath", "HouseAge", "RemodAge", "IsRemodeled", "IsNew",
        "TotalPorchSF", "QualxArea", "OverallScore", "QualSum",
        "HasPool", "HasGarage", "HasBsmt", "Has2ndFlr", "HasFireplace",
        "LotRatio",
    ]

    if "SalePrice" in train_df.columns:
        target = np.log1p(train_df.loc[X_train_out.index, "SalePrice"])
        present = [c for c in NEW_FEATURES if c in X_train_out.columns]
        corrs = {}
        for col in present:
            series = pd.to_numeric(X_train_out[col], errors="coerce")
            c = series.corr(target)
            corrs[col] = c
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)

        print()
        print("Correlation of new features with log1p(SalePrice) [sorted by |r|]")
        print(f"  {'Feature':<20}  {'r':>8}")
        print(f"  {'-'*20}  {'-'*8}")
        for feat, r in sorted_corrs:
            print(f"  {feat:<20}  {r:>8.4f}")
    else:
        print()
        print("SalePrice not found in train — skipping correlation.")


if __name__ == "__main__":
    main()
