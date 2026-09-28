from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from lifinity.config import get_project_root, load_params
from lifinity.features.cleaning import remove_outliers


def split_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data, perform stratified split, remove train outliers, and save processed datasets."""
    root = get_project_root()
    raw_path = root / "data" / "raw" / "train.csv"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    params = load_params()
    split_cfg = params.get("split", {})
    outliers_cfg = params.get("outliers", {})

    test_size = split_cfg.get("test_size", 0.2)
    seed = split_cfg.get("seed", params.get("seed", 42))
    grlivarea_max = outliers_cfg.get("grlivarea_max", 4000)
    saleprice_min = outliers_cfg.get("saleprice_min", 300000)

    # 1. Load data/raw/train.csv
    df = pd.read_csv(raw_path)

    # 2. Stratified split based on SalePrice deciles
    strata = pd.qcut(df["SalePrice"], q=10, labels=False)
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )

    # 3. remove_outliers on TRAIN part only; print rows removed
    train_df, n_removed = remove_outliers(
        train_df,
        grlivarea_max=grlivarea_max,
        saleprice_min=saleprice_min,
    )
    print(f"Outlier rows removed from train: {n_removed}")

    # 4. Save data/processed/train.parquet and data/processed/holdout.parquet; print both shapes
    train_parquet_path = processed_dir / "train.parquet"
    holdout_parquet_path = processed_dir / "holdout.parquet"

    train_df.to_parquet(train_parquet_path, index=False)
    holdout_df.to_parquet(holdout_parquet_path, index=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Holdout shape: {holdout_df.shape}")

    return train_df, holdout_df


if __name__ == "__main__":
    split_data()
