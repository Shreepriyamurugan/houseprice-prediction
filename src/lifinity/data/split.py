"""Data split module for 70/15/15 train/val/test split."""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

from lifinity.config import get_project_root, load_params
from lifinity.features.cleaning import remove_outliers


def make_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data, perform stratified 70/15/15 train/val/test split, and clean train outliers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_clean, val_clean, test_clean)
    """
    root = get_project_root()
    raw_path = root / "data" / "raw" / "train.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df = pd.read_csv(raw_path)
    params = load_params()

    seed = params.get("split", {}).get("seed", 42)

    # 1. First split: train 70% vs temp 30%, stratified on pd.qcut(SalePrice, q=10)
    strat_bins_1 = pd.qcut(df["SalePrice"], q=10, labels=False)
    train_raw, temp_raw = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=strat_bins_1,
    )

    # 2. Second split: temp -> val 50% / test 50%, stratified on pd.qcut(temp SalePrice, q=5)
    strat_bins_2 = pd.qcut(temp_raw["SalePrice"], q=5, labels=False)
    val_raw, test_raw = train_test_split(
        temp_raw,
        test_size=0.50,
        random_state=seed,
        stratify=strat_bins_2,
    )

    # 3. remove_outliers on TRAIN only (val and test stay untouched)
    grlivarea_max = params.get("outliers", {}).get("grlivarea_max", 4000)
    saleprice_min = params.get("outliers", {}).get("saleprice_min", 300000)
    train_clean, n_removed = remove_outliers(
        train_raw,
        grlivarea_max=grlivarea_max,
        saleprice_min=saleprice_min,
    )

    val_clean = val_raw.copy()
    test_clean = test_raw.copy()

    # 4. Assert no Id appears in more than one set
    train_ids = set(train_clean["Id"])
    val_ids = set(val_clean["Id"])
    test_ids = set(test_clean["Id"])

    assert len(train_ids & val_ids) == 0, "Id overlap between train and val"
    assert len(train_ids & test_ids) == 0, "Id overlap between train and test"
    assert len(val_ids & test_ids) == 0, "Id overlap between val and test"

    # 5. Save data/processed/train.parquet, val.parquet, test.parquet
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_clean.to_parquet(processed_dir / "train.parquet", index=False)
    val_clean.to_parquet(processed_dir / "val.parquet", index=False)
    test_clean.to_parquet(processed_dir / "test.parquet", index=False)

    # Delete old holdout.parquet if present
    holdout_file = processed_dir / "holdout.parquet"
    if holdout_file.exists():
        holdout_file.unlink()

    print("=" * 60)
    print("Dataset Split Summary (70 / 15 / 15)")
    print(f"  train.parquet : {train_clean.shape} (outliers removed: {n_removed})")
    print(f"  val.parquet   : {val_clean.shape}")
    print(f"  test.parquet  : {test_clean.shape}")
    print("Median SalePrice:")
    print(f"  train : ${train_clean['SalePrice'].median():,.2f}")
    print(f"  val   : ${val_clean['SalePrice'].median():,.2f}")
    print(f"  test  : ${test_clean['SalePrice'].median():,.2f}")

    return train_clean, val_clean, test_clean


if __name__ == "__main__":
    make_split()
