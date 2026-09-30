"""Synthetic dataset generator for Ames Housing CI testing."""

import json
import numpy as np
import pandas as pd

from api.schemas import (
    ALLOWED_MS_ZONINGS,
    ALLOWED_NEIGHBORHOODS,
    ALLOWED_SALE_CONDITIONS,
)
from lifinity.config import get_project_root


def make_synthetic_ames(n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Ames Housing dataset matching raw schemas.

    Parameters
    ----------
    n_samples : int, default 300
        Number of synthetic rows to generate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset containing all columns from models/input_defaults.json + Id + SalePrice.
    """
    rng = np.random.default_rng(seed)
    root = get_project_root()
    defaults_path = root / "models" / "input_defaults.json"

    with open(defaults_path, "r", encoding="utf-8") as f:
        defaults = json.load(f)

    data = {}
    data["Id"] = np.arange(1, n_samples + 1)

    # Allowed categories
    quality_ratings = ["Ex", "Gd", "TA", "Fa", "Po"]

    # Neighborhood effect mapping for SalePrice formula
    neigh_list = list(ALLOWED_NEIGHBORHOODS)
    neigh_effects = {n: (idx - len(neigh_list) / 2) * 0.02 for idx, n in enumerate(neigh_list)}

    for col, default_val in defaults.items():
        if col == "Neighborhood":
            data[col] = rng.choice(neigh_list, size=n_samples)
        elif col == "MSZoning":
            data[col] = rng.choice(list(ALLOWED_MS_ZONINGS), size=n_samples)
        elif col == "SaleCondition":
            data[col] = rng.choice(list(ALLOWED_SALE_CONDITIONS), size=n_samples)
        elif col in ("KitchenQual", "ExterQual"):
            data[col] = rng.choice(quality_ratings, size=n_samples)
        elif col in ("OverallQual", "OverallCond"):
            data[col] = rng.integers(1, 11, size=n_samples)
        elif col in ("YearBuilt", "YearRemodAdd", "GarageYrBlt"):
            data[col] = rng.integers(1900, 2011, size=n_samples)
        elif col == "YrSold":
            data[col] = rng.integers(2006, 2011, size=n_samples)
        elif col == "MoSold":
            data[col] = rng.integers(1, 13, size=n_samples)
        elif isinstance(default_val, (int, float)):
            # Numeric column: default * lognormal(mean=0, sigma=0.3)
            base_val = float(default_val) if default_val != 0 else 10.0
            noise = rng.lognormal(0.0, 0.3, size=n_samples)
            vals = base_val * noise
            if isinstance(default_val, int):
                vals = np.round(vals).astype(int)
            data[col] = vals
        else:
            # String categorical: use default value
            data[col] = [str(default_val)] * n_samples

    df = pd.DataFrame(data)

    # Ensure YearRemodAdd >= YearBuilt for logical consistency
    df["YearRemodAdd"] = np.maximum(df["YearRemodAdd"], df["YearBuilt"])

    # ~5% NaNs in specific columns
    for col in ["LotFrontage", "GarageType", "FireplaceQu", "PoolQC"]:
        if col in df.columns:
            nan_mask = rng.random(size=n_samples) < 0.05
            df.loc[nan_mask, col] = np.nan

    # Calculate synthetic target SalePrice according to formula
    overall_qual = df["OverallQual"].astype(float)
    gr_liv_area = df["GrLivArea"].astype(float)
    total_bsmt_sf = df["TotalBsmtSF"].astype(float)
    year_built = df["YearBuilt"].astype(float)
    n_effect = df["Neighborhood"].map(neigh_effects).fillna(0.0).astype(float)
    eps = rng.normal(0.0, 0.1, size=n_samples)

    log_price = (
        10.5
        + 0.12 * overall_qual
        + 0.0004 * gr_liv_area
        + 0.0002 * total_bsmt_sf
        + 0.003 * (year_built - 1950)
        + n_effect
        + eps
    )
    df["SalePrice"] = np.exp(log_price)

    return df
