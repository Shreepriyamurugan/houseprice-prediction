"""CI smoke model generator.

Refuses to overwrite an existing real model (models/model.joblib).
If models/model.joblib is missing (e.g. in CI), generates synthetic data,
fits a smoke model pipeline, and saves models/model.joblib and models/metrics.json.
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path when script is executed directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from lifinity.features.preprocessor import build_full_pipeline, rmse_log
from lifinity.models.ensemble import LogBlendRegressor
from tests.synthetic import make_synthetic_ames


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.joblib"
    metrics_path = models_dir / "metrics.json"

    if model_path.exists():
        print("real model present - not overwriting")
        sys.exit(0)

    print("models/model.joblib missing. Generating synthetic smoke model for CI...")
    df = make_synthetic_ames(n_samples=300, seed=42)
    train_df = df.iloc[:240].copy()
    test_df = df.iloc[240:].copy()

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]
    X_test = test_df.drop(columns=["SalePrice"], errors="ignore")
    y_test = test_df["SalePrice"]

    ridge_pipe = build_full_pipeline(Ridge(alpha=1.0), kind="linear")
    lgbm_pipe = build_full_pipeline(LGBMRegressor(n_estimators=50, random_state=42, verbose=-1), kind="tree")

    blend = LogBlendRegressor(
        estimators=[("Ridge", ridge_pipe), ("LGBM", lgbm_pipe)],
        weights=[0.5, 0.5],
    )
    blend.fit(X_train, y_train)

    preds = blend.predict(X_test)
    rmse = rmse_log(y_test, preds)
    mae = float(np.mean(np.abs(preds - y_test)))
    mape = float(np.mean(np.abs((y_test - preds) / y_test)))

    metrics = {
        "final_model": "LogBlendRegressor_Synthetic_CI",
        "rmse_log": float(rmse),
        "mae": mae,
        "mape": mape,
    }

    joblib.dump(blend, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Synthetic smoke model created at {model_path} with RMSE(log)={rmse:.4f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
