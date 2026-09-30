"""Model predictor for house price inference."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from lifinity.config import get_project_root

logger = logging.getLogger("lifinity.api.predictor")


class Predictor:
    """Predictor class that loads trained model, input defaults, and metrics ONCE."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or get_project_root()
        self.model_path = self.root_dir / "models" / "model.joblib"
        self.defaults_path = self.root_dir / "models" / "input_defaults.json"
        self.metrics_path = self.root_dir / "models" / "metrics.json"

        # Check required files
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.defaults_path.exists():
            raise FileNotFoundError(f"Input defaults file not found: {self.defaults_path}")
        if not self.metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {self.metrics_path}")

        # 1. Load input defaults & metrics
        with open(self.defaults_path, "r", encoding="utf-8") as f:
            self.input_defaults: dict[str, Any] = json.load(f)

        with open(self.metrics_path, "r", encoding="utf-8") as f:
            self.metrics: dict[str, Any] = json.load(f)

        self.model_type: str = self.metrics.get("final_model_type", "blend")
        self.model_version: str = str(
            self.metrics.get("git_commit") or self.metrics.get("version") or "1.0.0"
        )
        # Test MAPE is stored in percent, e.g. 8.28 -> 0.0828
        self.test_mape: float = float(self.metrics.get("mape", 8.28)) / 100.0

        # 2. Load model (joblib or MLflow via MODEL_SOURCE env var)
        self.model_source = os.getenv("MODEL_SOURCE", "joblib").lower()
        if self.model_source == "mlflow":
            import mlflow.pyfunc

            model_uri = "models:/lifinity-price@production"
            logger.info("Loading model from MLflow: %s", model_uri)
            self.model = mlflow.pyfunc.load_model(model_uri)
        else:
            self.model_source = "joblib"
            logger.info("Loading model from joblib: %s", self.model_path)
            self.model = joblib.load(self.model_path)

        # Expected raw columns in exact order
        self.raw_columns = list(self.input_defaults.keys())

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict house prices for a batch of records.

        Parameters
        ----------
        records : list[dict[str, Any]]
            Input feature dictionaries.

        Returns
        -------
        list[dict[str, Any]]
            List of prediction dictionaries containing predicted_price, price_range_low,
            price_range_high, model_type, model_version, fields_defaulted, warnings, latency_ms.
        """
        results = []
        for record in records:
            start_time = time.perf_counter()

            # Start from default dictionary
            filled_dict = dict(self.input_defaults)

            # Map alias names if present
            normalized_record = {}
            for k, v in record.items():
                if v is None:
                    continue
                if k == "first_flr_sf":
                    normalized_record["1stFlrSF"] = v
                elif k == "second_flr_sf":
                    normalized_record["2ndFlrSF"] = v
                else:
                    normalized_record[k] = v

            # Count fields supplied by user that overwrite defaults
            user_keys = set(normalized_record.keys())
            filled_dict.update(normalized_record)

            fields_defaulted = len(self.raw_columns) - len(user_keys & set(self.raw_columns))

            # Generate warnings
            warnings = []
            yb = filled_dict.get("YearBuilt")
            yra = filled_dict.get("YearRemodAdd")
            ys = filled_dict.get("YrSold")

            if (yb is not None and yb > 2010) or (yra is not None and yra > 2010):
                warnings.append(
                    "YearBuilt or YearRemodAdd > 2010 outside training data range; prediction less reliable"
                )
            if ys is not None and (ys < 2006 or ys > 2010):
                warnings.append(
                    "YrSold outside 2006-2010 training data range; prediction less reliable"
                )

            # Create single-row DataFrame with exact raw column order
            df_row = pd.DataFrame([filled_dict])[self.raw_columns]

            # Model prediction
            if hasattr(self.model, "predict"):
                raw_pred = self.model.predict(df_row)
            else:
                raw_pred = self.model(df_row)

            # Extract price value
            if hasattr(raw_pred, "flatten"):
                price_val = float(raw_pred.flatten()[0])
            elif isinstance(raw_pred, (list, tuple, pd.Series)):
                price_val = float(raw_pred[0])
            else:
                price_val = float(raw_pred)

            predicted_price = round(price_val, 2)
            price_low = round(predicted_price * (1.0 - self.test_mape), 2)
            price_high = round(predicted_price * (1.0 + self.test_mape), 2)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            results.append(
                {
                    "predicted_price": predicted_price,
                    "price_range_low": price_low,
                    "price_range_high": price_high,
                    "model_type": self.model_type,
                    "model_version": self.model_version,
                    "fields_defaulted": fields_defaulted,
                    "warnings": warnings,
                    "latency_ms": elapsed_ms,
                    "_filled_row": filled_dict,
                }
            )

        return results
