"""Final model evaluation, persistence, MLflow registration, and smoke checking."""

from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from lifinity.config import get_project_root, load_params
from lifinity.features.cleaning import remove_outliers
from lifinity.features.preprocessor import build_full_pipeline, rmse_log
from lifinity.models.ensemble import LogBlendRegressor
from lifinity.models.report import generate_report
from lifinity.models.train import get_data_md5, get_git_commit


def run_evaluation() -> None:
    root = get_project_root()
    params = load_params()

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    data_md5 = get_data_md5()

    print("=" * 70)
    print("STEP 9: Final Model Evaluation & Registration")
    print("=" * 70)

    # 1. Load train.parquet + val.parquet, concatenate & remove outliers
    train_df = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    val_df = pd.read_parquet(root / "data" / "processed" / "val.parquet")

    train_val_raw = pd.concat([train_df, val_df], ignore_index=True)

    grlivarea_max = params.get("outliers", {}).get("grlivarea_max", 4000)
    saleprice_min = params.get("outliers", {}).get("saleprice_min", 300000)
    train_val_clean, n_removed = remove_outliers(
        train_val_raw,
        grlivarea_max=grlivarea_max,
        saleprice_min=saleprice_min,
    )

    print(f"Final training set rows: {len(train_val_clean)} (outliers removed: {n_removed})")

    # 2. Compute input_defaults.json (for raw input columns except SalePrice and Id)
    raw_input_cols = [c for c in train_val_clean.columns if c not in ("SalePrice", "Id")]
    defaults = {}
    for col in raw_input_cols:
        s = train_val_clean[col].dropna()
        if pd.api.types.is_numeric_dtype(train_val_clean[col]):
            val = float(s.median())
        else:
            modes = s.mode()
            val = str(modes.iloc[0]) if len(modes) > 0 else ""
        defaults[col] = val

    defaults_path = models_dir / "input_defaults.json"
    with open(defaults_path, "w", encoding="utf-8") as f:
        json.dump(defaults, f, indent=2)
    print(f"Saved input defaults ({len(defaults)} columns) to {defaults_path}")

    # 3. Build & fit final model from params.yaml final_model
    final_cfg = params.get("final_model", {})
    model_type = final_cfg.get("type", "blend")
    cfg_params = final_cfg.get("params", {})
    seed = params.get("seed", 42)

    X_train_final = train_val_clean.drop(columns=["SalePrice"], errors="ignore")
    y_train_final = train_val_clean["SalePrice"]

    weights_dict = {}
    if model_type == "blend":
        weights_dict = cfg_params.get("weights", {})
        model_params = cfg_params.get("model_params", {})

        lasso_p = build_full_pipeline(Lasso(**model_params["Lasso"], max_iter=50000, random_state=seed), kind="linear")
        enet_p = build_full_pipeline(ElasticNet(**model_params["ElasticNet"], max_iter=50000, random_state=seed), kind="linear")
        xgb_p = build_full_pipeline(XGBRegressor(**model_params["XGBoost"], n_jobs=-1, random_state=seed), kind="tree")
        cat_p = build_full_pipeline(CatBoostRegressor(**model_params["CatBoost"], verbose=0, allow_writing_files=False, random_state=seed), kind="tree")

        estimators = [
            ("Lasso", lasso_p),
            ("ElasticNet", enet_p),
            ("XGBoost", xgb_p),
            ("CatBoost", cat_p),
        ]
        weights_list = [weights_dict[name] for name, _ in estimators]
        final_model = LogBlendRegressor(estimators=estimators, weights=weights_list)
    else:
        # Fallback to single Lasso if configured
        model_params = cfg_params
        final_model = build_full_pipeline(Lasso(**model_params.get("Lasso", {}), max_iter=50000, random_state=seed), kind="linear")

    print(f"Fitting final model ({model_type}) on final training set...")
    final_model.fit(X_train_final, y_train_final)

    # 4. Load test.parquet — predict ONCE
    test_df = pd.read_parquet(root / "data" / "processed" / "test.parquet")
    X_test = test_df.drop(columns=["SalePrice"], errors="ignore")
    y_test = test_df["SalePrice"]

    y_test_pred = final_model.predict(X_test)

    # 5. Compute test metrics
    test_rmse_log = float(rmse_log(y_test, y_test_pred))
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_mape = float(mean_absolute_percentage_error(y_test, y_test_pred) * 100.0)
    test_r2_log = float(r2_score(np.log1p(y_test), np.log1p(y_test_pred)))
    test_r2_dollar = float(r2_score(y_test, y_test_pred))

    # Error by price band
    band_masks = {
        "<$150k": y_test < 150000,
        "$150-300k": (y_test >= 150000) & (y_test < 300000),
        "$300-450k": (y_test >= 300000) & (y_test < 450000),
        ">$450k": y_test >= 450000,
    }

    band_metrics = {}
    for b_name, mask in band_masks.items():
        count = int(mask.sum())
        if count > 0:
            y_sub = y_test[mask]
            p_sub = y_test_pred[mask]
            b_mae = float(mean_absolute_error(y_sub, p_sub))
            b_mape = float(mean_absolute_percentage_error(y_sub, p_sub) * 100.0)
        else:
            b_mae = 0.0
            b_mape = 0.0
        band_metrics[b_name] = {"count": count, "mae": b_mae, "mape": b_mape}

    # 6. Fit reference Lasso
    model_params_all = cfg_params.get("model_params", {}) if model_type == "blend" else cfg_params
    lasso_params = model_params_all.get("Lasso", {})
    lasso_ref = build_full_pipeline(Lasso(**lasso_params, max_iter=50000, random_state=seed), kind="linear")
    lasso_ref.fit(X_train_final, y_train_final)
    y_lasso_pred = lasso_ref.predict(X_test)

    ref_lasso_metrics = {
        "label": "reference, not used for selection",
        "rmse_log": float(rmse_log(y_test, y_lasso_pred)),
        "mae": float(mean_absolute_error(y_test, y_lasso_pred)),
        "mape": float(mean_absolute_percentage_error(y_test, y_lasso_pred) * 100.0),
    }

    # 7. Save model.joblib & metrics.json
    joblib.dump(final_model, models_dir / "model.joblib")
    print(f"Saved fitted final model to {models_dir / 'model.joblib'}")

    metrics_data = {
        "rmse_log": test_rmse_log,
        "mae": test_mae,
        "rmse": test_rmse,
        "mape": test_mape,
        "r2_log": test_r2_log,
        "r2_dollar": test_r2_dollar,
        "bands": band_metrics,
        "reference_lasso": ref_lasso_metrics,
        "final_model_type": model_type,
        "final_model_weights": weights_dict if model_type == "blend" else None,
        "n_train": len(train_val_clean),
        "n_test": len(test_df),
        "git_commit": git_commit,
        "data_md5": data_md5,
        "timestamp": datetime.now().isoformat(),
    }

    metrics_path = models_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved test metrics to {metrics_path}")

    # 8. Save figures
    # Fig 16: Test Pred vs Actual
    fig, ax = plt.subplots(figsize=(8, 6))
    y_test_log = np.log1p(y_test)
    y_pred_log = np.log1p(y_test_pred)
    ax.scatter(y_test_log, y_pred_log, alpha=0.6, color="steelblue", edgecolors="k", linewidth=0.5)
    min_val = min(y_test_log.min(), y_pred_log.min()) - 0.1
    max_val = max(y_test_log.max(), y_pred_log.max()) + 0.1
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal Perfect Prediction (y = x)")
    ax.set_xlabel("Actual SalePrice (log1p)")
    ax.set_ylabel("Predicted SalePrice (log1p)")
    ax.set_title("Test Set: Predicted vs Actual (Log Scale)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "16_test_pred_vs_actual.png", dpi=300)
    plt.close()

    # Fig 17: Test Residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = y_pred_log - y_test_log
    ax.scatter(y_pred_log, residuals, alpha=0.6, color="coral", edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted SalePrice (log1p)")
    ax.set_ylabel("Residual (Predicted log1p - Actual log1p)")
    ax.set_title("Test Set: Residuals vs Predicted Log Price")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "17_test_residuals.png", dpi=300)
    plt.close()

    # Fig 18: Test Error by Band
    fig, ax1 = plt.subplots(figsize=(8, 6))
    b_names = list(band_metrics.keys())
    b_maes = [band_metrics[b]["mae"] for b in b_names]
    b_mapes = [band_metrics[b]["mape"] for b in b_names]

    x = np.arange(len(b_names))
    width = 0.35
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, b_maes, width, label="MAE ($)", color="teal", alpha=0.8)
    ax2.bar(x + width / 2, b_mapes, width, label="MAPE (%)", color="darkorange", alpha=0.8)

    ax1.set_xlabel("Price Band")
    ax1.set_ylabel("MAE ($)", color="teal")
    ax2.set_ylabel("MAPE (%)", color="darkorange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(b_names)
    ax1.set_title("Test Set: Performance Metrics by Price Band")
    ax1.grid(True, linestyle="--", alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(figures_dir / "18_test_error_by_band.png", dpi=300)
    plt.close()

    print("Saved test evaluation figures 16, 17, and 18.")

    # 9. MLflow Logging & Registration
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params.get("mlflow", {}).get("experiment", "lifinity")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = MlflowClient()

    with mlflow.start_run(run_name="final_evaluation"):
        mlflow.set_tags({"git_commit": git_commit, "data_md5": data_md5, "stage": "final"})
        mlflow.log_metrics(
            {
                "test_rmse_log": test_rmse_log,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_mape": test_mape,
                "test_r2_log": test_r2_log,
                "test_r2_dollar": test_r2_dollar,
            }
        )
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(defaults_path))
        mlflow.log_artifact(str(figures_dir / "16_test_pred_vs_actual.png"))
        mlflow.log_artifact(str(figures_dir / "17_test_residuals.png"))
        mlflow.log_artifact(str(figures_dir / "18_test_error_by_band.png"))

        input_example = X_test.head(5)
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="lifinity-price",
            serialization_format="cloudpickle",
        )

        reg_version = str(model_info.registered_model_version)
        client.set_registered_model_alias(
            name="lifinity-price",
            alias="production",
            version=reg_version,
        )
        print(f"Registered model 'lifinity-price' version {reg_version} set to alias 'production'.")

    # 10. Smoke Check
    print("\n" + "=" * 70)
    print("Smoke Check: Loading models/model.joblib and predicting 3 test rows")
    print("=" * 70)
    loaded_model = joblib.load(models_dir / "model.joblib")
    sample_3_x = X_test.head(3)
    sample_3_y_true = y_test.head(3).values
    sample_3_y_pred = loaded_model.predict(sample_3_x)

    for i in range(3):
        print(f"  Row {i+1}: True = ${sample_3_y_true[i]:,.2f} | Predicted = ${sample_3_y_pred[i]:,.2f}")

    # Print summary tables to stdout
    print("\n" + "=" * 70)
    print("TEST METRICS TABLE")
    print("=" * 70)
    print(f"  {'Metric':<25}  {'Value':>15}")
    print(f"  {'-'*25}  {'-'*15}")
    print(f"  {'RMSE(log)':<25}  {test_rmse_log:>15.4f}")
    print(f"  {'MAE ($)':<25}  {'$' + f'{test_mae:,.2f}':>15}")
    print(f"  {'RMSE ($)':<25}  {'$' + f'{test_rmse:,.2f}':>15}")
    print(f"  {'MAPE (%)':<25}  {f'{test_mape:.2f}%':>15}")
    print(f"  {'R2 (log scale)':<25}  {test_r2_log:>15.4f}")
    print(f"  {'R2 ($ scale)':<25}  {test_r2_dollar:>15.4f}")

    print("\nPRICE BAND METRICS TABLE")
    print(f"  {'Price Band':<15}  {'Count':>8}  {'MAE ($)':>15}  {'MAPE (%)':>12}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*15}  {'-'*12}")
    for b_name, b_stats in band_metrics.items():
        mae_str = f"${b_stats['mae']:,.2f}"
        mape_str = f"{b_stats['mape']:.2f}%"
        print(f"  {b_name:<15}  {b_stats['count']:>8}  {mae_str:>15}  {mape_str:>12}")

    print("\nREFERENCE LASSO TEST METRICS")
    print(f"  Label    : {ref_lasso_metrics['label']}")
    print(f"  RMSE(log): {ref_lasso_metrics['rmse_log']:.4f}")
    print(f"  MAE ($)  : ${ref_lasso_metrics['mae']:,.2f}")
    print(f"  MAPE (%) : {ref_lasso_metrics['mape']:.2f}%")

    # 11. Re-run report generation
    print("\nRe-running report generation...")
    generate_report()


if __name__ == "__main__":
    run_evaluation()
