"""Model training, 7-model comparison, feature selection, ablation study, and MLflow logging."""

import hashlib
from pathlib import Path
import subprocess
from typing import Any

from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import mlflow
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold, cross_validate
from xgboost import XGBRegressor

from lifinity.config import get_project_root, load_params
from lifinity.features.cleaning import ColumnDropper, DomainImputer
from lifinity.features.engineering import FeatureEngineer
from lifinity.features.preprocessor import (
    RMSE_LOG_SCORER,
    build_full_pipeline,
    build_preprocessor,
    rmse_log,
)


class DropColumns(BaseEstimator, TransformerMixin):
    """Transformer that drops specified columns if present."""

    def __init__(self, cols: tuple[str, ...] | list[str] = ()) -> None:
        self.cols = tuple(cols)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DropColumns":
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()
        existing = [c for c in self.cols if c in X.columns]
        if existing:
            X = X.drop(columns=existing)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            remaining = [c for c in input_features if c not in self.cols]
            return np.array(remaining, dtype=object)
        if hasattr(self, "feature_names_in_"):
            remaining = [c for c in self.feature_names_in_ if c not in self.cols]
            return np.array(remaining, dtype=object)
        raise ValueError("DropColumns not fitted and input_features not provided.")


def build_pipeline_with_ablation(model: Any, kind: str, cols: list[str]) -> Any:
    """Build full pipeline including an optional column ablation step."""
    preprocessor = build_preprocessor(kind=kind)
    target_regressor = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    from sklearn.pipeline import Pipeline
    return Pipeline(
        [
            ("impute", DomainImputer()),
            ("drop", ColumnDropper()),
            ("features", FeatureEngineer()),
            ("ablate", DropColumns(cols=cols)),
            ("prep", preprocessor),
            ("model", target_regressor),
        ]
    )


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(get_project_root()), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_data_md5() -> str:
    root = get_project_root()
    dvc_path = root / "data" / "raw" / "train.csv.dvc"
    if dvc_path.exists():
        import yaml
        with open(dvc_path, "r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)
            if "outs" in dvc_data and dvc_data["outs"]:
                return str(dvc_data["outs"][0].get("md5", ""))
    csv_path = root / "data" / "raw" / "train.csv"
    if csv_path.exists():
        return hashlib.md5(csv_path.read_bytes()).hexdigest()
    return "unknown"


def run_training() -> None:
    root = get_project_root()
    params = load_params()

    # Load data
    train_df = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    val_df = pd.read_parquet(root / "data" / "processed" / "val.parquet")

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]

    X_val = val_df.drop(columns=["SalePrice"], errors="ignore")
    y_val = val_df["SalePrice"]

    # Setup CV
    n_splits = params.get("cv", {}).get("n_splits", 5)
    n_repeats = params.get("cv", {}).get("n_repeats", 3)
    seed = params.get("seed", 42)

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # MLflow setup
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params.get("mlflow", {}).get("experiment", "lifinity")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_commit = get_git_commit()
    data_md5 = get_data_md5()

    # Models list: (name, estimator, kind)
    models = [
        ("Ridge", Ridge(alpha=10.0, random_state=42), "linear"),
        ("Lasso", Lasso(alpha=0.0005, max_iter=50000, random_state=42), "linear"),
        ("ElasticNet", ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=50000, random_state=42), "linear"),
        ("RandomForest", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42), "tree"),
        ("XGBoost", XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, n_jobs=-1, random_state=42), "tree"),
        ("LightGBM", LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=42), "tree"),
        ("CatBoost", CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0, allow_writing_files=False, random_state=42), "tree"),
    ]

    # =========================================================================
    # PART A — 7 Models Comparison
    # =========================================================================
    results = []
    fitted_pipelines = {}

    print("=" * 70)
    print("PART A: 7-Model Training and Cross-Validation")
    print("=" * 70)

    for name, est, kind in models:
        print(f"Evaluating {name} ({kind})...")
        pipe = build_full_pipeline(est, kind=kind)

        # Cross validate on train
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=RMSE_LOG_SCORER, return_train_score=True)
        cv_rmse_mean = float(-cv_res["test_score"].mean())
        cv_rmse_std = float(cv_res["test_score"].std())
        train_rmse_mean = float(-cv_res["train_score"].mean())
        overfit_gap = float(cv_rmse_mean - train_rmse_mean)
        fit_time_mean = float(cv_res["fit_time"].mean())

        # Fit on full train & evaluate on val
        full_pipe = build_full_pipeline(est, kind=kind)
        full_pipe.fit(X_train, y_train)
        fitted_pipelines[name] = full_pipe

        preds_val = full_pipe.predict(X_val)
        val_rmse = float(rmse_log(y_val, preds_val))
        val_mae = float(mean_absolute_error(y_val, preds_val))
        val_r2 = float(r2_score(np.log1p(y_val), np.log1p(preds_val)))
        val_mape = float(mean_absolute_percentage_error(y_val, preds_val))

        results.append(
            {
                "model": name,
                "kind": kind,
                "cv_rmse_log_mean": cv_rmse_mean,
                "cv_rmse_log_std": cv_rmse_std,
                "train_rmse_log_mean": train_rmse_mean,
                "overfit_gap": overfit_gap,
                "val_rmse_log": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2,
                "val_mape": val_mape,
                "fit_time_mean": fit_time_mean,
            }
        )

        # MLflow run
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model_name": name,
                    "kind": kind,
                    "estimator_params": str(est.get_params()),
                    "n_splits": n_splits,
                    "n_repeats": n_repeats,
                    "train_size": params.get("split", {}).get("train_size", 0.70),
                    "val_size": params.get("split", {}).get("val_size", 0.15),
                    "test_size": params.get("split", {}).get("test_size", 0.15),
                }
            )
            mlflow.log_metrics(
                {
                    "cv_rmse_log_mean": cv_rmse_mean,
                    "cv_rmse_log_std": cv_rmse_std,
                    "train_rmse_log_mean": train_rmse_mean,
                    "overfit_gap": overfit_gap,
                    "val_rmse_log": val_rmse,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                    "val_mape": val_mape,
                    "fit_time_mean": fit_time_mean,
                }
            )
            mlflow.set_tags(
                {
                    "git_commit": git_commit,
                    "data_md5": data_md5,
                    "stage": "model_comparison",
                }
            )

    comp_df = pd.DataFrame(results).sort_values(by="cv_rmse_log_mean", ascending=True)

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    comp_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    print("\nModel Comparison Table (sorted by CV RMSE log):")
    print(comp_df[["model", "kind", "cv_rmse_log_mean", "cv_rmse_log_std", "val_rmse_log", "val_mae", "overfit_gap"]].to_string(index=False))

    # Plot Model Comparison Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comp_df))
    ax.errorbar(
        x,
        comp_df["cv_rmse_log_mean"],
        yerr=comp_df["cv_rmse_log_std"],
        fmt="o",
        color="skyblue",
        ecolor="gray",
        elinewidth=2,
        capsize=4,
        label="CV RMSE(log) +/- std",
    )
    ax.scatter(x, comp_df["val_rmse_log"], color="red", zorder=5, label="Validation RMSE(log)")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["model"], rotation=15)
    ax.set_ylabel("RMSE(log)")
    ax.set_title("Model Comparison: CV vs Validation Performance")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "12_model_comparison.png", dpi=300)
    plt.close()

    # =========================================================================
    # PART B — Feature Selection & Importance
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART B: Feature Selection & Importance")
    print("=" * 70)

    # 1. Lasso Selection
    lasso_pipe = fitted_pipelines["Lasso"]
    prep_step = lasso_pipe.named_steps["prep"]
    feature_names = prep_step.get_feature_names_out()
    lasso_coefs = lasso_pipe.named_steps["model"].regressor_.coef_

    lasso_df = pd.DataFrame({"feature": feature_names, "coefficient": lasso_coefs})
    lasso_df["abs_coef"] = lasso_df["coefficient"].abs()
    lasso_df = lasso_df.sort_values(by="abs_coef", ascending=False)

    non_zero_count = int((lasso_df["coefficient"] != 0).sum())
    total_features = len(lasso_df)

    print(f"Lasso non-zero features: {non_zero_count} / {total_features}")
    print("\nTop 10 Lasso features by |coefficient|:")
    print(lasso_df[["feature", "coefficient"]].head(10).to_string(index=False))

    lasso_df.drop(columns=["abs_coef"]).to_csv(reports_dir / "lasso_selected_features.csv", index=False)

    # 2. LightGBM Feature Importance
    lgb_pipe = fitted_pipelines["LightGBM"]
    lgb_model = lgb_pipe.named_steps["model"].regressor_
    lgb_prep_names = lgb_pipe.named_steps["prep"].get_feature_names_out()

    if hasattr(lgb_model, "booster_"):
        gain_importances = lgb_model.booster_.feature_importance(importance_type="gain")
    else:
        gain_importances = lgb_model.feature_importances_

    lgb_imp_df = pd.DataFrame({"feature": lgb_prep_names, "importance_gain": gain_importances})
    lgb_imp_df = lgb_imp_df.sort_values(by="importance_gain", ascending=False)

    print("\nTop 10 LightGBM features by gain importance:")
    print(lgb_imp_df.head(10).to_string(index=False))

    lgb_imp_df.to_csv(reports_dir / "lgbm_feature_importance.csv", index=False)

    # Chart 14: Top 20 LightGBM feature importances
    top20_lgb = lgb_imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20_lgb["feature"], top20_lgb["importance_gain"], color="teal")
    ax.set_xlabel("Gain Importance")
    ax.set_title("LightGBM Top 20 Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(figures_dir / "14_feature_importance.png", dpi=300)
    plt.close()

    # 3. Ablation Study
    print("\nRunning Feature Ablation Study...")
    best_linear_name = comp_df[comp_df["kind"] == "linear"].iloc[0]["model"]
    model_dict = {m[0]: m[1] for m in models}
    best_linear_est = model_dict[best_linear_name]

    engineered_cols = [
        "TotalSF", "TotalBath", "HouseAge", "RemodAge", "IsRemodeled", "IsNew",
        "TotalPorchSF", "QualxArea", "OverallScore", "QualSum",
        "HasPool", "HasGarage", "HasBsmt", "Has2ndFlr", "HasFireplace", "LotRatio",
    ]

    variants = {
        "full": [],
        "minus_LotRatio": ["LotRatio"],
        "minus_IsRemodeled": ["IsRemodeled"],
        "minus_both_weak": ["LotRatio", "IsRemodeled"],
        "no_engineered": engineered_cols,
    }

    ablation_targets = [
        (best_linear_name, best_linear_est, "linear"),
        ("LightGBM", LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, random_state=42), "tree"),
    ]

    ablation_results = []

    for m_name, m_est, m_kind in ablation_targets:
        # Get baseline full score
        full_p = build_pipeline_with_ablation(m_est, kind=m_kind, cols=[])
        full_cv_res = cross_validate(full_p, X_train, y_train, cv=cv, scoring=RMSE_LOG_SCORER)
        full_cv_mean = float(-full_cv_res["test_score"].mean())

        for var_name, drop_cols in variants.items():
            ab_pipe = build_pipeline_with_ablation(m_est, kind=m_kind, cols=drop_cols)
            ab_cv_res = cross_validate(ab_pipe, X_train, y_train, cv=cv, scoring=RMSE_LOG_SCORER)
            ab_cv_mean = float(-ab_cv_res["test_score"].mean())
            ab_cv_std = float(ab_cv_res["test_score"].std())

            ab_pipe.fit(X_train, y_train)
            ab_val_preds = ab_pipe.predict(X_val)
            ab_val_rmse = float(rmse_log(y_val, ab_val_preds))

            delta = float(ab_cv_mean - full_cv_mean)

            ablation_results.append(
                {
                    "model": m_name,
                    "variant": var_name,
                    "cv_rmse_log_mean": ab_cv_mean,
                    "cv_rmse_log_std": ab_cv_std,
                    "val_rmse_log": ab_val_rmse,
                    "delta_vs_full": delta,
                }
            )

            run_label = f"{m_name}__{var_name}"
            with mlflow.start_run(run_name=run_label):
                mlflow.log_params(
                    {
                        "model_name": m_name,
                        "variant": var_name,
                        "dropped_cols": str(drop_cols),
                    }
                )
                mlflow.log_metrics(
                    {
                        "cv_rmse_log_mean": ab_cv_mean,
                        "cv_rmse_log_std": ab_cv_std,
                        "val_rmse_log": ab_val_rmse,
                        "delta_vs_full": delta,
                    }
                )
                mlflow.set_tags(
                    {
                        "git_commit": git_commit,
                        "data_md5": data_md5,
                        "stage": "feature_selection",
                    }
                )

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(reports_dir / "ablation.csv", index=False)

    print("\nAblation Results Table:")
    print(ablation_df.to_string(index=False))

    # Decision on weak features
    print("\n" + "=" * 50)
    print("Weak Feature Decisions (KEEP if delta_vs_full > 0.001 for either model, else DROP):")
    for wf in ["LotRatio", "IsRemodeled"]:
        var_key = f"minus_{wf}"
        max_delta = ablation_df[ablation_df["variant"] == var_key]["delta_vs_full"].max()
        decision = "KEEP" if max_delta > 0.001 else "DROP"
        print(f"  {wf:<15}: {decision} (max CV degradation: +{max_delta:.4f})")

    print("\nEngineered Features Gain (no_engineered CV RMSE - full CV RMSE):")
    for m_name in ablation_df["model"].unique():
        sub = ablation_df[ablation_df["model"] == m_name]
        no_eng_cv = sub[sub["variant"] == "no_engineered"]["cv_rmse_log_mean"].values[0]
        full_cv = sub[sub["variant"] == "full"]["cv_rmse_log_mean"].values[0]
        gain = no_eng_cv - full_cv
        print(f"  {m_name:<15}: +{gain:.4f} RMSE improvement from engineered features")

    # Chart 13: Feature Ablation comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models_in_ab = ablation_df["model"].unique()
    variants_in_ab = list(variants.keys())
    x_indices = np.arange(len(variants_in_ab))
    width = 0.35

    for i, m_name in enumerate(models_in_ab):
        sub = ablation_df[ablation_df["model"] == m_name]
        ax.bar(x_indices + i * width, sub["cv_rmse_log_mean"], width, label=m_name, yerr=sub["cv_rmse_log_std"], capsize=3)

    ax.set_xticks(x_indices + width / 2)
    ax.set_xticklabels(variants_in_ab, rotation=15)
    ax.set_ylabel("CV RMSE(log)")
    ax.set_title("Feature Ablation Study Across Variants")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "13_feature_ablation.png", dpi=300)
    plt.close()

    # =========================================================================
    # PART C — Summary Logging in MLflow
    # =========================================================================
    print("\nLogging summary artifacts to MLflow 'comparison_summary' run...")
    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "data_md5": data_md5,
                "stage": "summary",
            }
        )
        for csv_file in ["model_comparison.csv", "lasso_selected_features.csv", "lgbm_feature_importance.csv", "ablation.csv"]:
            mlflow.log_artifact(str(reports_dir / csv_file))
        for img_file in ["12_model_comparison.png", "13_feature_ablation.png", "14_feature_importance.png"]:
            mlflow.log_artifact(str(figures_dir / img_file))

    print("\nTraining, evaluation, feature selection, and MLflow logging completed successfully.")


if __name__ == "__main__":
    run_training()
