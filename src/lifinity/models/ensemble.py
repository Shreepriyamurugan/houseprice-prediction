"""Ensemble blending of tuned candidate models."""

import argparse
import json
from typing import Any
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from lifinity.config import get_project_root, load_params
from lifinity.features.preprocessor import build_full_pipeline, rmse_log
from lifinity.models.train import get_data_md5, get_git_commit


class LogBlendRegressor(BaseEstimator, RegressorMixin):
    """Ensemble regressor blending multiple estimators on log1p scale with non-negative weights summing to 1."""

    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        weights: list[float] | np.ndarray | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights

    def fit(self, X: pd.DataFrame, y: Any) -> "LogBlendRegressor":
        if self.weights is None:
            n = len(self.estimators)
            self.weights_ = np.ones(n, dtype=float) / n
        else:
            weights_arr = np.asarray(self.weights, dtype=float)
            if weights_arr.sum() > 0:
                self.weights_ = weights_arr / weights_arr.sum()
            else:
                self.weights_ = np.ones(len(self.estimators), dtype=float) / len(self.estimators)

        self.fitted_estimators_ = []
        for name, est in self.estimators:
            fitted_est = clone(est).fit(X, y)
            self.fitted_estimators_.append((name, fitted_est))

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        log_preds = []
        for i, (name, est) in enumerate(self.fitted_estimators_):
            preds = est.predict(X)
            preds_clipped = np.clip(preds, a_min=0, a_max=None)
            log_preds.append(self.weights_[i] * np.log1p(preds_clipped))

        blended_log = sum(log_preds)
        return np.expm1(blended_log)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        raise ValueError("LogBlendRegressor not fitted and input_features not provided.")


def run_ensemble(use_saved_weights: bool = False) -> None:
    root = get_project_root()
    params = load_params()

    train_df = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    val_df = pd.read_parquet(root / "data" / "processed" / "val.parquet")

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]

    X_val = val_df.drop(columns=["SalePrice"], errors="ignore")
    y_val = val_df["SalePrice"]

    seed = params.get("seed", 42)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Load tuned best params
    params_file = root / "reports" / "best_params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Best params file not found: {params_file}. Run tune.py first.")

    with open(params_file, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    # Build 4 tuned pipelines
    lasso_p = build_full_pipeline(Lasso(**best_params["Lasso"], max_iter=50000, random_state=seed), kind="linear")
    enet_p = build_full_pipeline(ElasticNet(**best_params["ElasticNet"], max_iter=50000, random_state=seed), kind="linear")
    xgb_p = build_full_pipeline(XGBRegressor(**best_params["XGBoost"], n_jobs=-1, random_state=seed), kind="tree")
    cat_p = build_full_pipeline(CatBoostRegressor(**best_params["CatBoost"], verbose=0, allow_writing_files=False, random_state=seed), kind="tree")

    tuned_models = [
        ("Lasso", lasso_p),
        ("ElasticNet", enet_p),
        ("XGBoost", xgb_p),
        ("CatBoost", cat_p),
    ]

    print("=" * 70)
    print("PART 2: Ensemble Blending Optimization")
    print("=" * 70)

    oof_log_preds = []
    single_oof_scores = {}
    single_val_rmse = {}
    single_val_mae = {}
    single_val_mape = {}
    single_val_r2 = {}

    y_train_log = np.log1p(y_train)

    for name, pipe in tuned_models:
        print(f"Generating OOF predictions for {name}...")
        oof_preds = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict")
        oof_preds_clip = np.clip(oof_preds, a_min=0, a_max=None)
        oof_log = np.log1p(oof_preds_clip)
        oof_log_preds.append(oof_log)

        oof_rmse = float(np.sqrt(np.mean((y_train_log - oof_log) ** 2)))
        single_oof_scores[name] = oof_rmse

        # Fit on train & evaluate on val
        pipe_fitted = clone(pipe).fit(X_train, y_train)
        val_preds = pipe_fitted.predict(X_val)
        val_rmse = float(rmse_log(y_val, val_preds))
        single_val_rmse[name] = val_rmse
        single_val_mae[name] = float(mean_absolute_error(y_val, val_preds))
        single_val_mape[name] = float(mean_absolute_percentage_error(y_val, val_preds))
        single_val_r2[name] = float(r2_score(np.log1p(y_val), np.log1p(val_preds)))

    n_models = len(tuned_models)

    def blend_loss(weights: np.ndarray) -> float:
        w_norm = weights / weights.sum()
        blended = sum(w_norm[i] * oof_log_preds[i] for i in range(n_models))
        return float(np.sqrt(np.mean((y_train_log - blended) ** 2)))

    weights_json_file = root / "reports" / "ensemble_weights.json"

    if use_saved_weights and weights_json_file.exists():
        print(f"Loading weights from {weights_json_file}...")
        with open(weights_json_file, "r", encoding="utf-8") as f:
            saved_info = json.load(f)
        saved_w_dict = saved_info["weights"]
        opt_weights = np.array([saved_w_dict[name] for name, _ in tuned_models], dtype=float)
        opt_weights = opt_weights / opt_weights.sum()
        blend_oof_rmse = float(blend_loss(opt_weights))
    else:
        init_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        opt_res = minimize(blend_loss, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = opt_res.x / opt_res.x.sum()
        blend_oof_rmse = float(opt_res.fun)

    # Evaluate blend on val
    blend_model = LogBlendRegressor(estimators=tuned_models, weights=opt_weights)
    blend_model.fit(X_train, y_train)
    blend_val_preds = blend_model.predict(X_val)
    blend_val_rmse = float(rmse_log(y_val, blend_val_preds))
    blend_val_mae = float(mean_absolute_error(y_val, blend_val_preds))
    blend_val_mape = float(mean_absolute_percentage_error(y_val, blend_val_preds))
    blend_val_r2 = float(r2_score(np.log1p(y_val), np.log1p(blend_val_preds)))

    # Identify best single model by OOF RMSE(log)
    best_single_name = min(single_oof_scores, key=single_oof_scores.get)
    best_single_oof = single_oof_scores[best_single_name]
    best_single_val = single_val_rmse[best_single_name]

    # Decision Rule: USE BLEND if blend_oof < (best_single_oof - 0.002) AND blend_val <= best_single_val
    decision = "USE_BLEND" if (blend_oof_rmse < (best_single_oof - 0.002) and blend_val_rmse <= best_single_val) else "USE_SINGLE"

    print("\nSingle Tuned Models vs Blend Results:")
    print(f"  {'Model':<15}  {'OOF RMSE(log)':>14}  {'Val RMSE(log)':>14}  {'Val MAE':>12}  {'Val MAPE':>10}  {'Val R2':>10}  {'Weight':>8}")
    print(f"  {'-'*15}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i, (name, _) in enumerate(tuned_models):
        print(f"  {name:<15}  {single_oof_scores[name]:>14.4f}  {single_val_rmse[name]:>14.4f}  {single_val_mae[name]:>12.2f}  {single_val_mape[name]:>10.4f}  {single_val_r2[name]:>10.4f}  {opt_weights[i]:>8.4f}")
    print(f"  {'-'*15}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(f"  {'Blend':<15}  {blend_oof_rmse:>14.4f}  {blend_val_rmse:>14.4f}  {blend_val_mae:>12.2f}  {blend_val_mape:>10.4f}  {blend_val_r2:>10.4f}  {'1.0000':>8}")

    # Log to MLflow
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params.get("mlflow", {}).get("experiment", "lifinity")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_commit = get_git_commit()
    data_md5 = get_data_md5()

    with mlflow.start_run(run_name="blend"):
        mlflow.set_tags({"git_commit": git_commit, "data_md5": data_md5, "stage": "ensemble"})
        for i, (name, _) in enumerate(tuned_models):
            mlflow.log_metric(f"weight_{name}", float(opt_weights[i]))
            mlflow.log_metric(f"oof_rmse_log_{name}", single_oof_scores[name])
            mlflow.log_metric(f"val_rmse_log_{name}", single_val_rmse[name])

        mlflow.log_metrics(
            {
                "blend_oof_rmse_log": blend_oof_rmse,
                "blend_val_rmse_log": blend_val_rmse,
                "blend_val_mae": blend_val_mae,
                "blend_val_mape": blend_val_mape,
                "blend_val_r2": blend_val_r2,
            }
        )
        mlflow.log_param("decision", decision)
        mlflow.log_param("best_single_model", best_single_name)

    # Save results
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    weights_dict = {name: float(opt_weights[i]) for i, (name, _) in enumerate(tuned_models)}
    with open(reports_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump({"weights": weights_dict, "decision": decision, "best_single": best_single_name}, f, indent=2)

    ens_rows = []
    for name, _ in tuned_models:
        ens_rows.append(
            {
                "model": name,
                "oof_rmse_log": single_oof_scores[name],
                "val_rmse_log": single_val_rmse[name],
                "val_mae": single_val_mae[name],
                "val_mape": single_val_mape[name],
                "val_r2": single_val_r2[name],
                "weight": float(weights_dict[name]),
                "type": "single",
            }
        )
    ens_rows.append(
        {
            "model": "Blend",
            "oof_rmse_log": blend_oof_rmse,
            "val_rmse_log": blend_val_rmse,
            "val_mae": blend_val_mae,
            "val_mape": blend_val_mape,
            "val_r2": blend_val_r2,
            "weight": 1.0,
            "type": "blend",
        }
    )
    ens_df = pd.DataFrame(ens_rows)
    ens_df.to_csv(reports_dir / "ensemble_results.csv", index=False)

    print(f"\nSaved ensemble weights to {reports_dir / 'ensemble_weights.json'}")
    print(f"Saved ensemble results to {reports_dir / 'ensemble_results.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble blending optimization")
    parser.add_argument("--use-saved-weights", action="store_true", help="Use saved weights from ensemble_weights.json")
    args = parser.parse_args()
    run_ensemble(use_saved_weights=args.use_saved_weights)
