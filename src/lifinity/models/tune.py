"""Hyperparameter tuning with Optuna for top candidate models."""

import json
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from xgboost import XGBRegressor

from lifinity.config import get_project_root, load_params
from lifinity.features.preprocessor import (
    RMSE_LOG_SCORER,
    build_full_pipeline,
    rmse_log,
)
from lifinity.models.train import get_data_md5, get_git_commit

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_tuning() -> None:
    root = get_project_root()
    params = load_params()

    train_df = pd.read_parquet(root / "data" / "processed" / "train.parquet")
    val_df = pd.read_parquet(root / "data" / "processed" / "val.parquet")

    X_train = train_df.drop(columns=["SalePrice"], errors="ignore")
    y_train = train_df["SalePrice"]

    X_val = val_df.drop(columns=["SalePrice"], errors="ignore")
    y_val = val_df["SalePrice"]

    seed = params.get("seed", 42)
    tune_config = params.get("tune", {})
    cv_folds = tune_config.get("cv_folds", 5)
    trials_cfg = tune_config.get("trials", {"Lasso": 60, "ElasticNet": 60, "XGBoost": 40, "CatBoost": 25})
    timeout_sec = tune_config.get("timeout_sec", 900)

    # 1-repeat 5-fold CV for tuning objective
    tuning_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Full 5x3 RepeatedKFold for final re-scoring
    n_splits = params.get("cv", {}).get("n_splits", 5)
    n_repeats = params.get("cv", {}).get("n_repeats", 3)
    eval_cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = params.get("mlflow", {}).get("experiment", "lifinity")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    git_commit = get_git_commit()
    data_md5 = get_data_md5()

    models_to_tune = ["Lasso", "ElasticNet", "XGBoost", "CatBoost"]
    best_params_dict = {}
    tuning_results = []

    # Load baseline comparison scores for before vs after table
    comp_file = root / "reports" / "model_comparison.csv"
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
        baseline_map = comp_df.set_index("model").to_dict(orient="index")
    else:
        baseline_map = {}

    print("=" * 70)
    print("PART 1: Optuna Hyperparameter Tuning")
    print("=" * 70)

    for model_name in models_to_tune:
        n_trials = trials_cfg.get(model_name, 30)
        print(f"\nTuning {model_name} ({n_trials} trials)...")

        kind = "linear" if model_name in ["Lasso", "ElasticNet"] else "tree"

        def objective(trial: optuna.Trial) -> float:
            if model_name == "Lasso":
                alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
                est = Lasso(alpha=alpha, max_iter=50000, random_state=seed)
            elif model_name == "ElasticNet":
                alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
                est = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000, random_state=seed)
            elif model_name == "XGBoost":
                n_est = trial.suggest_int("n_estimators", 400, 2000)
                lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
                depth = trial.suggest_int("max_depth", 2, 6)
                min_child = trial.suggest_int("min_child_weight", 1, 10)
                subsample = trial.suggest_float("subsample", 0.5, 1.0)
                colsample = trial.suggest_float("colsample_bytree", 0.3, 1.0)
                reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
                reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)
                est = XGBRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=depth,
                    min_child_weight=min_child,
                    subsample=subsample,
                    colsample_bytree=colsample,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                    n_jobs=-1,
                    random_state=seed,
                )
            elif model_name == "CatBoost":
                iters = trial.suggest_int("iterations", 500, 2000)
                lr = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
                depth = trial.suggest_int("depth", 4, 8)
                l2_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True)
                rand_str = trial.suggest_float("random_strength", 0.0, 2.0)
                bag_temp = trial.suggest_float("bagging_temperature", 0.0, 1.0)
                est = CatBoostRegressor(
                    iterations=iters,
                    learning_rate=lr,
                    depth=depth,
                    l2_leaf_reg=l2_reg,
                    random_strength=rand_str,
                    bagging_temperature=bag_temp,
                    verbose=0,
                    allow_writing_files=False,
                    random_state=seed,
                )

            pipe = build_full_pipeline(est, kind=kind)
            scores = cross_validate(pipe, X_train, y_train, cv=tuning_cv, scoring=RMSE_LOG_SCORER)
            return float(-scores["test_score"].mean())

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(),
        )

        with mlflow.start_run(run_name=f"tune__{model_name}") as parent_run:
            mlflow.set_tags({"git_commit": git_commit, "data_md5": data_md5, "stage": "tuning"})

            def mlflow_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    mlflow.log_params(trial.params)
                    if trial.value is not None:
                        mlflow.log_metric("rmse_log", trial.value)

            study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, callbacks=[mlflow_callback])

            best_params = study.best_params
            best_params_dict[model_name] = best_params

            # Re-instantiate best estimator
            if model_name == "Lasso":
                best_est = Lasso(**best_params, max_iter=50000, random_state=seed)
            elif model_name == "ElasticNet":
                best_est = ElasticNet(**best_params, max_iter=50000, random_state=seed)
            elif model_name == "XGBoost":
                best_est = XGBRegressor(**best_params, n_jobs=-1, random_state=seed)
            elif model_name == "CatBoost":
                best_est = CatBoostRegressor(**best_params, verbose=0, allow_writing_files=False, random_state=seed)

            # Re-score on full 5x3 RepeatedKFold
            eval_pipe = build_full_pipeline(best_est, kind=kind)
            res_eval = cross_validate(eval_pipe, X_train, y_train, cv=eval_cv, scoring=RMSE_LOG_SCORER)
            tuned_cv_mean = float(-res_eval["test_score"].mean())
            tuned_cv_std = float(res_eval["test_score"].std())

            # Fit on full train & evaluate on val
            eval_pipe.fit(X_train, y_train)
            preds_val = eval_pipe.predict(X_val)
            val_rmse = float(rmse_log(y_val, preds_val))
            val_mae = float(mean_absolute_error(y_val, preds_val))
            val_r2 = float(r2_score(np.log1p(y_val), np.log1p(preds_val)))
            val_mape = float(mean_absolute_percentage_error(y_val, preds_val))

            base_info = baseline_map.get(model_name, {})
            base_cv_rmse = base_info.get("cv_rmse_log_mean", np.nan)
            base_val_rmse = base_info.get("val_rmse_log", np.nan)

            tuning_results.append(
                {
                    "model": model_name,
                    "kind": kind,
                    "baseline_cv_rmse": base_cv_rmse,
                    "tuned_cv_rmse": tuned_cv_mean,
                    "tuned_cv_std": tuned_cv_std,
                    "baseline_val_rmse": base_val_rmse,
                    "tuned_val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_mape": val_mape,
                    "val_r2": val_r2,
                }
            )

            # Log parent metrics & params
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_params({"n_trials_completed": len(study.trials), "model_name": model_name})
            mlflow.log_metrics(
                {
                    "tuned_cv_rmse_log_mean": tuned_cv_mean,
                    "tuned_cv_rmse_log_std": tuned_cv_std,
                    "val_rmse_log": val_rmse,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                    "val_mape": val_mape,
                }
            )

            # Optuna plot artifact
            fig_dir = root / "reports" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_path = fig_dir / f"optuna_{model_name.lower()}.png"

            try:
                fig_opt = optuna.visualization.matplotlib.plot_optimization_history(study)
                fig_opt.figure.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig_opt.figure)
                mlflow.log_artifact(str(plot_path))
            except Exception as e:
                print(f"Could not save Optuna plot for {model_name}: {e}")

            print(f"  Best params: {best_params}")
            print(f"  CV RMSE(log): {tuned_cv_mean:.4f} ± {tuned_cv_std:.4f} (baseline: {base_cv_rmse:.4f})")
            print(f"  Val RMSE(log): {val_rmse:.4f} (baseline: {base_val_rmse:.4f})")

    # Save best params json & tuning results csv
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params_dict, f, indent=2)

    tune_df = pd.DataFrame(tuning_results)
    tune_df.to_csv(reports_dir / "tuning_results.csv", index=False)

    print("\nTuning Results Summary:")
    print(tune_df[["model", "baseline_cv_rmse", "tuned_cv_rmse", "baseline_val_rmse", "tuned_val_rmse", "val_mae"]].to_string(index=False))
    print(f"\nSaved best params to {reports_dir / 'best_params.json'}")
    print(f"Saved tuning results to {reports_dir / 'tuning_results.csv'}")


if __name__ == "__main__":
    run_tuning()
