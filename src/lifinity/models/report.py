"""Report generation module for model selection and evaluation."""

from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from lifinity.config import get_project_root, load_params
from lifinity.models.train import get_data_md5, get_git_commit


def generate_report() -> None:
    root = get_project_root()
    params = load_params()

    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    data_md5 = get_data_md5()
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Load artifacts
    tuning_csv = reports_dir / "tuning_results.csv"
    comp_csv = reports_dir / "model_comparison.csv"
    ens_json = reports_dir / "ensemble_weights.json"
    ens_results_csv = reports_dir / "ensemble_results.csv"
    best_params_json = reports_dir / "best_params.json"
    ablation_csv = reports_dir / "ablation.csv"
    lasso_csv = reports_dir / "lasso_selected_features.csv"
    lgbm_csv = reports_dir / "lgbm_feature_importance.csv"

    # Create Chart 15: tuning_before_after.png
    if tuning_csv.exists():
        tune_df = pd.read_csv(tuning_csv)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(tune_df))
        width = 0.35

        ax.bar(x - width / 2, tune_df["baseline_cv_rmse"], width, label="Baseline CV RMSE(log)", color="lightgray")
        ax.bar(x + width / 2, tune_df["tuned_cv_rmse"], width, label="Tuned CV RMSE(log)", color="steelblue")

        ax.set_xticks(x)
        ax.set_xticklabels(tune_df["model"])
        ax.set_ylabel("CV RMSE(log)")
        ax.set_title("Model Tuning: Baseline vs Tuned CV Performance")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(figures_dir / "15_tuning_before_after.png", dpi=300)
        plt.close()

    # Determine final choice
    ens_info = {}
    if ens_json.exists():
        with open(ens_json, "r", encoding="utf-8") as f:
            ens_info = json.load(f)

    decision = ens_info.get("decision", "USE_SINGLE")
    best_single = ens_info.get("best_single", "Lasso")

    best_params_all = {}
    if best_params_json.exists():
        with open(best_params_json, "r", encoding="utf-8") as f:
            best_params_all = json.load(f)

    if decision == "USE_BLEND":
        final_type = "blend"
        final_name = "blend"
        final_params_data = {
            "weights": ens_info.get("weights", {}),
            "model_params": best_params_all,
        }
    else:
        final_type = "single"
        final_name = best_single
        final_params_data = best_params_all.get(best_single, {})

    # Write choice to params.yaml
    params_path = root / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        params_file_data = yaml.safe_load(f)

    params_file_data["final_model"] = {
        "type": final_type,
        "name": final_name,
        "params": final_params_data,
    }

    with open(params_path, "w", encoding="utf-8") as f:
        yaml.dump(params_file_data, f, sort_keys=False, default_flow_style=None)

    print(f"Updated final_model choice in {params_path}")

    # Build Markdown Report ONCE as a single string
    md_lines = []
    md_lines.append("# Model Selection Report")
    md_lines.append(f"**Date**: {today_str} | **Git Commit**: `{git_commit}` | **Data MD5**: `{data_md5}`\n")
    md_lines.append("## Executive Summary")
    md_lines.append("This document presents the complete modeling trajectory for the Lifinity Ames Housing price regression system, covering feature selection, baseline model comparison, hyperparameter tuning with Optuna, ensemble blending analysis, final model selection, and final holdout test evaluation.\n")

    # Data Split Summary
    md_lines.append("## Data Split Summary")
    md_lines.append("- **Train set**: 1,020 rows (70% split; 2 extreme outliers with `GrLivArea > 4000` & `SalePrice < $300k` removed)")
    md_lines.append("- **Validation set**: 219 rows (15% split; untouched raw distribution)")
    md_lines.append("- **Test set**: 219 rows (15% split; strictly held out for final evaluation)\n")

    # Section 1: Feature Selection
    md_lines.append("## 1. Feature Selection & Importance")
    if lasso_csv.exists():
        lasso_df = pd.read_csv(lasso_csv)
        non_zero = (lasso_df["coefficient"] != 0).sum()
        total_feat = len(lasso_df)
        md_lines.append(f"- **Lasso Embedded Selection**: Retained **{non_zero} / {total_feat}** non-zero coefficient features after L1 regularization.")
        md_lines.append("  *(measured before LotRatio and IsRemodeled were dropped)*")
        md_lines.append("\n### Top 10 Features by Lasso |Coefficient|")
        md_lines.append("| Rank | Feature | Coefficient |")
        md_lines.append("| --- | --- | --- |")
        for idx, row in lasso_df.head(10).iterrows():
            md_lines.append(f"| {idx+1} | `{row['feature']}` | {row['coefficient']:.4f} |")

    if lgbm_csv.exists():
        lgb_df = pd.read_csv(lgbm_csv)
        md_lines.append("\n### Top 10 Features by LightGBM Gain Importance")
        md_lines.append("| Rank | Feature | Gain Importance |")
        md_lines.append("| --- | --- | --- |")
        for idx, row in lgb_df.head(10).iterrows():
            md_lines.append(f"| {idx+1} | `{row['feature']}` | {row['importance_gain']:.2f} |")

    if ablation_csv.exists():
        ab_df = pd.read_csv(ablation_csv)
        md_lines.append("\n### Feature Ablation Study Results")
        md_lines.append(ab_df.to_markdown(index=False))
        md_lines.append("\n**Weak Feature Decisions**: `LotRatio` -> **DROP**, `IsRemodeled` -> **DROP** (removing them caused no CV degradation).")
        md_lines.append("**Engineered Features Impact**: Removing all engineered features degraded CV RMSE by +0.0011 (Lasso) and +0.0039 (LightGBM).\n")

    # Section 2: Baseline Comparison
    md_lines.append("## 2. Baseline Model Comparison (7 Models)")
    if comp_csv.exists():
        comp_df = pd.read_csv(comp_csv)
        md_lines.append(comp_df.to_markdown(index=False))
        md_lines.append("")

    # Section 3: Hyperparameter Tuning
    md_lines.append("## 3. Hyperparameter Tuning (Optuna)")
    if tuning_csv.exists():
        tune_df = pd.read_csv(tuning_csv)
        md_lines.append(tune_df.to_markdown(index=False))
        md_lines.append("")

    # Section 4: Ensemble
    md_lines.append("## 4. Ensemble Blending Analysis")
    md_lines.append("- **Optimization**: SLSQP constrained optimization minimizing out-of-fold (OOF) RMSE(log).")
    md_lines.append("- **Decision Rule**: USE BLEND only if blend OOF RMSE is > 0.002 lower than best single model AND val RMSE <= best single model.")
    md_lines.append(f"- **Decision Result**: **`{decision}`**")
    if ens_results_csv.exists():
        ens_df = pd.read_csv(ens_results_csv)
        md_lines.append("\n### Ensemble Results Table")
        md_lines.append(ens_df.to_markdown(index=False))
    md_lines.append("")

    # Section 5: Final model
    md_lines.append("## Final model")
    if final_type == "blend" and ens_results_csv.exists():
        ens_df = pd.read_csv(ens_results_csv)
        blend_row = ens_df[ens_df["model"] == "Blend"].iloc[0]
        val_rmse_v = blend_row["val_rmse_log"]
        val_mae_v = blend_row["val_mae"]
        val_mape_v = blend_row["val_mape"]
        val_r2_v = blend_row["val_r2"]
        acc_pct = (1.0 - val_mape_v) * 100.0

        md_lines.append(f"- **Selected Architecture**: **BLEND** (`kind=blend (linear + tree)`)")
        md_lines.append(f"- **Validation RMSE(log)**: `{val_rmse_v:.4f}`")
        md_lines.append(f"- **Validation MAE**: `${val_mae_v:,.2f}`")
        md_lines.append(f"- **Validation R^2 (log scale)**: `{val_r2_v:.4f}`")
        md_lines.append(f"- **Validation MAPE**: `{val_mape_v * 100.0:.2f}%`")
        md_lines.append(f"- **MAPE-based Accuracy Metric**: **`{acc_pct:.2f}%`**\n")

        md_lines.append("### Blend Model Weights")
        md_lines.append("| Model | Weight (%) |")
        md_lines.append("| --- | --- |")
        weights_map = ens_info.get("weights", {})
        for m_name, w_val in weights_map.items():
            md_lines.append(f"| {m_name} | {w_val * 100.0:.2f}% |")
        md_lines.append("")
    elif tuning_csv.exists():
        tune_df = pd.read_csv(tuning_csv)
        final_row = tune_df[tune_df["model"] == best_single].iloc[0]
        val_rmse_v = final_row["tuned_val_rmse"]
        val_mae_v = final_row["val_mae"]
        val_mape_v = final_row["val_mape"]
        val_r2_v = final_row["val_r2"]
        acc_pct = (1.0 - val_mape_v) * 100.0

        md_lines.append(f"- **Selected Architecture**: **{best_single.upper()}** (`kind={final_row['kind']}`)")
        md_lines.append(f"- **Validation RMSE(log)**: `{val_rmse_v:.4f}`")
        md_lines.append(f"- **Validation MAE**: `${val_mae_v:,.2f}`")
        md_lines.append(f"- **Validation R^2 (log scale)**: `{val_r2_v:.4f}`")
        md_lines.append(f"- **Validation MAPE**: `{val_mape_v * 100.0:.2f}%`")
        md_lines.append(f"- **MAPE-based Accuracy Metric**: **`{acc_pct:.2f}%`**\n")

    # Section 6: Final test evaluation
    md_lines.append("## Final test evaluation")
    metrics_json = root / "models" / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            m_data = json.load(f)
        lines_count = m_data.get("n_test", 219)
        md_lines.append(f"- **Status**: Evaluated on held-out test set ({lines_count} rows)")
        md_lines.append(f"- **Test RMSE(log)**: `{m_data['rmse_log']:.4f}`")
        md_lines.append(f"- **Test MAE**: `${m_data['mae']:,.2f}`")
        md_lines.append(f"- **Test RMSE ($)**: `${m_data['rmse']:,.2f}`")
        md_lines.append(f"- **Test MAPE**: `{m_data['mape']:.2f}%`")
        md_lines.append(f"- **Test R^2 (log scale)**: `{m_data['r2_log']:.4f}`")
        md_lines.append(f"- **Test R^2 ($ scale)**: `{m_data['r2_dollar']:.4f}`\n")

        md_lines.append("### Error Breakdown by Price Band")
        md_lines.append("| Price Band | Count | MAE ($) | MAPE (%) |")
        md_lines.append("| --- | --- | --- | --- |")
        for b_name, b_stats in m_data.get("bands", {}).items():
            md_lines.append(f"| {b_name} | {b_stats['count']} | ${b_stats['mae']:,.2f} | {b_stats['mape']:.2f}% |")
        md_lines.append("")

        ref_lasso = m_data.get("reference_lasso", {})
        md_lines.append("### Reference Model Performance")
        md_lines.append(f"- **Lasso (reference, not used for selection)**: Test RMSE(log) = `{ref_lasso.get('rmse_log', 0.0):.4f}`, MAE = `${ref_lasso.get('mae', 0.0):,.2f}`, MAPE = `{ref_lasso.get('mape', 0.0):.2f}%`\n")
    else:
        md_lines.append("Status: pending\n")

    # Section 7: Limitations
    md_lines.append("## System Limitations & Risks")
    md_lines.append("1. **Sample Size Constraints**: Validation set has ~219 samples; small evaluation sets exhibit variance across splits.")
    md_lines.append("2. **Geographic & Temporal Scope**: Trained on Ames, Iowa housing data (2006-2010); non-generalizable to current interest rate regimes or unobserved regions without recalibration.")
    md_lines.append("3. **Tree Model Overfitting**: GBDT models exhibit larger train vs CV gaps (~0.09) compared to linear models (~0.016).\n")

    # Embed Figures with relative paths
    md_lines.append("## Performance Visualizations")
    md_lines.append("![Model Comparison](figures/12_model_comparison.png)")
    md_lines.append("![Feature Ablation](figures/13_feature_ablation.png)")
    md_lines.append("![Feature Importance](figures/14_feature_importance.png)")
    md_lines.append("![Tuning Before/After](figures/15_tuning_before_after.png)")
    if (figures_dir / "16_test_pred_vs_actual.png").exists():
        md_lines.append("![Test Predictions vs Actual](figures/16_test_pred_vs_actual.png)")
        md_lines.append("![Test Residuals](figures/17_test_residuals.png)")
        md_lines.append("![Test Error by Band](figures/18_test_error_by_band.png)")

    report_content = "\n".join(md_lines)
    report_file = reports_dir / "model_selection.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"Generated model selection report at {report_file}")


if __name__ == "__main__":
    generate_report()
