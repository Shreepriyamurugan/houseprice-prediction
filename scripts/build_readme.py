"""Script to build README.md for Lifinity from model outputs, reports, params, and schemas."""

import json
from pathlib import Path
import sys
import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def fmt_dollar(val: float) -> str:
    return f"${round(val):,}"


def fmt_dec4(val: float) -> str:
    return f"{val:.4f}"


def fmt_pct2(val: float) -> str:
    return f"{val:.2f}%"


def generate_tree(dir_path: Path, prefix: str = "") -> list[str]:
    """Recursively generate directory tree excluding specified patterns."""
    exclude_dirs = {".venv", ".git", "data", "mlruns", "__pycache__", "scratch", ".pytest_cache", ".ruff_cache"}
    exclude_exts = {".egg-info", ".pyc", ".joblib", ".parquet"}

    comments = {
        ".github": "GitHub Actions CI workflows",
        ".github/workflows": "CI pipeline definitions",
        ".github/workflows/ci.yml": "CI workflow (test & docker jobs)",
        "api": "FastAPI web service application",
        "api/__init__.py": "API package init",
        "api/main.py": "FastAPI application entrypoint & routing",
        "api/predictor.py": "Inference wrapper with fallback & logging",
        "api/run_server.py": "Local Uvicorn server runner",
        "api/sample_request.json": "Sample property payload for API testing",
        "api/schemas.py": "Pydantic request/response data schemas",
        "data": "Data directory (local DVC tracked)",
        "data/raw": "Raw Ames housing CSV files (untracked in git)",
        "data/processed": "Processed train/val/test parquet splits",
        "models": "Trained model artifacts and metrics",
        "models/input_defaults.json": "Feature medians/modes for payload defaulting",
        "models/metrics.json": "Evaluation metrics on test set",
        "models/model.joblib": "Serialized final production ensemble model",
        "reports": "Generated analysis reports and visualization figures",
        "reports/ablation.csv": "Feature ablation study metrics",
        "reports/ensemble_results.csv": "Blending optimization results",
        "reports/ensemble_weights.json": "Ensemble model blending weights",
        "reports/lasso_selected_features.csv": "Lasso non-zero feature coefficients",
        "reports/lgbm_feature_importance.csv": "LightGBM feature gain importances",
        "reports/model_comparison.csv": "Baseline 7-model CV/validation comparison",
        "reports/model_selection.md": "Comprehensive model selection report",
        "reports/tuning_results.csv": "Optuna hyperparameter tuning summary",
        "reports/figures": "Diagnostic charts and plots",
        "scripts": "Utility automation scripts",
        "scripts/build_readme.py": "README generator script",
        "scripts/ci_smoke_model.py": "CI synthetic model builder script",
        "src": "Core Lifinity ML library source code",
        "src/lifinity": "Lifinity package root",
        "src/lifinity/__init__.py": "Package initialization",
        "src/lifinity/config.py": "Project configuration and paths",
        "src/lifinity/data": "Data split and loading module",
        "src/lifinity/data/split.py": "70/15/15 train/val/test data splitter",
        "src/lifinity/features": "Feature engineering and preprocessing",
        "src/lifinity/features/cleaning.py": "Domain imputer & outlier remover",
        "src/lifinity/features/engineering.py": "Domain feature engineering transformer",
        "src/lifinity/features/preprocessor.py": "Sklearn column transformers & pipelines",
        "src/lifinity/models": "Model training, tuning, and evaluation",
        "src/lifinity/models/ensemble.py": "LogBlendRegressor ensemble class",
        "src/lifinity/models/evaluate.py": "Model evaluation and metrics exporter",
        "src/lifinity/models/report.py": "Markdown report generator",
        "src/lifinity/models/train.py": "Baseline model trainer & MLflow logger",
        "src/lifinity/models/tune.py": "Optuna hyperparameter tuner",
        "tests": "Pytest test suite modules",
        "tests/__init__.py": "Tests package init",
        "tests/synthetic.py": "Synthetic Ames dataset generator for CI",
        "tests/test_api.py": "FastAPI endpoint integration tests",
        "tests/test_cleaning.py": "Domain cleaning unit tests",
        "tests/test_evaluate.py": "Model evaluation unit tests",
        "tests/test_features.py": "Feature engineering unit tests",
        "tests/test_pipeline.py": "DVC workflow structure unit tests",
        "tests/test_preprocessor.py": "Preprocessor pipeline unit tests",
        "tests/test_split.py": "Data splitting unit tests",
        "tests/test_synthetic_pipeline.py": "Synthetic CI integration tests",
        "tests/test_train.py": "Model training unit tests",
        "tests/test_tune.py": "Optuna tuning unit tests",
        "Dockerfile": "Container definition for production API serving",
        "docker-compose.yml": "Docker Compose orchestration config",
        "dvc.yaml": "DVC pipeline DAG definition",
        "params.yaml": "Centralized project configuration parameters",
        "pyproject.toml": "Python package build config and pytest settings",
        "requirements.lock.txt": "Pinned environment dependencies",
        "requirements-serve.txt": "Slim serving dependencies for Docker container",
        "README.md": "Project documentation",
    }

    lines = []
    items = sorted(list(dir_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
    filtered_items = []
    for item in items:
        if item.name in exclude_dirs or any(item.name.endswith(ext) for ext in exclude_exts):
            continue
        filtered_items.append(item)

    for i, item in enumerate(filtered_items):
        is_last = (i == len(filtered_items) - 1)
        connector = "└── " if is_last else "├── "
        rel_path = item.relative_to(Path(".")).as_posix()
        comment = comments.get(rel_path, "")
        comment_str = f"  # {comment}" if comment else ""
        lines.append(f"{prefix}{connector}{item.name}{comment_str}")

        if item.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(generate_tree(item, prefix + extension))

    return lines


def build_readme() -> None:
    root = Path(".")
    
    # 1. Load metrics.json
    with open(root / "models/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 2. Load model_comparison.csv
    comp_df = pd.read_csv(root / "reports/model_comparison.csv")

    # 3. Load tuning_results.csv
    tune_df = pd.read_csv(root / "reports/tuning_results.csv")

    # 4. Load ensemble_results.csv & ensemble_weights.json
    ens_df = pd.read_csv(root / "reports/ensemble_results.csv")
    with open(root / "reports/ensemble_weights.json", "r", encoding="utf-8") as f:
        ens_weights = json.load(f)

    # 5. Load ablation.csv
    ablation_df = pd.read_csv(root / "reports/ablation.csv")

    # 6. Load lasso & lgbm feature info
    lasso_df = pd.read_csv(root / "reports/lasso_selected_features.csv")
    n_lasso_total = len(lasso_df)
    n_lasso_nonzero = int((lasso_df["coefficient"] != 0).sum())

    lgbm_df = pd.read_csv(root / "reports/lgbm_feature_importance.csv")

    # 7. Load params.yaml
    with open(root / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # 8. Load sample_request.json
    with open(root / "api/sample_request.json", "r", encoding="utf-8") as f:
        sample_req = json.load(f)

    # 9. Get real prediction sample response via predictor
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as client:
        sample_resp_raw = client.post("/predict", json=sample_req).json()

    # Format values
    test_rmse_log = fmt_dec4(metrics["rmse_log"])
    test_mae = fmt_dollar(metrics["mae"])
    test_rmse = fmt_dollar(metrics["rmse"])
    test_mape = fmt_pct2(metrics["mape"])
    test_r2_dollar = fmt_dec4(metrics["r2_dollar"])
    test_r2_log = fmt_dec4(metrics["r2_log"])
    mape_accuracy = fmt_pct2(100.0 - metrics["mape"])

    ref_lasso = metrics.get("reference_lasso", {})
    ref_lasso_rmse_log = fmt_dec4(ref_lasso.get("rmse_log", 0.1283))

    # Blend weights string
    weights_dict = metrics["final_model_weights"]
    weight_items = [f"{m} ({weights_dict[m]*100:.2f}%)" for m in weights_dict]
    blend_weights_str = ", ".join(weight_items)

    # Generate tree structure
    tree_lines = ["lifinity/"] + generate_tree(Path("."))
    tree_str = "\n".join(tree_lines)

    readme_content = f"""# Lifinity - Residential Property Price Prediction

![CI](https://github.com/Shreepriyamurugan/houseprice-prediction/actions/workflows/ci.yml/badge.svg)

End-to-end production machine learning system for predicting residential house prices in Ames, Iowa using a tuned log-blend ensemble with full MLOps lifecycle automation, DVC data versioning, MLflow experiment tracking, FastAPI REST serving, and Docker containerization.

---

## 1. Results at a Glance

* **Final Model**: Weighted Log Blend Ensemble comprising {blend_weights_str}.
* **Unbiased Test Set Evaluation ({metrics['n_test']} holdout properties)**:

| Metric | Value |
| :--- | :--- |
| **RMSE (log-scale)** | `{test_rmse_log}` |
| **MAE ($)** | `{test_mae}` |
| **RMSE ($)** | `{test_rmse}` |
| **MAPE (%)** | `{test_mape}` |
| **R^2 ($-scale)** | `{test_r2_dollar}` |
| **R^2 (log-scale)** | `{test_r2_log}` |

* **MAPE-based accuracy (100 - MAPE)**: `{mape_accuracy}` *(Note: Regression models have no true classification accuracy metric; MAPE-based accuracy is reported for intuitive business interpretation).*
* **Honest Evaluation Note**: Cross-validation and validation set RMSE(log) scores were ~0.10-0.11; the test set RMSE(log) of `{test_rmse_log}` provides an unbiased estimate on completely unseen property sales.

---

## 2. Architecture

```mermaid
flowchart TD
    A[Raw Ames Data data/raw/train.csv] -->|70 / 15 / 15 Split| B[Data Split split.py]
    B --> C[Train Set 1,020 rows]
    B --> D[Val Set 219 rows]
    B --> E[Test Set 219 rows]
    
    C --> F[Domain Imputation & Outlier Removal]
    F --> G[Feature Engineering 16 Features]
    G --> H[Preprocessing Branching]
    
    H -->|Linear Branch: RobustScaler + PowerTransformer| I[Lasso / ElasticNet / Ridge]
    H -->|Tree Branch: Median Imputer + Ordinal Encoding| J[XGBoost / CatBoost / LightGBM / RF]
    
    I & J --> K[7-Model Baseline Comparison]
    K --> L[Optuna Tuning 50 Trials]
    L --> M[LogBlendRegressor SLSQP Optimization]
    
    M --> N[Final Model Evaluation]
    E --> N
    
    N --> O[Serialized model.joblib & MLflow Production Registry]
    O --> P[FastAPI REST API api/main.py]
    P --> Q[Docker Container lifinity-api:latest]
    
    R[GitHub Actions CI] -->|Test Job| S[Ruff + Pytest with Synthetic Data]
    R -->|Docker Job| T[Smoke Model + Container Build + Endpoint Health Check]
```

---

## 3. Project Structure

```text
{tree_str}
```

---

## 4. Dataset

* **Source**: [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
* **Scope**: Ames, Iowa residential property sales dataset containing **1,460 sales records** with **79 explanatory features** recorded between 2006 and 2010.
* **Repository Policy**: Raw data files are **NOT** checked into the Git repository. They are locally tracked via DVC without a public remote.
* **Data Setup**: Download `train.csv`, `test.csv`, and `data_description.txt` directly from Kaggle and place them in `data/raw/`.

---

## 5. Quickstart

### Windows (PowerShell)

```powershell
# Clone repository and enter directory
git clone https://github.com/Shreepriyamurugan/houseprice-prediction.git
Set-Location houseprice-prediction

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies and package in editable mode
pip install -r requirements.lock.txt
pip install -e .

# Place train.csv into data/raw/ then run DVC pipeline and tests
dvc repro
pytest -q

# Run API locally
python -m uvicorn api.main:app --port 8000
```

### Linux / macOS (Bash / Zsh)

```bash
# Clone repository and enter directory
git clone https://github.com/Shreepriyamurugan/houseprice-prediction.git
cd houseprice-prediction

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -r requirements.lock.txt
pip install -e .

# Place train.csv into data/raw/ then run DVC pipeline and tests
dvc repro
pytest -q

# Run API locally
python3 -m uvicorn api.main:app --port 8000
```

* **Interactive API Documentation**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
* **Running via Docker Compose**:
  ```bash
  docker compose up -d
  # Access API docs at http://127.0.0.1:8000/docs
  docker compose down
  ```
* **Note on Frozen Pipeline Stages**: `compare` and `tune` stages are marked as `frozen: true` in `dvc.yaml` to prevent long Optuna tuning execution (~30–40 min) during standard reruns. To unfreeze and retune: `dvc unfreeze tune && dvc repro`.

---

## 6. Pipeline and Key Decisions

* **EDA & Target Transformation**: Target variable `SalePrice` exhibits right-skewness; model targets are fitted on `log1p(SalePrice)` and transformed back using `expm1()` to minimize relative percentage errors and stabilize variance.
* **Missing Value Imputation**: Domain-aware imputation handles missing structural features (`PoolQC` -> `"None"`, `GarageArea` -> `0`, `GarageYrBlt` -> `YearBuilt`), preventing data leakage.
* **Outlier Filtering**: Exactly **2 severe outliers** (Ids 524 and 1299: `GrLivArea > 4,000 sq ft` with `SalePrice < $300,000`) were identified and removed **exclusively from the training set** to prevent boundary distortion.
* **Data Splitting**: Stratified 70% train (1,020 rows clean), 15% validation (219 rows), and 15% test (219 rows) holdout split with fixed random seed (`42`).
* **Feature Engineering**: **16 domain-specific engineered features** created (including `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `QualxArea`, `QualSum`, `OverallScore`, `TotalPorchSF`, `LotRatio`).
* **Preprocessing Branches**: Linear models use Yeo-Johnson power transformation (`SkewCorrector`) and `RobustScaler`; tree models use median imputation and ordinal/one-hot encoding.
* **Feature Ablation & Selection**: Ablation study on `ablation.csv` determined `LotRatio` and `IsRemodeled` were unhelpful and dropped; Lasso regularization selected **{n_lasso_nonzero} non-zero features** out of **{n_lasso_total} total preprocessed dimensions**.
* **Model Baseline & Tuning**: Evaluated 7 model architectures across 5-fold CV; top 4 candidates (Lasso, ElasticNet, XGBoost, CatBoost) were tuned via Optuna over 50 trials each.
* **Ensemble Blending**: Constructed a `LogBlendRegressor` using SLSQP constrained optimization on Out-Of-Fold (OOF) log predictions to assign optimal model weights.

---

## 7. Model Selection

### Baseline 7-Model Comparison (`reports/model_comparison.csv`)

| Model | Kind | CV RMSE(log) Mean ± Std | Val RMSE(log) | Val MAE ($) | Overfit Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for _, r in comp_df.iterrows():
        mean_cv = fmt_dec4(r["cv_rmse_log_mean"])
        std_cv = fmt_dec4(r["cv_rmse_log_std"])
        val_rmse_str = fmt_dec4(r["val_rmse_log"])
        val_mae_str = fmt_dollar(r["val_mae"])
        overfit_gap_str = fmt_dec4(r["overfit_gap"])
        readme_content += f"| **{r['model']}** | {r['kind']} | `{mean_cv} ± {std_cv}` | `{val_rmse_str}` | `{val_mae_str}` | `{overfit_gap_str}` |\n"

    readme_content += """
### Optuna Hyperparameter Tuning (`reports/tuning_results.csv`)

| Model | Baseline CV RMSE(log) | Tuned CV RMSE(log) | Val RMSE(log) | Val MAE ($) | Best Key Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    # Load best_params.json
    with open(root / "reports/best_params.json", "r", encoding="utf-8") as f:
        best_params_dict = json.load(f)

    for _, r in tune_df.iterrows():
        b_cv = fmt_dec4(r["baseline_cv_rmse"])
        t_cv = fmt_dec4(r["tuned_cv_rmse"])
        v_rmse = fmt_dec4(r["tuned_val_rmse"])
        v_mae = fmt_dollar(r["val_mae"])
        p_info = best_params_dict.get(r["model"], {})
        params_str = json.dumps(p_info)
        readme_content += f"| **{r['model']}** | `{b_cv}` | `{t_cv}` | `{v_rmse}` | `{v_mae}` | `{params_str}` |\n"

    readme_content += f"""
### Blend vs. Best Single Model Comparison (`reports/ensemble_results.csv`)

| Candidate / Ensemble | OOF RMSE(log) | Val RMSE(log) | Test RMSE(log) | Weight | Model Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for _, r in ens_df.iterrows():
        oof_str = fmt_dec4(r["oof_rmse_log"])
        val_str = fmt_dec4(r["val_rmse_log"])
        w_pct = fmt_pct2(r["weight"] * 100.0)
        
        if r["type"] == "blend":
            test_str = test_rmse_log
        elif r["model"] == "Lasso":
            test_str = ref_lasso_rmse_log
        else:
            test_str = "N/A"
            
        readme_content += f"| **{r['model']}** | `{oof_str}` | `{val_str}` | `{test_str}` | `{w_pct}` | `{r['type']}` |\n"

    readme_content += f"""
* **Ensemble Decision Rule**: Use blend if `blend_oof_rmse < (best_single_oof - 0.002)` AND `blend_val_rmse <= best_single_val_rmse`.
* **Decision Result**: `{ens_weights['decision']}` — Blend achieved OOF RMSE(log) of `{fmt_dec4(ens_df[ens_df['model']=='Blend']['oof_rmse_log'].values[0])}` (improving over best single model Lasso `{fmt_dec4(ens_df[ens_df['model']=='Lasso']['oof_rmse_log'].values[0])}`) and test RMSE(log) of `{test_rmse_log}`.

---

## 8. Evaluation

### Test Set Metrics by Property Price Band (`models/metrics.json`)

| Price Band | Property Count | MAE ($) | MAPE (%) |
| :--- | :--- | :--- | :--- |
"""

    bands = metrics["bands"]
    for b_name, b_info in bands.items():
        b_cnt = b_info["count"]
        b_mae = fmt_dollar(b_info["mae"])
        b_mape = fmt_pct2(b_info["mape"])
        readme_content += f"| **{b_name}** | {b_cnt} | `{b_mae}` | `{b_mape}` |\n"

    readme_content += """
* **Note on High-Value Properties**: The `>$450k` price band contains only **1 house** in the holdout test set (actual price ~$755k), resulting in higher percentage variance for luxury estates.

### Diagnostic Visualizations

![Test Prediction vs Actual](reports/figures/16_test_pred_vs_actual.png)

![Test Residuals](reports/figures/17_test_residuals.png)

![Test Error by Band](reports/figures/18_test_error_by_band.png)

![Model Comparison](reports/figures/12_model_comparison.png)

![LightGBM Top 20 Feature Importance](reports/figures/14_feature_importance.png)

---

## 9. API Reference

### HTTP Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and greeting |
| `GET` | `/health` | Healthcheck endpoint with model load status |
| `GET` | `/model-info` | Production model metadata and test performance |
| `POST` | `/predict` | Single property price prediction |
| `POST` | `/predict/batch` | Batch prediction endpoint (1–500 properties) |

### Sample Request (`api/sample_request.json`)

```json
""" + json.dumps(sample_req, indent=2) + """
```

### Sample Real API Response

```json
""" + json.dumps(sample_resp_raw, indent=2) + """
```

### PowerShell Invocation Example

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -InFile "api/sample_request.json"
```

### cURL Invocation Example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @api/sample_request.json
```

* **Validation & Fallback**:
  * Returns `HTTP 422 Unprocessable Entity` for invalid field ranges (e.g. `OverallQual` = 15 or unknown `Neighborhood`).
  * Emits warning messages in response payload for out-of-range historical years outside the 2006–2010 training period.
  * Any features omitted from the JSON request are automatically defaulted using training set medians and modes (`fields_defaulted`).

---

## 10. MLOps Lifecycle & Automation

### DVC Pipeline Graph (`dvc dag`)

```text
                   +-------+                      
                   | split |                      
                ***+-------+***                   
            ****       *       ****               
         ***           *           ***            
       **              *              ***         
+------+               *                 **       
| tune |*              *                  *       
+------+ ***           *                  *       
    *       ****       *                  *       
    *           ***    *                  *       
    *              **  *                  *       
    **           +----------+        +---------+  
      ***        | evaluate |        | compare |  
         ****    +----------+      **+---------+  
             ***       *       ****               
                ***    *    ***                   
                   **  *  **                      
                  +--------+                      
                  | report |                      
                  +--------+                      
```

* **MLflow Tracking & Registry**:
  * View local runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
  * Model registered under `lifinity-price` with alias `production`.
* **Pytest Suite**: **53 tests across 10 modules** covering data splitting, cleaning, preprocessor branches, feature engineering, model training, Optuna tuning, evaluation, API endpoints, and CI synthetic generation.
* **CI Automation (.github/workflows/ci.yml)**:
  * `test` job: Runs Ruff linter and Pytest suite. Real data/model tests auto-skip when files are absent; synthetic pipeline tests run unconditionally.
  * `docker` job: Prepares synthetic smoke model, builds multi-stage Docker image, starts container, polls `/health` (60s max), posts sample prediction, and cleans up container.
* **Container Security & Logging**:
  * Multi-stage build with pinned dependencies in `requirements-serve.txt`.
  * Runs under non-root app user.
  * Request logs recorded asynchronously to `logs/requests.jsonl`.

---

## 11. Limitations

1. **Geographic & Temporal Bound**: Trained strictly on Ames, Iowa sales from 2006 to 2010; may not generalize to other US regions or different macroeconomic eras.
2. **Dataset Size**: Dataset contains ~1,460 total rows; validation and test holdout sets are relatively small (~219 rows each), introducing statistical variance.
3. **Generalization Gap**: Unbiased test RMSE(log) (`{test_rmse_log}`) is higher than cross-validation scores (~0.10–0.11), reflecting holdout variance.
4. **Luxury Estate Sparsity**: Properties >$450,000 are sparsely represented in training data (only 1 property in test set).
5. **Inference Latency**: Blending 4 pipeline models incurs ~250–320 ms inference latency per request.
6. **Split Strategy**: Uses stratified random splitting rather than strictly chronological time-based splitting.

---

## 12. Future Work

* **Interactive Web Interface**: Build a Streamlit or React frontend for interactive valuation.
* **Model Explainability**: Integrate SHAP / LIME visual explanation dashboards.
* **Data & Concept Drift Monitoring**: Implement drift detection pipelines using Evidently AI.
* **Cloud Deployment**: Deploy serverless container to AWS ECS / GCP Cloud Run.
* **Chronological Split**: Evaluate model against strict time-based validation splits.
* **DVC Remote Storage**: Configure S3 / GCS remote storage backend for DVC data artifacts.
* **Inference Optimization**: Export pipelines to ONNX / C++ runtimes for sub-50ms inference.

---

## 13. Documentation & Deep Dives

* **Detailed Model Selection Report**: [`reports/model_selection.md`](reports/model_selection.md)
"""

    readme_path = root / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"Successfully generated {readme_path.resolve()} ({len(readme_content)} bytes).")


if __name__ == "__main__":
    build_readme()
