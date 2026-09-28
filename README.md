# Lifinity – Residential Property Price Prediction

End-to-end ML + MLOps regression system for the Ames Housing dataset.

---

## 1. Project Architecture & Directory Layout

The project follows a modular production MLOps design separating raw data, pipelines, model artifacts, reporting, and deployment endpoints:

```
lifinity/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Continuous Integration workflow
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI application entrypoint
│   ├── predictor.py                   # Model inference service
│   └── schemas.py                     # Pydantic request/response schemas
├── data/
│   ├── processed/                     # Engineered feature tables
│   │   └── .gitkeep
│   └── raw/                           # Raw Ames Housing dataset
│       ├── .gitkeep
│       ├── data_description.txt
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── logs/                              # Execution & inference logs
│   └── .gitkeep
├── models/                            # Trained model artifacts (.joblib)
│   └── .gitkeep
├── notebooks/
│   └── 01_eda.ipynb                   # Executed EDA diagnostic notebook
├── reports/
│   └── figures/                       # Generated diagnostic plots (PNG)
│       ├── 02_target_distribution.png
│       ├── 03_missing_values.png
│       ├── 04_outliers_grlivarea_saleprice.png
│       ├── 06_top15_correlations_bar.png
│       ├── 06_top15_correlations_heatmap.png
│       ├── 08_price_bands.png
│       ├── 09_price_and_volume_by_year.png
│       ├── 10_sale_condition.png
│       └── 11_saleprice_by_neighborhood.png
├── scripts/
│   ├── scaffold.py                    # Repository scaffold generator
│   ├── test_eda.py                    # Standalone CLI diagnostic runner
│   └── generate_and_execute_eda.py    # Programmatic notebook builder & executor
├── src/
│   └── lifinity/
│       ├── __init__.py
│       ├── config.py                  # Project paths & hyperparameters
│       ├── data/                      # Ingestion, validation & splitting
│       │   ├── __init__.py
│       │   ├── ingest.py
│       │   ├── split.py
│       │   └── validate.py
│       ├── features/                  # Cleaning & feature engineering
│       │   ├── __init__.py
│       │   ├── cleaning.py
│       │   ├── engineering.py
│       │   └── preprocessor.py
│       ├── models/                    # Training, tuning & evaluation
│       │   ├── __init__.py
│       │   ├── ensemble.py
│       │   ├── evaluate.py
│       │   ├── train.py
│       │   └── tune.py
│       └── monitoring/                # Drift detection & monitoring
│           ├── __init__.py
│           └── drift.py
├── tests/                             # Unit & integration test suite
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_cleaning.py
│   └── test_features.py
├── ui/
│   └── app.py                         # Streamlit interactive dashboard
├── .gitignore
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Multi-service composition
├── dvc.yaml                           # DVC pipeline specification
├── params.yaml                        # Central configuration parameters
├── pyproject.toml                     # Build system & tooling configuration
├── README.md                          # Main project documentation
└── requirements.txt                   # Unpinned Python dependencies
```

---

## 2. Core Automation Scripts (`scripts/`)

Lifinity includes three purpose-built utility scripts under `scripts/`:

### 1. `scripts/scaffold.py`
- **Purpose**: Generates the complete directory tree, `.gitkeep` markers, starter modules, and configuration files (`params.yaml`, `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`).
- **Execution**:
  ```bash
  python scripts/scaffold.py
  ```

### 2. `scripts/test_eda.py`
- **Purpose**: Standalone command-line diagnostic tool. Verifies raw data integrity and computes exact empirical numbers across all 11 diagnostic dimensions without needing to launch Jupyter.
- **Execution**:
  ```bash
  python scripts/test_eda.py
  ```
- **Output**: Detailed terminal report printing shapes, missing count classification, outlier coordinates, skewness tables, correlation coefficients, near-constant category percentages, price bands, and time/location statistics.

### 3. `scripts/generate_and_execute_eda.py`
- **Purpose**: Programmatically builds the entire `notebooks/01_eda.ipynb` notebook from source, executes all 36 code and markdown cells via `nbclient.NotebookClient`, and renders high-resolution figures into `reports/figures/`.
- **Execution**:
  ```bash
  python scripts/generate_and_execute_eda.py
  ```
- **Output**: Populated `notebooks/01_eda.ipynb` with embedded execution counts, stdout tables, and plot images.

---

## 3. Exploratory Data Analysis (EDA) Findings

The EDA performed on the 1,460 observations and 81 features in `data/raw/train.csv` diagnosed key data challenges that dictate downstream pipeline requirements:

### Section-by-Section Key Diagnostics

| Section | Focus Area | Key Metric / Evidence | Production Implication |
| :--- | :--- | :--- | :--- |
| **1. Overview** | Dataset shape & types | 1,460 rows, 81 columns (38 numeric, 43 object), 0 duplicate rows | Clean baseline; high dimensional feature space |
| **2. Target** | `SalePrice` distribution | Raw Skew: **+1.8829** (Mean: $180,921 vs. Median: $163,000)<br>Log1p Skew: **+0.1213** | Target transformation `y = log1p(SalePrice)` is mandatory for linear models & RMSLE optimization |
| **3. Missing Values** | Dual-nature of NAs | **19 missing columns total**:<br>• **16 structural "absent"**: `PoolQC` (99.52%), `MiscFeature` (96.30%), `Alley` (93.77%), `Fence` (80.75%), `FireplaceQu` (47.26%), `MasVnrType` (59.73%), 5 `Garage*` (5.55%), 5 `Bsmt*` (2.53%–2.60%)<br>• **3 true "unknown"**: `LotFrontage` (17.74%, 259 rows), `MasVnrArea` (0.55%, 8 rows), `Electrical` (0.07%, 1 row) | Do **not** drop rows with NAs. Impute structural absence as `'None'` or `0`. Impute `LotFrontage` using median grouped by `Neighborhood`. |
| **4. Outliers** | Living area vs. price | **2 severe partial-build outliers**:<br>• Id 524: 4,676 sqft, sold for $184,750 (`Partial`)<br>• Id 1299: 5,642 sqft, sold for $160,000 (`Partial`) | Prune rows where `GrLivArea > 4000 & SalePrice < 300000` to prevent coefficient distortion (De Cock recommendation). |
| **5. Skewed Features** | Numeric distributions | **22 numeric features** have \|skew\| > 0.75 (top: `MiscVal` +24.48, `PoolArea` +14.83, `LotArea` +12.21, `3SsnPorch` +10.30, `LowQualFinSF` +9.01) | Apply Box-Cox / Yeo-Johnson or `log1p` power transformations during preprocessing. |
| **6. Correlations** | Collinear predictors | Top target correlations: `OverallQual` (0.79), `GrLivArea` (0.71), `GarageCars` (0.64), `GarageArea` (0.62), `TotalBsmtSF` (0.61)<br>Extreme collinearity: `GarageCars` $\leftrightarrow$ `GarageArea` (**r = 0.8825**) | Address variance inflation via L1/L2 regularization (Ridge/Lasso/ElasticNet) or drop redundant garage dimension. |
| **7. Rare Categories** | Zero-variance levels | • **7 columns > 95% single category**: `Utilities` (99.93% AllPub), `Street` (99.59% Pave), `PoolQC` (99.52% NA), `Condition2` (98.97% Norm), `RoofMatl` (98.22% CompShg), `Heating` (97.81% GasA), `MiscFeature` (96.30% NA)<br>• **25 columns** contain **68 rare levels** with <10 rows | Drop near-zero variance features; group rare categorical levels to prevent CV fold instability and test-time unseen label errors. |
| **8. Price Imbalance** | Distribution by band | • <$150k: **615** (42.12%)<br>• $150k–$300k: **730** (50.00%)<br>• $300k–$450k: **101** (6.92%)<br>• >$450k: **14** (0.96%) | 92.12% of homes are <$300k. Stratified K-Fold based on price bands ensures fair validation representation of rare luxury homes. |
| **9. Time Dynamics** | Sales across 2006–2010 | Annual volume 304–338 sales/yr (2006–2009); collapses to 175 in 2010 due to mid-year cut-off. Median price peaked at $167,000 (2007) and dipped to $155,000 (2010). | Capture macro-economic housing cycle and avoid data leakage from incomplete 2010 collection. |
| **10. Sale Condition** | Transaction types | `Normal`: 1,198 sales (82.05%), median $160k<br>`Partial`: 125 sales (8.56%), median $244.6k (**+52.88%** premium)<br>`Abnorml`: 101 sales (6.92%), median $130k (**-18.75%** discount)<br>`AdjLand`: 4 sales (0.27%), median $104k (**-35.00%** discount) | Encode `SaleCondition` as a significant price adjustment indicator. |
| **11. Location** | Neighborhood stratification | 25 neighborhoods. Median prices range from **$88,000** (`MeadowV`) to **$315,000** (`NridgHt`) — a **3.58x price ratio**. | Incorporate target encoding or cluster-based neighborhood aggregations. |

---

## 4. Generated Figure Artifacts (`reports/figures/`)

The following diagnostic charts are generated automatically:
1. `02_target_distribution.png`: Raw vs. Log1p `SalePrice` histograms with KDE curves and mean/median markers.
2. `03_missing_values.png`: Horizontal bar chart of missing percentages color-coded by structural absence vs. unknown data.
3. `04_outliers_grlivarea_saleprice.png`: Scatter plot highlighting the two extreme partial-build outliers in red with callout annotations.
4. `06_top15_correlations_bar.png`: Ranked horizontal bar chart of the top 15 numeric features correlated with `SalePrice`.
5. `06_top15_correlations_heatmap.png`: $16 \times 16$ Pearson correlation heatmap for top features.
6. `08_price_bands.png`: Categorical distribution showing severe concentration in lower and middle price bands.
7. `09_price_and_volume_by_year.png`: Dual-axis plot of median price (line) and transaction volume (bars) from 2006 to 2010.
8. `10_sale_condition.png`: Two-panel bar chart comparing sales volume and median price across sale conditions.
9. `11_saleprice_by_neighborhood.png`: Ranked boxplot showing price distributions across all 25 neighborhoods.
