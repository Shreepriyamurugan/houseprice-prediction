"""
Lifinity Project Scaffold Generator
===================================
Automates creation of the complete ML/MLOps project layout for the Lifinity Ames
Housing Price regression project.

Outputs:
- Full directory hierarchy (data/, notebooks/, src/, api/, ui/, tests/, models/, reports/, logs/, .github/)
- Starter configurations: requirements.txt, params.yaml, pyproject.toml, .gitignore, README.md
- Initial empty EDA notebook: notebooks/01_eda.ipynb
"""

import os
import json
from pathlib import Path

# Resolve project root (defaulting to the parent of this script's directory if in scripts/, else current dir)
CURRENT_PATH = Path(__file__).resolve() if "__file__" in globals() else Path(".").resolve()
if CURRENT_PATH.is_file():
    PROJECT_ROOT = CURRENT_PATH.parent.parent if CURRENT_PATH.parent.name == "scripts" else CURRENT_PATH.parent
else:
    PROJECT_ROOT = CURRENT_PATH if CURRENT_PATH.name == "lifinity" else CURRENT_PATH / "lifinity"

PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Target Project Root: {PROJECT_ROOT}")

# Directories to establish
DIRECTORIES = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/lifinity/data",
    "src/lifinity/features",
    "src/lifinity/models",
    "src/lifinity/monitoring",
    "api",
    "ui",
    "tests",
    "models",
    "reports/figures",
    "logs",
    ".github/workflows",
]

for directory in DIRECTORIES:
    (PROJECT_ROOT / directory).mkdir(parents=True, exist_ok=True)

# Empty placeholder files and .gitkeep markers
EMPTY_FILES = [
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "src/lifinity/__init__.py",
    "src/lifinity/config.py",
    "src/lifinity/data/__init__.py",
    "src/lifinity/data/ingest.py",
    "src/lifinity/data/validate.py",
    "src/lifinity/data/split.py",
    "src/lifinity/features/__init__.py",
    "src/lifinity/features/cleaning.py",
    "src/lifinity/features/engineering.py",
    "src/lifinity/features/preprocessor.py",
    "src/lifinity/models/__init__.py",
    "src/lifinity/models/train.py",
    "src/lifinity/models/tune.py",
    "src/lifinity/models/ensemble.py",
    "src/lifinity/models/evaluate.py",
    "src/lifinity/monitoring/__init__.py",
    "src/lifinity/monitoring/drift.py",
    "api/__init__.py",
    "api/main.py",
    "api/schemas.py",
    "api/predictor.py",
    "ui/app.py",
    "tests/__init__.py",
    "tests/test_cleaning.py",
    "tests/test_features.py",
    "tests/test_api.py",
    "models/.gitkeep",
    "reports/figures/.gitkeep",
    "logs/.gitkeep",
    ".github/workflows/ci.yml",
    "dvc.yaml",
    "Dockerfile",
    "docker-compose.yml",
]

for file_path in EMPTY_FILES:
    target_file = PROJECT_ROOT / file_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    if not target_file.exists():
        target_file.touch()

# Starter notebook (01_eda.ipynb)
notebook_data = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Lifinity – EDA"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}
notebook_file = PROJECT_ROOT / "notebooks/01_eda.ipynb"
if not notebook_file.exists():
    with open(notebook_file, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=1)

# requirements.txt
DEPENDENCIES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
    "shap",
    "mlflow",
    "dvc",
    "pandera",
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "streamlit",
    "evidently",
    "matplotlib",
    "seaborn",
    "jupyter",
    "pytest",
    "pytest-cov",
    "ruff",
    "pre-commit",
    "joblib",
    "pyarrow",
]
with open(PROJECT_ROOT / "requirements.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(DEPENDENCIES) + "\n")

# params.yaml (Block-style YAML)
PARAMS_YAML = """seed: 42

split:
  test_size: 0.2
  seed: 42

outliers:
  grlivarea_max: 4000
  saleprice_min: 300000

cv:
  n_splits: 5
  n_repeats: 3

model:
  optuna_trials: 60
"""
with open(PROJECT_ROOT / "params.yaml", "w", encoding="utf-8") as f:
    f.write(PARAMS_YAML)

# pyproject.toml
PYPROJECT_TOML = """[project]
name = "lifinity"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
pythonpath = ["src", "."]
testpaths = ["tests"]
"""
with open(PROJECT_ROOT / "pyproject.toml", "w", encoding="utf-8") as f:
    f.write(PYPROJECT_TOML)

# .gitignore
GITIGNORE = """__pycache__/
*.pyc
.venv/
venv/
.env
.ipynb_checkpoints/
/data/raw/*
!/data/raw/.gitkeep
/data/processed/*
!/data/processed/.gitkeep
/models/*.joblib
mlruns/
mlflow.db
logs/*.jsonl
"""
with open(PROJECT_ROOT / ".gitignore", "w", encoding="utf-8") as f:
    f.write(GITIGNORE)

# README.md
README_CONTENT = """# Lifinity – Residential Property Price Prediction
End-to-end ML + MLOps regression system.
"""
readme_file = PROJECT_ROOT / "README.md"
if not readme_file.exists():
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(README_CONTENT)

print("Scaffold generation successfully finalized!")
