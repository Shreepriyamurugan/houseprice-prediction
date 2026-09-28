"""
Lifinity – Test EDA Diagnostic Engine
======================================
CLI data diagnostic script for the Ames House Prices dataset.
Computes and validates statistics across 11 key diagnostic dimensions
without modifying data or requiring an interactive notebook session.

Usage:
    python scripts/test_eda.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Dynamically resolve project root and raw dataset path
CURRENT_PATH = Path(__file__).resolve() if "__file__" in globals() else Path(".").resolve()
PROJECT_ROOT = CURRENT_PATH.parent.parent if CURRENT_PATH.parent.name == "scripts" else CURRENT_PATH
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Cannot locate raw training data at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("=" * 60)
print(" LIFINITY – AMES HOUSING DATASET DIAGNOSTIC REPORT")
print("=" * 60)

# 1. Overview
print("\n[SECTION 1: OVERVIEW]")
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
num_cols = df.select_dtypes(include=[np.number]).columns
obj_cols = df.select_dtypes(include=['object']).columns
print(f"Numeric Features: {len(num_cols)} | Categorical Features: {len(obj_cols)}")
print(f"Duplicate Rows Count: {df.duplicated().sum()}")

# 2. Target Variable (SalePrice)
print("\n[SECTION 2: TARGET VARIABLE]")
sp_skew = df['SalePrice'].skew()
log_sp = np.log1p(df['SalePrice'])
log_sp_skew = log_sp.skew()
print(f"SalePrice Skewness:      {sp_skew:.4f} (Positive right-skew)")
print(f"log1p(SalePrice) Skew:   {log_sp_skew:.4f} (Near-normal distribution)")
print(f"SalePrice Mean:         ${df['SalePrice'].mean():,.2f}")
print(f"SalePrice Median:       ${df['SalePrice'].median():,.2f}")

# 3. Missing Values
print("\n[SECTION 3: MISSING VALUES]")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(df)) * 100

absent_patterns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Garage', 'Bsmt', 'MasVnrType']

def classify_na(col):
    for pat in absent_patterns:
        if col.startswith(pat) or pat in col:
            return "absent"
    return "unknown"

na_df = pd.DataFrame({
    'missing_count': missing,
    'missing_pct': missing_pct.round(2),
    'na_meaning': [classify_na(c) for c in missing.index]
})
print(f"Columns with missing values: {len(na_df)}")
print(f"  Structural 'absent' columns: {(na_df['na_meaning'] == 'absent').sum()}")
print(f"  True 'unknown' columns:      {(na_df['na_meaning'] == 'unknown').sum()}")
print(na_df.to_string())

# 4. Outliers
print("\n[SECTION 4: OUTLIERS (GrLivArea > 4000 & SalePrice < $300k)]")
outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)]
print(f"Outliers Count: {len(outliers)}")
print(outliers[['Id', 'GrLivArea', 'SalePrice', 'SaleCondition']].to_string(index=False))

# 5. Skewed Features
print("\n[SECTION 5: SKEWED NUMERIC FEATURES (|skew| > 0.75)]")
num_df = df.select_dtypes(include=[np.number]).drop(columns=['Id'], errors='ignore')
skews = num_df.skew().sort_values(key=abs, ascending=False)
skewed_features = skews[skews.abs() > 0.75]
print(f"Total features with |skew| > 0.75: {len(skewed_features)}")
print(skewed_features.to_string())

# 6. Correlation & Multicollinearity
print("\n[SECTION 6: CORRELATIONS & MULTICOLLINEARITY]")
corr_matrix = num_df.corr()
sale_corr = corr_matrix['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False).head(15)
print("Top 15 Correlated with SalePrice:")
for col in sale_corr.index:
    print(f"  {col:<16}: r = {corr_matrix.loc[col, 'SalePrice']:.4f}")

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        f1, f2 = corr_matrix.columns[i], corr_matrix.columns[j]
        if f1 == 'SalePrice' or f2 == 'SalePrice':
            continue
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.85:
            high_corr_pairs.append((f1, f2, val))

print(f"Feature pairs with severe collinearity (|r| > 0.85): {len(high_corr_pairs)}")
for f1, f2, r in high_corr_pairs:
    print(f"  {f1} <-> {f2}: r = {r:.4f}")

# 7. Rare & Near-Constant Categories
print("\n[SECTION 7: RARE & NEAR-CONSTANT CATEGORIES]")
near_constant = {}
for col in obj_cols:
    top_share = df[col].value_counts(normalize=True, dropna=False).iloc[0] * 100
    top_val = df[col].value_counts(dropna=False).index[0]
    if top_share > 95:
        near_constant[col] = (top_val, top_share)

print(f"Near-Constant Features (> 95% single category, total = {len(near_constant)}):")
for col, (v, sh) in near_constant.items():
    print(f"  {col:<14}: {str(v):<10} ({sh:.2f}%)")

rare_levels_count = 0
rare_cols = set()
for col in obj_cols:
    counts = df[col].value_counts(dropna=True)
    rare = counts[counts < 10]
    if len(rare) > 0:
        rare_levels_count += len(rare)
        rare_cols.add(col)
print(f"Categorical columns with levels having < 10 rows: {len(rare_cols)} of {len(obj_cols)}")
print(f"Total rare levels across all columns: {rare_levels_count}")

# 8. Price Imbalance
print("\n[SECTION 8: PRICE IMBALANCE BANDS]")
bins = [0, 150000, 300000, 450000, float('inf')]
labels = ['<150k', '150-300k', '300-450k', '>450k']
price_bands = pd.cut(df['SalePrice'], bins=bins, labels=labels, right=False)
counts = price_bands.value_counts()[labels]
for band, cnt in counts.items():
    print(f"  {band:<10}: {cnt:>4} houses ({cnt / len(df) * 100:5.2f}%)")

# 9. Time Trend
print("\n[SECTION 9: TIME DYNAMICS (YrSold)]")
yr_grouped = df.groupby('YrSold')['SalePrice'].agg(['count', 'median', 'mean'])
print(yr_grouped.to_string())

# 10. Sale Condition
print("\n[SECTION 10: SALE CONDITION]")
sc_grouped = df.groupby('SaleCondition')['SalePrice'].agg(['count', 'median', 'mean']).sort_values('count', ascending=False)
sc_grouped['share_pct'] = (sc_grouped['count'] / len(df) * 100).round(2)
print(sc_grouped.to_string())

# 11. Location
print("\n[SECTION 11: LOCATION (NEIGHBORHOOD)]")
neigh_summary = df.groupby('Neighborhood')['SalePrice'].agg(['count', 'median']).sort_values('median', ascending=False)
print(f"Total Neighborhoods: {len(neigh_summary)}")
print("Top 3 Affluent Neighborhoods (by Median):")
print(neigh_summary.head(3).to_string())
print("Bottom 3 Affordable Neighborhoods (by Median):")
print(neigh_summary.tail(3).to_string())
ratio = neigh_summary['median'].max() / neigh_summary['median'].min()
print(f"Affordability Ratio (Max / Min Median): {ratio:.2f}x")

print("\n" + "=" * 60)
print(" DIAGNOSTIC RUN COMPLETED SUCCESSFULLY")
print("=" * 60)
