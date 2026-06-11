#!/usr/bin/env python3
"""Analyze the UCI Adult dataset as a baseline/reference.

This script is part of a privacy-preserving computing project. The Adult
dataset is used *only* as a reference to inform the design of a 4-attribute
simulated dataset built around:

    age, education_num, hours_per_week, income

It loads and cleans the raw Adult data, computes descriptive statistics,
correlations (Pearson + Spearman) and pairwise mutual information between the
four baseline attributes, and writes summaries, a clean subset and a handful
of distribution plots.

Run with:

    python src/analyze_adult.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

from sklearn.metrics import mutual_info_score


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve everything relative to the project root (the parent of src/) so the
# script works regardless of the current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def _p(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


ADULT_DATA = _p("data", "external", "adult.data")
ADULT_TEST = _p("data", "external", "adult.test")

OUT_SUBSET_CSV = _p("data", "processed", "adult_clean_subset.csv")
OUT_SUMMARY_TXT = _p("outputs", "phase1", "adult_summary.txt")
OUT_CORR_CSV = _p("outputs", "phase1", "adult_numeric_correlations.csv")
OUT_MI_CSV = _p("outputs", "phase1", "adult_pairwise_mutual_information.csv")

FIG_DIR = _p("reports", "figures")
FIG_INCOME = os.path.join(FIG_DIR, "adult_income_distribution.png")
FIG_AGE = os.path.join(FIG_DIR, "adult_age_distribution.png")
FIG_EDU = os.path.join(FIG_DIR, "adult_education_num_distribution.png")
FIG_HOURS = os.path.join(FIG_DIR, "adult_hours_distribution.png")


# Official Adult column names (in file order).
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_COLS = ["age", "education_num", "hours_per_week"]
BASELINE_COLS = ["age", "education_num", "hours_per_week", "income"]


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def load_raw(path: str, is_test: bool) -> pd.DataFrame:
    """Load one Adult file with the official column names.

    The test file ships with a leading metadata line ("|1x3 Cross validator")
    which we skip, and its income labels carry a trailing ".".
    """
    # The first line of adult.test is metadata; skip it when present.
    skiprows = 0
    if is_test:
        with open(path, "r", encoding="utf-8") as fh:
            first = fh.readline()
        if first.startswith("|"):
            skiprows = 1

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,  # values are written as ", value"
        na_values="?",          # treat "?" as missing
        keep_default_na=True,
        comment=None,
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, normalise income labels, "?"->NaN, drop missing."""
    df = df.copy()

    # Strip whitespace from every string/object column.
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    # "?" can survive as a literal string if skipinitialspace left padding;
    # normalise any remaining ones to NaN.
    df = df.replace("?", np.nan)

    # Remove trailing "." from income labels (present in adult.test).
    if "income" in df.columns:
        df["income"] = df["income"].str.rstrip(".").str.strip()

    # Drop rows with any missing values.
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Binning helpers
# --------------------------------------------------------------------------- #
def age_bins(series: pd.Series) -> pd.Series:
    bins = [0, 25, 35, 45, 55, 65, np.inf]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(series, bins=bins, labels=labels, right=False)


def hours_bins(series: pd.Series) -> pd.Series:
    bins = [0, 20, 35, 40, 45, 60, np.inf]
    labels = ["<20", "20-34", "35-39", "40-44", "45-59", "60+"]
    return pd.cut(series, bins=bins, labels=labels, right=False)


def discretize(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Quantile-bin a numeric series into integer codes for MI on continuous
    variables. Duplicate bin edges are dropped to stay robust."""
    try:
        codes = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = pd.cut(series, bins=min(n_bins, series.nunique()), labels=False)
    return codes.astype("Int64")


# --------------------------------------------------------------------------- #
# Mutual information
# --------------------------------------------------------------------------- #
def pairwise_mutual_information(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mutual information (natural log, in nats) between every pair of cols.

    Continuous numeric columns are quantile-discretised first; categorical
    columns (income) are used as-is.
    """
    # Build a discretised view once.
    disc = {}
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            disc[col] = discretize(df[col]).astype("float").fillna(-1).astype(int)
        else:
            disc[col] = df[col].astype("category").cat.codes

    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            mi = mutual_info_score(disc[a], disc[b])
            rows.append({"var_a": a, "var_b": b, "mutual_information_nats": mi})

    return pd.DataFrame(rows).sort_values(
        "mutual_information_nats", ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_income(df: pd.DataFrame, path: str) -> None:
    counts = df["income"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_title("Adult: income distribution")
    ax.set_xlabel("income")
    ax.set_ylabel("count")
    for x, v in enumerate(counts.values):
        ax.text(x, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hist(series: pd.Series, title: str, xlabel: str, path: str,
              bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(series.values, bins=bins, color="#55A868", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # Ensure output directories exist.
    for d in (
        os.path.dirname(OUT_SUBSET_CSV),
        os.path.dirname(OUT_SUMMARY_TXT),
        FIG_DIR,
    ):
        os.makedirs(d, exist_ok=True)

    # ---- Load -----------------------------------------------------------
    train_raw = load_raw(ADULT_DATA, is_test=False)
    test_raw = load_raw(ADULT_TEST, is_test=True)
    combined_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    shape_before = combined_raw.shape

    # ---- Clean ----------------------------------------------------------
    df = clean(combined_raw)
    shape_after = df.shape

    # ---- Baseline subset ------------------------------------------------
    subset = df[BASELINE_COLS].copy()

    # ---- Statistics -----------------------------------------------------
    summary_stats = subset[NUMERIC_COLS].describe()
    income_dist = subset["income"].value_counts().sort_index()
    income_pct = (subset["income"].value_counts(normalize=True) * 100).sort_index()

    age_dist = age_bins(subset["age"]).value_counts().sort_index()
    edu_dist = subset["education_num"].value_counts().sort_index()
    hours_dist = hours_bins(subset["hours_per_week"]).value_counts().sort_index()

    pearson = subset[NUMERIC_COLS].corr(method="pearson")
    spearman = subset[NUMERIC_COLS].corr(method="spearman")

    mi_table = pairwise_mutual_information(subset, BASELINE_COLS)

    # ---- Console output -------------------------------------------------
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 70)
    emit("UCI Adult dataset — baseline analysis (reference for simulation)")
    emit("=" * 70)
    emit()
    emit(f"Raw shape (train+test combined): {shape_before[0]:,} rows x "
         f"{shape_before[1]} cols")
    emit(f"  - adult.data rows: {train_raw.shape[0]:,}")
    emit(f"  - adult.test rows: {test_raw.shape[0]:,}")
    emit(f"Clean shape (no missing):        {shape_after[0]:,} rows x "
         f"{shape_after[1]} cols")
    emit(f"Rows dropped:                    "
         f"{shape_before[0] - shape_after[0]:,}")
    emit(f"Baseline subset columns:         {BASELINE_COLS}")
    emit()

    emit("-" * 70)
    emit("Summary statistics (age, education_num, hours_per_week)")
    emit("-" * 70)
    emit(summary_stats.to_string())
    emit()

    emit("-" * 70)
    emit("Income distribution")
    emit("-" * 70)
    for label in income_dist.index:
        emit(f"  {label:<8} {income_dist[label]:>8,}  "
             f"({income_pct[label]:5.2f}%)")
    emit()

    emit("-" * 70)
    emit("Age-bin distribution")
    emit("-" * 70)
    for label in age_dist.index:
        emit(f"  {str(label):<8} {age_dist[label]:>8,}")
    emit()

    emit("-" * 70)
    emit("Education-num distribution")
    emit("-" * 70)
    for label in edu_dist.index:
        emit(f"  {label:<3} {edu_dist[label]:>8,}")
    emit()

    emit("-" * 70)
    emit("Hours-per-week bin distribution")
    emit("-" * 70)
    for label in hours_dist.index:
        emit(f"  {str(label):<8} {hours_dist[label]:>8,}")
    emit()

    emit("-" * 70)
    emit("Pearson correlation (numeric columns)")
    emit("-" * 70)
    emit(pearson.to_string())
    emit()

    emit("-" * 70)
    emit("Spearman correlation (numeric columns)")
    emit("-" * 70)
    emit(spearman.to_string())
    emit()

    emit("-" * 70)
    emit("Pairwise mutual information (nats) — age, education_num, "
         "hours_per_week, income")
    emit("-" * 70)
    for _, r in mi_table.iterrows():
        emit(f"  {r['var_a']:<14} <-> {r['var_b']:<14} "
             f"{r['mutual_information_nats']:.4f}")
    emit()

    # ---- Save text summary ---------------------------------------------
    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- Save CSVs ------------------------------------------------------
    subset.to_csv(OUT_SUBSET_CSV, index=False)

    # Combine Pearson + Spearman into one labelled correlation CSV.
    pearson_labeled = pearson.copy()
    pearson_labeled.insert(0, "method", "pearson")
    spearman_labeled = spearman.copy()
    spearman_labeled.insert(0, "method", "spearman")
    corr_out = pd.concat([pearson_labeled, spearman_labeled])
    corr_out.index.name = "variable"
    corr_out.to_csv(OUT_CORR_CSV)

    mi_table.to_csv(OUT_MI_CSV, index=False)

    # ---- Plots ----------------------------------------------------------
    plot_income(subset, FIG_INCOME)
    plot_hist(subset["age"], "Adult: age distribution", "age", FIG_AGE, bins=30)
    plot_hist(subset["education_num"], "Adult: education_num distribution",
              "education_num", FIG_EDU, bins=range(1, 18))
    plot_hist(subset["hours_per_week"], "Adult: hours_per_week distribution",
              "hours_per_week", FIG_HOURS, bins=30)

    emit("Saved outputs:")
    for path in (OUT_SUBSET_CSV, OUT_SUMMARY_TXT, OUT_CORR_CSV, OUT_MI_CSV,
                 FIG_INCOME, FIG_AGE, FIG_EDU, FIG_HOURS):
        emit(f"  - {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
