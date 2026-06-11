#!/usr/bin/env python3
"""Generate Adult-like simulated datasets for the main privacy experiment.

The UCI Adult dataset is used ONLY as a statistical reference/baseline (see
``src/analyze_adult.py``). The datasets produced here are the actual data used
for the privacy experiments: k-anonymity, l-diversity, t-closeness, and both
non-private and private PrivBayes.

Schema (4 attributes):
    age            quasi-identifier, integer  [18, 80]
    education_num  quasi-identifier, integer  [6, 16]
    hours_per_week quasi-identifier, integer  [1, 70]
    income_class   sensitive, categorical ordinal
                   (Low < LowerMiddle < UpperMiddle < High < VeryHigh < Extreme)

Attributes are generated with an explicit dependency chain so that the joint
distribution carries realistic, non-trivial mutual information:

    age -> education_num -> hours_per_week -> income_class

Run with:

    python src/generate_simulated_data.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def _p(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


RAW_DIR = _p("data", "raw")
PHASE1_DIR = _p("outputs", "phase1")

OUT_METADATA = os.path.join(RAW_DIR, "phase1_metadata.json")
OUT_SUMMARY_TXT = os.path.join(PHASE1_DIR, "generated_summary.txt")
OUT_INCOME_CSV = os.path.join(PHASE1_DIR, "generated_income_distributions.csv")
OUT_CORR_CSV = os.path.join(PHASE1_DIR, "generated_numeric_correlations.csv")
OUT_MI_CSV = os.path.join(PHASE1_DIR, "generated_pairwise_mi.csv")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 0
DATASET_SIZES = [3000, 5000, 10000]

QUASI_IDENTIFIERS = ["age", "education_num", "hours_per_week"]
SENSITIVE_ATTR = "income_class"
NUMERIC_COLS = ["age", "education_num", "hours_per_week"]
ALL_COLS = NUMERIC_COLS + [SENSITIVE_ATTR]

# Age bands: (low, high_inclusive, probability).
AGE_BANDS = [
    (18, 25, 0.16),
    (26, 35, 0.24),
    (36, 50, 0.31),
    (51, 65, 0.22),
    (66, 80, 0.07),
]

# Additive effect of the (coarse) age band on weekly working hours.
AGE_WORK_EFFECT = [
    (18, 25, -5.0),
    (26, 35, 3.0),
    (36, 50, 5.0),
    (51, 65, 1.0),
    (66, 80, -12.0),
]

# Income classes in ascending order and their TARGET proportions.
INCOME_CLASSES = ["Low", "LowerMiddle", "UpperMiddle", "High", "VeryHigh",
                  "Extreme"]
INCOME_TARGET_PROBS = {
    "Low": 0.28,
    "LowerMiddle": 0.34,
    "UpperMiddle": 0.24,
    "High": 0.10,
    "VeryHigh": 0.037,
    "Extreme": 0.003,
}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _age_band_effect(age: np.ndarray, table) -> np.ndarray:
    """Map each age to the additive effect of its band."""
    effect = np.zeros(len(age), dtype=float)
    for low, high, val in table:
        effect[(age >= low) & (age <= high)] = val
    return effect


def generate_age(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample an age band, then sample uniformly inside the chosen band."""
    probs = np.array([b[2] for b in AGE_BANDS], dtype=float)
    probs = probs / probs.sum()  # guard against rounding
    band_idx = rng.choice(len(AGE_BANDS), size=n, p=probs)

    age = np.empty(n, dtype=int)
    for i, (low, high, _) in enumerate(AGE_BANDS):
        mask = band_idx == i
        if mask.any():
            # high is inclusive -> randint upper bound is high + 1
            age[mask] = rng.integers(low, high + 1, size=mask.sum())
    return age


def generate_education_num(age: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """education_num as a (noisy) quadratic function of age, clipped to [6,16]."""
    noise = rng.normal(0.0, 2.0, size=len(age))
    edu = (10.5
           + 0.035 * (age - 35)
           - 0.0012 * (age - 45) ** 2
           + noise)
    edu = np.rint(edu).astype(int)
    return np.clip(edu, 6, 16)


def generate_hours(age: np.ndarray, education_num: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """hours_per_week from age band + education, clipped to [1, 70]."""
    age_effect = _age_band_effect(age, AGE_WORK_EFFECT)
    noise = rng.normal(0.0, 7.0, size=len(age))
    # Intercept raised from the reference 34 -> 38.8 so the realized mean lands
    # mid-target (38-43); the age/education/noise terms are unchanged.
    hours = (38.8
             + 0.7 * (education_num - 10)
             + age_effect
             + noise)
    hours = np.rint(hours).astype(int)
    return np.clip(hours, 1, 70)


def generate_income_class(age: np.ndarray, education_num: np.ndarray,
                          hours_per_week: np.ndarray,
                          rng: np.random.Generator):
    """Latent income score -> ordinal income_class via quantile thresholds.

    Thresholds are calibrated per dataset to the cumulative target
    proportions, which keeps the realized distribution close to target while
    remaining a deterministic function of the latent score (so each
    equivalence class does NOT automatically contain every class, and Extreme
    stays rare but present).
    """
    noise = rng.normal(0.0, 1.4, size=len(age))
    score = (-5.0
             + 0.055 * age
             + 0.60 * education_num
             + 0.050 * hours_per_week
             + noise)

    # Cumulative cut points (drop the final 1.0 -> open-ended top class).
    cum = np.cumsum([INCOME_TARGET_PROBS[c] for c in INCOME_CLASSES])[:-1]
    thresholds = np.quantile(score, cum)

    # np.digitize -> index 0..len(classes)-1 in ascending score order.
    idx = np.digitize(score, thresholds)
    labels = np.array(INCOME_CLASSES)[idx]
    return labels, score, thresholds


def generate_dataset(n: int, seed: int) -> pd.DataFrame:
    """Generate one Adult-like dataset following the dependency chain."""
    rng = np.random.default_rng(seed)
    age = generate_age(n, rng)
    education_num = generate_education_num(age, rng)
    hours_per_week = generate_hours(age, education_num, rng)
    income_class, _, _ = generate_income_class(
        age, education_num, hours_per_week, rng)

    df = pd.DataFrame({
        "age": age,
        "education_num": education_num,
        "hours_per_week": hours_per_week,
        "income_class": pd.Categorical(
            income_class, categories=INCOME_CLASSES, ordered=True),
    })
    return df


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def discretize(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Quantile-bin a numeric series into integer codes for MI."""
    try:
        codes = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = pd.cut(series, bins=min(n_bins, series.nunique()), labels=False)
    return pd.Series(codes).fillna(-1).astype(int).to_numpy()


def pairwise_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    """MI (nats) between every pair of the 4 attributes; numeric ones binned."""
    disc = {}
    for col in ALL_COLS:
        if pd.api.types.is_numeric_dtype(df[col]):
            disc[col] = discretize(df[col])
        else:
            disc[col] = df[col].astype("category").cat.codes.to_numpy()

    rows = []
    for i in range(len(ALL_COLS)):
        for j in range(i + 1, len(ALL_COLS)):
            a, b = ALL_COLS[i], ALL_COLS[j]
            mi = mutual_info_score(disc[a], disc[b])
            rows.append({"var_a": a, "var_b": b, "mutual_information_nats": mi})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, n: int, emit) -> dict:
    """Print + collect validation stats for one dataset. Returns a dict of
    DataFrames used for the aggregated CSV summaries."""
    emit("=" * 70)
    emit(f"Validation - simulated_adult_like_n{n}_seed{SEED}")
    emit("=" * 70)

    emit(f"shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    emit(f"missing values (total): {int(df.isna().sum().sum())}")
    emit()

    # Numeric stats.
    stats = df[NUMERIC_COLS].agg(["min", "max", "mean", "std"]).T
    emit("Numeric quasi-identifier statistics:")
    emit(stats.to_string())
    emit()

    # Income distribution.
    counts = df[SENSITIVE_ATTR].value_counts().reindex(INCOME_CLASSES).fillna(0)
    counts = counts.astype(int)
    props = counts / counts.sum()
    income_tbl = pd.DataFrame({
        "income_class": INCOME_CLASSES,
        "count": counts.values,
        "proportion": props.values,
        "target_proportion": [INCOME_TARGET_PROBS[c] for c in INCOME_CLASSES],
    })
    emit("income_class distribution (sensitive attribute):")
    for _, r in income_tbl.iterrows():
        emit(f"  {r['income_class']:<12} {int(r['count']):>7,}  "
             f"obs={r['proportion']:.4f}  target={r['target_proportion']:.4f}")
    emit()

    # Correlations.
    pearson = df[NUMERIC_COLS].corr(method="pearson")
    spearman = df[NUMERIC_COLS].corr(method="spearman")
    emit("Pearson correlation (numeric QIs):")
    emit(pearson.to_string())
    emit()
    emit("Spearman correlation (numeric QIs):")
    emit(spearman.to_string())
    emit()

    # Mutual information.
    mi = pairwise_mutual_information(df)
    mi_sorted = mi.sort_values("mutual_information_nats", ascending=False)
    emit("Pairwise mutual information (nats) - all 4 attributes:")
    for _, r in mi_sorted.iterrows():
        emit(f"  {r['var_a']:<14} <-> {r['var_b']:<14} "
             f"{r['mutual_information_nats']:.4f}")
    emit()

    # Tag with dataset size for the aggregated CSVs.
    income_tbl.insert(0, "n", n)
    pearson_l = pearson.copy()
    pearson_l.insert(0, "method", "pearson")
    pearson_l.insert(0, "n", n)
    spearman_l = spearman.copy()
    spearman_l.insert(0, "method", "spearman")
    spearman_l.insert(0, "n", n)
    corr_tbl = pd.concat([pearson_l, spearman_l])
    corr_tbl.index.name = "variable"
    mi = mi.copy()
    mi.insert(0, "n", n)

    return {"income": income_tbl, "corr": corr_tbl, "mi": mi}


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def build_metadata() -> dict:
    return {
        "baseline_dataset": "UCI Adult",
        "explanation": (
            "The UCI Adult dataset was used ONLY as a statistical reference "
            "to calibrate marginal ranges/means and the sign/strength of "
            "dependencies. No Adult records are present in these datasets. "
            "The simulated data below is the dataset actually used for the "
            "privacy experiments (k-anonymity, l-diversity, t-closeness, "
            "non-private PrivBayes, private PrivBayes)."
        ),
        "schema": {
            "age": {"type": "integer", "range": [18, 80],
                    "role": "quasi-identifier"},
            "education_num": {"type": "integer", "range": [6, 16],
                              "role": "quasi-identifier"},
            "hours_per_week": {"type": "integer", "range": [1, 70],
                               "role": "quasi-identifier"},
            "income_class": {"type": "categorical_ordinal",
                             "categories": INCOME_CLASSES,
                             "role": "sensitive"},
        },
        "quasi_identifiers": QUASI_IDENTIFIERS,
        "sensitive_attribute": SENSITIVE_ATTR,
        "income_class_target_probabilities": INCOME_TARGET_PROBS,
        "dataset_sizes": DATASET_SIZES,
        "random_seed": SEED,
        "dependency_structure": {
            "order": ["age", "education_num", "hours_per_week", "income_class"],
            "edges": [
                "age -> education_num",
                "age -> hours_per_week",
                "education_num -> hours_per_week",
                "age -> income_class",
                "education_num -> income_class",
                "hours_per_week -> income_class",
            ],
            "notes": (
                "Attributes are NOT independent. education_num is a noisy "
                "quadratic function of age; hours_per_week depends on age band "
                "and education_num; income_class is derived from a latent score "
                "that is a linear function of all three quasi-identifiers, then "
                "mapped to ordinal classes via quantile thresholds calibrated "
                "to the target proportions."
            ),
        },
        "age_bands": [
            {"range": [b[0], b[1]], "probability": b[2]} for b in AGE_BANDS
        ],
        "output_files": [
            f"data/raw/simulated_adult_like_n{n}_seed{SEED}.csv"
            for n in DATASET_SIZES
        ],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PHASE1_DIR, exist_ok=True)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    income_parts, corr_parts, mi_parts = [], [], []

    for n in DATASET_SIZES:
        df = generate_dataset(n, SEED)

        out_csv = os.path.join(RAW_DIR,
                               f"simulated_adult_like_n{n}_seed{SEED}.csv")
        df.to_csv(out_csv, index=False)

        result = validate(df, n, emit)
        income_parts.append(result["income"])
        corr_parts.append(result["corr"])
        mi_parts.append(result["mi"])

        emit(f"saved: {os.path.relpath(out_csv, PROJECT_ROOT)}")
        emit()

    # ---- Aggregated CSV summaries --------------------------------------
    pd.concat(income_parts, ignore_index=True).to_csv(OUT_INCOME_CSV,
                                                       index=False)
    pd.concat(corr_parts).to_csv(OUT_CORR_CSV)
    pd.concat(mi_parts, ignore_index=True).to_csv(OUT_MI_CSV, index=False)

    with open(OUT_SUMMARY_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---- Metadata JSON --------------------------------------------------
    with open(OUT_METADATA, "w", encoding="utf-8") as fh:
        json.dump(build_metadata(), fh, indent=2)

    emit("Saved summaries:")
    for path in (OUT_SUMMARY_TXT, OUT_INCOME_CSV, OUT_CORR_CSV, OUT_MI_CSV,
                 OUT_METADATA):
        emit(f"  - {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
