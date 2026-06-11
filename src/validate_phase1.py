#!/usr/bin/env python3
"""Independent validation of the Phase 1 generated datasets.

This script deliberately does NOT import or reuse anything from
``src/generate_simulated_data.py``. It re-derives every expectation from the
written spec so that it acts as a genuine, independent check on the CSV files
that were produced.

Run with:

    python src/validate_phase1.py
"""

from __future__ import annotations

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

OUT_REPORT_TXT = os.path.join(PHASE1_DIR, "generated_validation_report.txt")
OUT_PASSFAIL_CSV = os.path.join(PHASE1_DIR, "generated_validation_pass_fail.csv")
OUT_MI_CSV = os.path.join(PHASE1_DIR, "generated_validation_pairwise_mi.csv")


# --------------------------------------------------------------------------- #
# Expected schema (re-stated independently from the spec)
# --------------------------------------------------------------------------- #
SEED = 0
EXPECTED_ROWS = {3000: 3000, 5000: 5000, 10000: 10000}

REQUIRED_COLUMNS = ["age", "education_num", "hours_per_week", "income_class"]
NUMERIC_COLS = ["age", "education_num", "hours_per_week"]
SENSITIVE_ATTR = "income_class"
ALL_COLS = REQUIRED_COLUMNS

NUMERIC_RANGES = {
    "age": (18, 80),
    "education_num": (6, 16),
    "hours_per_week": (1, 70),
}

ALLOWED_INCOME_VALUES = [
    "Low", "LowerMiddle", "UpperMiddle", "High", "VeryHigh", "Extreme",
]

EXTREME_MIN_PROP = 0.001
EXTREME_MAX_PROP = 0.005

# Mutual-information strength bands.
MI_WEAK_MAX = 0.05      # weak:   0 < MI < 0.05
MI_MEDIUM_MAX = 0.15    # medium: 0.05 <= MI < 0.15; strong: MI >= 0.15


# --------------------------------------------------------------------------- #
# Mutual information (independent implementation)
# --------------------------------------------------------------------------- #
def encode_for_mi(df: pd.DataFrame) -> dict:
    """Quantile-bin numeric columns into 10 bins (qcut, duplicates='drop');
    encode income_class as categorical codes."""
    encoded = {}
    for col in ALL_COLS:
        if col in NUMERIC_COLS:
            binned = pd.qcut(df[col], q=10, labels=False, duplicates="drop")
            encoded[col] = pd.Series(binned).astype("int64").to_numpy()
        else:
            encoded[col] = df[col].astype("category").cat.codes.to_numpy()
    return encoded


def compute_pairwise_mi(df: pd.DataFrame) -> pd.DataFrame:
    enc = encode_for_mi(df)
    rows = []
    for i in range(len(ALL_COLS)):
        for j in range(i + 1, len(ALL_COLS)):
            a, b = ALL_COLS[i], ALL_COLS[j]
            mi = mutual_info_score(enc[a], enc[b])
            if mi <= 0:
                strength = "none"
            elif mi < MI_WEAK_MAX:
                strength = "weak"
            elif mi < MI_MEDIUM_MAX:
                strength = "medium"
            else:
                strength = "strong"
            rows.append({
                "pair": f"{a}__{b}",
                "var_a": a,
                "var_b": b,
                "mutual_information_nats": mi,
                "strength": strength,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
class CheckCollector:
    """Accumulates (check_name, passed, detail) tuples for one dataset."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append((name, bool(passed), detail))
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        return all(p for _, p, _ in self.results)


def is_integer_valued(series: pd.Series) -> bool:
    """True if every (non-null) value equals its rounded value, regardless of
    whether the dtype is int or float."""
    if series.isna().any():
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        vals = series.to_numpy()
        return bool(np.all(np.equal(np.floor(vals), vals)))
    # object / other dtypes: try a strict cast.
    try:
        vals = series.astype(float).to_numpy()
    except (ValueError, TypeError):
        return False
    return bool(np.all(np.equal(np.floor(vals), vals)))


def validate_dataset(n: int, df: pd.DataFrame, mi_table: pd.DataFrame,
                     cc: CheckCollector) -> None:
    # 1. Correct row count.
    cc.add(
        "row_count",
        len(df) == EXPECTED_ROWS[n],
        f"rows={len(df)} expected={EXPECTED_ROWS[n]}",
    )

    # 2. Required columns exactly (same set, no extras, no missing).
    cols = list(df.columns)
    cc.add(
        "columns_exact",
        set(cols) == set(REQUIRED_COLUMNS) and len(cols) == len(REQUIRED_COLUMNS),
        f"columns={cols}",
    )

    # 3. No missing values.
    missing = int(df.isna().sum().sum())
    cc.add("no_missing_values", missing == 0, f"missing_total={missing}")

    # 4. Numeric columns integer-valued.
    non_int = [c for c in NUMERIC_COLS
               if c in df.columns and not is_integer_valued(df[c])]
    cc.add("numeric_integer_valued", not non_int,
           f"non_integer_cols={non_int}")

    # 5. Numeric columns in expected ranges.
    range_problems = []
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            range_problems.append(f"{col}:missing")
            continue
        cmin, cmax = df[col].min(), df[col].max()
        if cmin < lo or cmax > hi:
            range_problems.append(f"{col}:[{cmin},{cmax}]!⊆[{lo},{hi}]")
    cc.add("numeric_in_range", not range_problems,
           f"problems={range_problems}" if range_problems else "all in range")

    # 6. income_class only allowed values.
    if SENSITIVE_ATTR in df.columns:
        observed = set(df[SENSITIVE_ATTR].astype(str).unique())
        illegal = observed - set(ALLOWED_INCOME_VALUES)
        cc.add("income_allowed_values", not illegal,
               f"illegal={sorted(illegal)}" if illegal
               else f"values={sorted(observed)}")
    else:
        cc.add("income_allowed_values", False, "income_class column missing")

    # Distribution (used by checks 7-9).
    counts = df[SENSITIVE_ATTR].value_counts()
    props = counts / counts.sum()

    # 7. Non-uniform distribution.
    n_classes = df[SENSITIVE_ATTR].nunique()
    uniform_prop = 1.0 / n_classes if n_classes else 0.0
    # "Non-uniform" => proportions are not all (approximately) equal.
    max_dev = float((props - uniform_prop).abs().max()) if n_classes else 0.0
    cc.add("income_non_uniform", n_classes > 1 and max_dev > 0.02,
           f"n_classes={n_classes} max_dev_from_uniform={max_dev:.4f}")

    # 8. Extreme present.
    extreme_count = int(counts.get("Extreme", 0))
    cc.add("extreme_present", extreme_count > 0,
           f"extreme_count={extreme_count}")

    # 9. Extreme proportion within [0.001, 0.005].
    extreme_prop = float(props.get("Extreme", 0.0))
    cc.add(
        "extreme_proportion_in_band",
        EXTREME_MIN_PROP <= extreme_prop <= EXTREME_MAX_PROP,
        f"extreme_prop={extreme_prop:.4f} "
        f"band=[{EXTREME_MIN_PROP},{EXTREME_MAX_PROP}]",
    )

    # 10. Every pair has nonzero MI.
    min_mi = float(mi_table["mutual_information_nats"].min())
    cc.add("all_pairs_nonzero_mi", (mi_table["mutual_information_nats"] > 0).all(),
           f"min_pairwise_mi={min_mi:.4f}")

    # 11. Mixed MI strengths: >=1 weak, >=1 medium, >=1 strong.
    strengths = set(mi_table["strength"])
    have_weak = "weak" in strengths
    have_medium = "medium" in strengths
    have_strong = "strong" in strengths
    cc.add(
        "mi_mixed_strengths",
        have_weak and have_medium and have_strong,
        f"weak={have_weak} medium={have_medium} strong={have_strong}",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(PHASE1_DIR, exist_ok=True)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    passfail_rows: list[dict] = []
    mi_rows: list[pd.DataFrame] = []
    overall_pass = True

    emit("=" * 72)
    emit("PHASE 1 — INDEPENDENT VALIDATION REPORT")
    emit("=" * 72)
    emit()

    for n in EXPECTED_ROWS:
        name = f"simulated_adult_like_n{n}_seed{SEED}"
        path = os.path.join(RAW_DIR, f"{name}.csv")

        emit("-" * 72)
        emit(f"Dataset: {name}")
        emit(f"Path:    {os.path.relpath(path, PROJECT_ROOT)}")
        emit("-" * 72)

        cc = CheckCollector()

        if not os.path.exists(path):
            cc.add("file_exists", False, f"not found: {path}")
            emit(f"  [FAIL] file_exists — not found")
            for cname, passed, detail in cc.results:
                passfail_rows.append({
                    "dataset": name, "n": n, "check": cname,
                    "result": "PASS" if passed else "FAIL", "detail": detail,
                })
            overall_pass = False
            emit()
            continue

        df = pd.read_csv(path)

        # Compute MI once and reuse for checks + CSV output.
        mi_table = compute_pairwise_mi(df)
        mi_out = mi_table.copy()
        mi_out.insert(0, "n", n)
        mi_out.insert(0, "dataset", name)
        mi_rows.append(mi_out)

        validate_dataset(n, df, mi_table, cc)

        # Report per check.
        for cname, passed, detail in cc.results:
            tag = "PASS" if passed else "FAIL"
            emit(f"  [{tag}] {cname:<26} {detail}")
            passfail_rows.append({
                "dataset": name, "n": n, "check": cname,
                "result": tag, "detail": detail,
            })

        emit()
        emit("  Pairwise mutual information:")
        for _, r in mi_table.sort_values(
                "mutual_information_nats", ascending=False).iterrows():
            emit(f"    {r['var_a']:<14} <-> {r['var_b']:<14} "
                 f"{r['mutual_information_nats']:.4f}  ({r['strength']})")

        ds_pass = cc.all_passed
        overall_pass = overall_pass and ds_pass
        emit()
        emit(f"  => {name}: {'PASS' if ds_pass else 'FAIL'} "
             f"({sum(p for _, p, _ in cc.results)}/{len(cc.results)} checks)")
        emit()

    emit("=" * 72)
    emit(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    emit("=" * 72)

    # ---- Save outputs ---------------------------------------------------
    pd.DataFrame(passfail_rows).to_csv(OUT_PASSFAIL_CSV, index=False)
    if mi_rows:
        pd.concat(mi_rows, ignore_index=True).to_csv(OUT_MI_CSV, index=False)
    with open(OUT_REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    emit("")
    emit("Saved:")
    for path in (OUT_REPORT_TXT, OUT_PASSFAIL_CSV, OUT_MI_CSV):
        emit(f"  - {os.path.relpath(path, PROJECT_ROOT)}")

    # Non-zero exit code on failure so CI / shell can detect it.
    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
