#!/usr/bin/env python3
"""Phase 3 — shared utilities for the NON-PRIVATE simplified PrivBayes baseline.

This module is intentionally self-contained and side-effect free so that it can
be imported by the generation, validation, and evaluation scripts.

IMPORTANT — privacy disclaimer
------------------------------
Everything here is part of a **non-private** synthetic-data *utility upper
baseline*. No noise / differential privacy is applied. Mutual information,
structure learning, and the conditional probability tables are all computed
*exactly* from the data. Phase 4 will add differential privacy; until then these
outputs must NOT be described as private.

Conventions
-----------
* Schema (raw + synthetic):  age, education_num, hours_per_week, income_class
* QIs:                       age, education_num, hours_per_week
* Sensitive attribute:       income_class
* Mutual information unit:    **nats** (natural log). Ranking is base-invariant;
                              we document nats so MI numbers line up with the
                              Phase 1 sklearn ``mutual_info_score`` convention.

Discretization (for Bayesian-network learning)
----------------------------------------------
PrivBayes operates on discrete domains, so the numeric QIs are binned:

* age            -> 10-year bins      : 18-29, 30-39, 40-49, 50-59, 60-69, 70-80
* education_num  -> exact integers    : 6 .. 16
* hours_per_week -> 10-hour bins      : 1-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-70
* income_class   -> original category : Low, LowerMiddle, ... , Extreme

After sampling we decode each numeric bin back to its **rounded midpoint** so the
synthetic schema is identical to the raw schema:

* age   bins -> 24, 35, 45, 55, 65, 75
* hours bins -> 5, 15, 25, 35, 45, 55, 65
* education_num is already an exact integer and is passed through unchanged.

(Rounded midpoints, e.g. the 30-39 age bin -> midpoint 34.5 -> 35, and the 40-49
hours bin -> 44.5 -> 45, matching the project specification.)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
QIS = ["age", "education_num", "hours_per_week"]
SA = "income_class"
ATTRS = ["age", "education_num", "hours_per_week", "income_class"]

INCOME_DOMAIN = ["Low", "LowerMiddle", "UpperMiddle", "High", "VeryHigh",
                 "Extreme"]

DOMAINS = {
    "age": (18, 80),
    "education_num": (6, 16),
    "hours_per_week": (1, 70),
}

DATASET_SIZES = [3000, 5000, 10000]
R_VALUES = [0, 1, 2]
SEEDS = [0, 1, 2, 3, 4]
RAW_SEED = 0  # raw datasets only exist for seed 0


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


def synthetic_dir() -> Path:
    return project_root() / "data" / "synthetic" / "nonprivate"


def phase3_dir() -> Path:
    return project_root() / "outputs" / "phase3"


def raw_path(n: int) -> Path:
    return raw_dir() / f"simulated_adult_like_n{n}_seed{RAW_SEED}.csv"


def synthetic_path(n: int, r: int, seed: int) -> Path:
    return synthetic_dir() / f"privbayes_np_n{n}_r{r}_seed{seed}.csv"


def load_raw(n: int) -> pd.DataFrame:
    return pd.read_csv(raw_path(n))


# --------------------------------------------------------------------------- #
# Discretization for Bayesian-network learning
# --------------------------------------------------------------------------- #
# Each numeric QI maps to ordered, contiguous, fully-covering bins.  Bins are
# stored as (low, high, label) with INCLUSIVE integer bounds.
AGE_BINS = [(18, 29, "18-29"), (30, 39, "30-39"), (40, 49, "40-49"),
            (50, 59, "50-59"), (60, 69, "60-69"), (70, 80, "70-80")]
HOURS_BINS = [(1, 9, "1-9"), (10, 19, "10-19"), (20, 29, "20-29"),
              (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"),
              (60, 70, "60-70")]
# education_num uses its exact integer value (6..16) as the bin label.
EDU_VALUES = list(range(6, 17))

NUMERIC_BINS = {"age": AGE_BINS, "hours_per_week": HOURS_BINS}

def _round_half_up(x: float) -> int:
    """Round-half-up (not Python's banker's rounding) so documented midpoints
    match the spec exactly: 34.5 -> 35, 44.5 -> 45, 23.5 -> 24."""
    return int(np.floor(x + 0.5))


# Bin label -> rounded midpoint, used when decoding samples back to numbers.
# age   bins -> 24, 35, 45, 55, 65, 75 ; hours bins -> 5, 15, 25, 35, 45, 55, 65
AGE_MIDPOINT = {label: _round_half_up((lo + hi) / 2.0)
                for lo, hi, label in AGE_BINS}
HOURS_MIDPOINT = {label: _round_half_up((lo + hi) / 2.0)
                  for lo, hi, label in HOURS_BINS}
MIDPOINT = {"age": AGE_MIDPOINT, "hours_per_week": HOURS_MIDPOINT}


def _bin_numeric(values: pd.Series, bins: list[tuple[int, int, str]]) -> pd.Series:
    """Map integer values to their inclusive-bin label. Out-of-range values are
    clamped to the nearest edge bin so discretization never produces NaN."""
    lo_edges = [lo for lo, _, _ in bins]
    hi_edges = [hi for _, hi, _ in bins]
    labels = [lab for _, _, lab in bins]

    def to_label(v: float) -> str:
        v = int(round(float(v)))
        if v <= lo_edges[0]:
            return labels[0]
        if v >= hi_edges[-1]:
            return labels[-1]
        for lo, hi, lab in bins:
            if lo <= v <= hi:
                return lab
        return labels[-1]  # unreachable given contiguous bins

    return values.map(to_label)


def discretize(df: pd.DataFrame) -> pd.DataFrame:
    """Return a fully-discrete (all-string) copy of ``df`` in PrivBayes domain.

    age / hours_per_week -> bin labels, education_num -> str(int), income passed
    through. Column order is preserved.
    """
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in NUMERIC_BINS:
            out[col] = _bin_numeric(df[col], NUMERIC_BINS[col])
        elif col == "education_num":
            out[col] = df[col].astype(float).round().astype(int).clip(6, 16) \
                .astype(str)
        else:  # income_class (or any categorical passthrough)
            out[col] = df[col].astype(str)
    return out[list(df.columns)]


def decode(disc_df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`discretize` for sampled records: bin labels -> rounded
    midpoint integers, education_num -> int, income passed through.

    Returns a frame with the canonical schema/column order ``ATTRS``.
    """
    out = pd.DataFrame(index=disc_df.index)
    out["age"] = disc_df["age"].map(AGE_MIDPOINT).astype(int)
    out["education_num"] = disc_df["education_num"].astype(int)
    out["hours_per_week"] = disc_df["hours_per_week"].map(HOURS_MIDPOINT) \
        .astype(int)
    out["income_class"] = disc_df["income_class"].astype(str)
    return out[ATTRS]


def domain_of(disc_df: pd.DataFrame, attr: str) -> list[str]:
    """Sorted list of observed discrete values for ``attr`` (as strings)."""
    return sorted(disc_df[attr].astype(str).unique())


# --------------------------------------------------------------------------- #
# Mutual information (empirical, in nats)
# --------------------------------------------------------------------------- #
def mutual_information(disc_df: pd.DataFrame, x: str,
                      parents: tuple[str, ...]) -> float:
    """Empirical mutual information I(X; Pi) in **nats** on discretized data.

        I(X; Pi) = sum_{x,pi} p(x,pi) * log( p(x,pi) / (p(x) p(pi)) )

    Zero-probability terms are skipped. The parent set Pi is treated as a single
    composite categorical (the tuple of parent values). An empty parent set
    gives I(X; {}) = 0 by definition.
    """
    if not parents:
        return 0.0

    n = len(disc_df)
    x_ser = disc_df[x].astype(str)
    if len(parents) == 1:
        pi_ser = disc_df[parents[0]].astype(str)
    else:
        pi_ser = disc_df[list(parents)].astype(str).agg("\x1f".join, axis=1)

    px = x_ser.value_counts() / n
    ppi = pi_ser.value_counts() / n
    joint = pd.crosstab(x_ser, pi_ser) / n

    mi = 0.0
    for xv, row in joint.iterrows():
        pxv = px[xv]
        for piv, pxy in row.items():
            if pxy <= 0.0:
                continue
            mi += pxy * np.log(pxy / (pxv * ppi[piv]))
    return float(mi)


def pairwise_mutual_information(disc_df: pd.DataFrame,
                               cols: list[str] = ATTRS) -> dict:
    """All unordered-pair MIs (nats) over ``cols`` -> {(a, b): mi}."""
    return {(a, b): mutual_information(disc_df, a, (b,))
            for a, b in combinations(cols, 2)}


# --------------------------------------------------------------------------- #
# Marginal total variation distance (reused from Phase 2 evaluation)
# --------------------------------------------------------------------------- #
def joint_distribution(df: pd.DataFrame, attrs: list[str]) -> dict:
    """Empirical joint distribution over ``attrs`` as {cell: probability}.

    Every value is treated as an explicit string category.
    """
    sub = df[list(attrs)].astype(str)
    counts = sub.value_counts()
    probs = counts / counts.sum()
    return probs.to_dict()


def tvd_between(dist_a: dict, dist_b: dict) -> float:
    """TVD = 0.5 * sum over the union of cells |P_a(cell) - P_b(cell)|."""
    keys = set(dist_a) | set(dist_b)
    return float(0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0))
                           for k in keys))


def marginal_subsets() -> list[tuple[int, list[str]]]:
    """All 2-way and 3-way attribute subsets over ATTRS."""
    subsets = []
    for order in (2, 3):
        for combo in combinations(ATTRS, order):
            subsets.append((order, list(combo)))
    return subsets


# --------------------------------------------------------------------------- #
# Phase 2 evaluation bin space (shared raw/synthetic bins for marginal TVD)
# --------------------------------------------------------------------------- #
# Per the project decision, Phase 3 marginal TVD is computed in the SAME bin
# space Phase 2 used, so the two phases' TVD numbers are directly comparable.
# Both raw exact values and synthetic midpoint values are mapped into these bins.
EVAL_BIN_EDGES = {
    "age": [18, 25, 35, 45, 55, 65, 81],
    "education_num": [6, 10, 14, 17],
    "hours_per_week": [1, 21, 41, 61, 71],
}
EVAL_BIN_LABELS = {
    "age": ["18-24", "25-34", "35-44", "45-54", "55-64", "65-80"],
    "education_num": ["6-9", "10-13", "14-16"],
    "hours_per_week": ["1-20", "21-40", "41-60", "61-70"],
}


def eval_bin_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map a numeric-schema frame (raw or decoded synthetic) into the shared
    Phase 2 evaluation bins. income_class is passed through unchanged."""
    out = pd.DataFrame(index=df.index)
    for attr in QIS:
        out[attr] = pd.cut(df[attr].astype(float), bins=EVAL_BIN_EDGES[attr],
                           labels=EVAL_BIN_LABELS[attr], right=False,
                           include_lowest=True).astype(str)
    out[SA] = df[SA].astype(str).values
    return out
