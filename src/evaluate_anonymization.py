#!/usr/bin/env python3
"""Phase 2 — utility evaluation of anonymized outputs.

For every feasible anonymized output we compute:

  * information loss   — a level-based loss and a data-dependent numeric NCP
                         (parsing generalized intervals via their midpoints /
                         spans);
  * marginal fidelity  — total variation distance between the raw and the
                         anonymized empirical joint distributions for all
                         2-way and 3-way marginals over the four attributes.

Generalized intervals and "*" are treated as explicit categories when forming
marginals (i.e. an interval is its own category, distinct from the exact raw
values), exactly as specified.

Run from the project root with:

    python src/evaluate_anonymization.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
QIS = ["age", "education_num", "hours_per_week"]
SA = "income_class"
ATTRS = ["age", "education_num", "hours_per_week", "income_class"]
SEED = 0
MAX_LEVEL = 4

DOMAINS = {
    "age": (18, 80),
    "education_num": (6, 16),
    "hours_per_week": (1, 70),
}
EDU_CATS = {"LowEdu": (6, 9), "MidEdu": (10, 13), "HighEdu": (14, 16)}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def phase2_dir() -> Path:
    return project_root() / "outputs" / "phase2"


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


# --------------------------------------------------------------------------- #
# Interval parsing for information loss
# --------------------------------------------------------------------------- #
def parse_bounds(token: str, attr: str) -> tuple[int, int]:
    """Parse a generalized QI cell into (low, high) integer bounds."""
    dmin, dmax = DOMAINS[attr]
    token = str(token)
    if token == "*":
        return dmin, dmax
    if token in EDU_CATS:
        return EDU_CATS[token]
    if token.startswith("[") and token.endswith("]"):
        lo_s, hi_s = token[1:-1].split("-", 1)
        return int(lo_s), int(hi_s)
    return int(token), int(token)


def cell_midpoint(token: str, attr: str) -> float:
    lo, hi = parse_bounds(token, attr)
    return (lo + hi) / 2.0


def cell_ncp(token: str, attr: str) -> float:
    dmin, dmax = DOMAINS[attr]
    lo, hi = parse_bounds(token, attr)
    span = dmax - dmin
    return (hi - lo) / span if span > 0 else 0.0


def level_information_loss(levels: dict) -> float:
    """Mean of (level / MAX_LEVEL) across the QIs."""
    return float(np.mean([levels[a] / MAX_LEVEL for a in QIS]))


def numeric_ncp_information_loss(anon: pd.DataFrame) -> float:
    """Mean per-cell NCP over all QI cells of the anonymized frame."""
    parts = []
    for attr in QIS:
        parts.append(anon[attr].map(lambda v, a=attr: cell_ncp(v, a)).to_numpy())
    return float(np.mean(np.concatenate(parts))) if parts else 0.0


# --------------------------------------------------------------------------- #
# Marginal total variation distance
# --------------------------------------------------------------------------- #
def joint_distribution(df: pd.DataFrame, attrs: list[str]) -> dict:
    """Empirical joint distribution over `attrs` as a {cell: probability} dict.

    Every value (including generalized intervals and "*") is treated as an
    explicit string category.
    """
    sub = df[list(attrs)].astype(str)
    counts = sub.value_counts()          # MultiIndex (or Index) -> counts
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
# Midpoint binning (interpretable marginal TVD)
# --------------------------------------------------------------------------- #
# Common fixed bins applied to BOTH raw and anonymized numeric midpoints so the
# two distributions live in the same category space.
BIN_EDGES = {
    "age": [18, 25, 35, 45, 55, 65, 81],
    "education_num": [6, 10, 14, 17],
    "hours_per_week": [1, 21, 41, 61, 71],
}
BIN_LABELS = {
    "age": ["18-24", "25-34", "35-44", "45-54", "55-64", "65-80"],
    "education_num": ["6-9", "10-13", "14-16"],
    "hours_per_week": ["1-20", "21-40", "41-60", "61-70"],
}


def bin_midpoints(midpoints: pd.Series, attr: str) -> pd.Series:
    """Bin numeric midpoints into the fixed (right-open) bins for `attr`."""
    return pd.cut(midpoints, bins=BIN_EDGES[attr], labels=BIN_LABELS[attr],
                  right=False, include_lowest=True).astype(str)


def midpoint_bin_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each QI column with its midpoint-derived bin label.

    Works for both raw frames (exact integers map to themselves) and
    anonymized frames (intervals / "*" / edu categories map to midpoints).
    income_class is passed through unchanged.
    """
    out = pd.DataFrame(index=df.index)
    for attr in QIS:
        mids = df[attr].map(lambda v, a=attr: cell_midpoint(v, a))
        out[attr] = bin_midpoints(mids, attr)
    out[SA] = df[SA].astype(str).values
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_raw(n: int, cache: dict) -> pd.DataFrame:
    if n not in cache:
        path = raw_dir() / f"simulated_adult_like_n{n}_seed{SEED}.csv"
        cache[n] = pd.read_csv(path)
    return cache[n]


def raw_marginals(n: int, raw: pd.DataFrame, cache: dict) -> dict:
    """Cache raw joint distributions per n — reused across every output.

    Returns {"strict": {attrs: dist}, "binned": {attrs: dist}} where the strict
    marginals use raw exact values as categories and the binned marginals use
    the shared midpoint bins.
    """
    if n not in cache:
        raw_binned = midpoint_bin_frame(raw)
        cache[n] = {
            "strict": {tuple(attrs): joint_distribution(raw, attrs)
                       for _, attrs in marginal_subsets()},
            "binned": {tuple(attrs): joint_distribution(raw_binned, attrs)
                       for _, attrs in marginal_subsets()},
        }
    return cache[n]


def main() -> None:
    phase2_dir().mkdir(parents=True, exist_ok=True)
    runtime_path = phase2_dir() / "runtime_summary.csv"
    if not runtime_path.exists():
        raise SystemExit(f"missing {runtime_path}; run src/anonymize.py first")

    runtime = pd.read_csv(runtime_path)
    feasible = runtime[runtime["feasible"] == True]  # noqa: E712

    raw_cache: dict = {}
    raw_marg_cache: dict = {}

    info_rows: list[dict] = []
    marg_rows: list[dict] = []

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 2 — UTILITY EVALUATION (information loss + marginal TVD)")
    emit("=" * 74)
    emit()

    for _, row in feasible.iterrows():
        n = int(row["n"])
        param = f"{row['parameter_name']}={row['parameter_value']}"
        out_path = project_root() / row["output_path"]
        if not out_path.exists():
            emit(f"[SKIP] missing output {row['output_path']}")
            continue

        anon = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        raw = load_raw(n, raw_cache)
        raw_m = raw_marginals(n, raw, raw_marg_cache)
        anon_binned = midpoint_bin_frame(anon)

        levels = {"age": int(row["age_level"]),
                  "education_num": int(row["education_num_level"]),
                  "hours_per_week": int(row["hours_per_week_level"])}

        level_loss = level_information_loss(levels)
        numeric_ncp = numeric_ncp_information_loss(anon)
        rows_suppressed = int(float(row["rows_suppressed"])) if str(
            row["rows_suppressed"]).strip() != "" else 0

        info_rows.append({
            "dataset": row["dataset"], "n": n, "method": row["method"],
            "parameter": param,
            "age_level": levels["age"],
            "education_num_level": levels["education_num"],
            "hours_per_week_level": levels["hours_per_week"],
            "level_loss": round(level_loss, 6),
            "numeric_ncp": round(numeric_ncp, 6),
            "rows_suppressed": rows_suppressed,
        })

        # ---- Marginal TVDs (two metric types) --------------------------
        # strict_symbolic_tvd : generalized intervals / "*" are their own
        #                       categories, disjoint from raw exact values.
        # midpoint_binned_tvd : raw and anon mapped to shared midpoint bins,
        #                       giving an interpretable distributional distance.
        per_metric = {
            "strict_symbolic_tvd": {"two": [], "three": [],
                                    "raw": raw_m["strict"], "anon": anon},
            "midpoint_binned_tvd": {"two": [], "three": [],
                                    "raw": raw_m["binned"], "anon": anon_binned},
        }

        for order, attrs in marginal_subsets():
            for metric, ctx in per_metric.items():
                anon_dist = joint_distribution(ctx["anon"], attrs)
                tvd = tvd_between(ctx["raw"][tuple(attrs)], anon_dist)
                marg_rows.append({
                    "dataset": row["dataset"], "n": n, "method": row["method"],
                    "parameter": param, "marginal_order": order,
                    "attributes": "|".join(attrs),
                    "metric_type": metric, "tvd": round(tvd, 6),
                })
                (ctx["two"] if order == 2 else ctx["three"]).append(tvd)

        emit_parts = [f"{row['dataset']} | {row['method']:<12s} {param:<8s}  "
                      f"level_loss={level_loss:.3f} numeric_ncp={numeric_ncp:.3f}"]
        for metric, ctx in per_metric.items():
            avg2 = float(np.mean(ctx["two"])) if ctx["two"] else float("nan")
            avg3 = float(np.mean(ctx["three"])) if ctx["three"] else float("nan")
            for label, val in (("avg_2way", avg2), ("avg_3way", avg3)):
                marg_rows.append({
                    "dataset": row["dataset"], "n": n, "method": row["method"],
                    "parameter": param, "marginal_order": label,
                    "attributes": "ALL", "metric_type": metric,
                    "tvd": round(val, 6),
                })
            tag = "strict" if metric.startswith("strict") else "binned"
            emit_parts.append(f"{tag}(2w={avg2:.3f},3w={avg3:.3f})")
        emit("  " + "  ".join(emit_parts))

    # ---- Save -----------------------------------------------------------
    info_df = pd.DataFrame(info_rows)
    marg_df = pd.DataFrame(marg_rows)
    info_df.to_csv(phase2_dir() / "information_loss_summary.csv", index=False)
    marg_df.to_csv(phase2_dir() / "marginal_tvd_summary.csv", index=False)

    emit()
    emit("-" * 74)
    emit("Aggregate utility by method (mean over all feasible settings):")
    emit("-" * 74)
    agg = (info_df.groupby("method")[["level_loss", "numeric_ncp"]]
           .mean().round(4))
    emit(agg.to_string())
    emit()
    avg_marg = marg_df[marg_df["marginal_order"].isin(["avg_2way", "avg_3way"])]
    for metric in ("strict_symbolic_tvd", "midpoint_binned_tvd"):
        sub = avg_marg[avg_marg["metric_type"] == metric]
        pivot = (sub.pivot_table(index="method", columns="marginal_order",
                                 values="tvd", aggfunc="mean").round(4))
        emit(f"Mean {metric} by method:")
        emit(pivot.to_string())
        emit()

    with open(phase2_dir() / "phase2_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    emit()
    emit("Saved:")
    for name in ("information_loss_summary.csv", "marginal_tvd_summary.csv",
                 "phase2_summary.txt"):
        emit(f"  - {(phase2_dir() / name).relative_to(project_root())}")


if __name__ == "__main__":
    main()
