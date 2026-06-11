#!/usr/bin/env python3
"""Phase 2 — independent validation of anonymized outputs.

This script re-reads ``outputs/phase2/runtime_summary.csv`` and independently
re-checks every feasible anonymized CSV. It does NOT import anonymize.py and
does NOT trust the statistics that anonymize.py recorded; every privacy
property is recomputed here from the raw and anonymized files.

Infeasible settings are reported (not failed).

Run from the project root with:

    python src/validate_anonymization.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Constants (re-stated independently)
# --------------------------------------------------------------------------- #
QIS = ["age", "education_num", "hours_per_week"]
SA = "income_class"
INCOME_DOMAIN = ["Low", "LowerMiddle", "UpperMiddle", "High", "VeryHigh",
                 "Extreme"]
EXPECTED_COLUMNS = QIS + [SA]
SEED = 0

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
# Helpers (independent re-implementations)
# --------------------------------------------------------------------------- #
def global_sa_distribution(df: pd.DataFrame) -> pd.Series:
    counts = df[SA].astype(str).value_counts()
    probs = counts.reindex(INCOME_DOMAIN, fill_value=0).astype(float)
    total = probs.sum()
    return probs / total if total > 0 else probs


def class_tvds(df: pd.DataFrame, global_dist: pd.Series) -> pd.Series:
    """Per-equivalence-class TVD to the supplied global SA distribution."""
    ct = (df.groupby(QIS + [SA], sort=False).size()
            .unstack(SA, fill_value=0)
            .reindex(columns=INCOME_DOMAIN, fill_value=0))
    prob = ct.div(ct.sum(axis=1), axis=0)
    return 0.5 * prob.sub(global_dist, axis=1).abs().sum(axis=1)


def valid_qi_token(token: str, attr: str) -> bool:
    """True if a generalized QI cell is a well-formed value for the attribute:
    "*", an in-range exact integer, a valid "[lo-hi]" interval, or (for
    education_num) one of the broad categories."""
    token = str(token)
    dmin, dmax = DOMAINS[attr]
    if token == "*":
        return True
    if attr == "education_num" and token in EDU_CATS:
        return True
    if token.startswith("[") and token.endswith("]"):
        body = token[1:-1]
        if "-" not in body:
            return False
        lo_s, hi_s = body.split("-", 1)
        try:
            lo, hi = int(lo_s), int(hi_s)
        except ValueError:
            return False
        return dmin <= lo <= hi <= dmax
    try:
        v = int(token)
    except ValueError:
        return False
    return dmin <= v <= dmax


# --------------------------------------------------------------------------- #
# Per-output validation
# --------------------------------------------------------------------------- #
def validate_output(row: pd.Series, raw_cache: dict) -> list[dict]:
    """Run all required checks on one feasible anonymized output.

    Returns a list of {check, result, detail} dicts.
    """
    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "result": "PASS" if passed else "FAIL",
                       "detail": detail})

    out_path = project_root() / row["output_path"]
    if not out_path.exists():
        add("file_exists", False, f"missing: {row['output_path']}")
        return checks

    anon = pd.read_csv(out_path, dtype=str, keep_default_na=False)

    # Raw dataset (cached) for row-count + SA-domain comparison.
    n = int(row["n"])
    if n not in raw_cache:
        raw_path = raw_dir() / f"simulated_adult_like_n{n}_seed{SEED}.csv"
        raw_cache[n] = pd.read_csv(raw_path)
    raw = raw_cache[n]

    # 1. schema_exact
    add("schema_exact", list(anon.columns) == EXPECTED_COLUMNS,
        f"columns={list(anon.columns)}")

    # 2. no_missing_values
    missing = int((anon == "").sum().sum() + anon.isna().sum().sum())
    add("no_missing_values", missing == 0, f"missing={missing}")

    # 3. row_count_preserved_or_reported
    rows_out = len(anon)
    row_supp_used = str(row["row_suppression_used"]).strip().lower() == "true"
    rows_suppressed = int(float(row["rows_suppressed"])) if str(
        row["rows_suppressed"]).strip() != "" else 0
    if row_supp_used:
        ok = rows_out == len(raw) - rows_suppressed
        detail = (f"rows_out={rows_out} raw={len(raw)} "
                  f"suppressed={rows_suppressed} (row suppression reported)")
    else:
        ok = rows_out == len(raw)
        detail = f"rows_out={rows_out} raw={len(raw)} (no suppression)"
    add("row_count_preserved_or_reported", ok, detail)

    # 4. suppressed_qi_values_valid (well-formed generalized QI cells)
    bad = {}
    for attr in QIS:
        invalid = [v for v in anon[attr].unique() if not valid_qi_token(v, attr)]
        if invalid:
            bad[attr] = invalid[:5]
    add("suppressed_qi_values_valid", not bad,
        "all QI cells well-formed" if not bad else f"invalid={bad}")

    # 5. income_domain_unchanged
    anon_sa = set(anon[SA].unique())
    illegal = anon_sa - set(INCOME_DOMAIN)
    domain_ok = not illegal
    # When no row suppression, the SA multiset must be identical to raw.
    if not row_supp_used:
        same_counts = (anon[SA].value_counts().sort_index()
                       .equals(raw[SA].astype(str).value_counts().sort_index()))
        domain_ok = domain_ok and same_counts
        detail = (f"illegal={sorted(illegal)} sa_counts_match={same_counts}")
    else:
        detail = f"illegal={sorted(illegal)} (row suppression — counts not compared)"
    add("income_domain_unchanged", domain_ok, detail)

    # ---- Privacy properties (recomputed) -------------------------------
    sizes = anon.groupby(QIS, sort=False).size()
    k = int(float(row["k"]))

    # 6. k_anonymity
    min_size = int(sizes.min()) if len(sizes) else 0
    add("k_anonymity", len(sizes) > 0 and min_size >= k,
        f"min_class_size={min_size} k={k} classes={len(sizes)}")

    method = row["method"]

    # 7. l_diversity (where applicable)
    if method == "l_diversity":
        ell = int(float(row["ell"]))
        min_distinct = int(anon.groupby(QIS, sort=False)[SA].nunique().min())
        add("l_diversity", min_distinct >= ell,
            f"min_distinct_sa={min_distinct} ell={ell}")

    # 8. t_closeness (where applicable)
    if method == "t_closeness":
        t = float(row["t"])
        global_dist = global_sa_distribution(raw)  # global RAW distribution
        max_tvd = float(class_tvds(anon, global_dist).max())
        add("t_closeness", max_tvd <= t + 1e-9,
            f"max_class_tvd={max_tvd:.4f} t={t}")

    return checks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    phase2_dir().mkdir(parents=True, exist_ok=True)
    runtime_path = phase2_dir() / "runtime_summary.csv"
    if not runtime_path.exists():
        raise SystemExit(f"missing {runtime_path}; run src/anonymize.py first")

    runtime = pd.read_csv(runtime_path)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 2 — INDEPENDENT VALIDATION OF ANONYMIZED OUTPUTS")
    emit("=" * 74)
    emit()

    summary_rows: list[dict] = []
    raw_cache: dict = {}
    overall_pass = True
    n_feasible = 0
    n_outputs_pass = 0

    for _, row in runtime.iterrows():
        label = (f"{row['dataset']} | {row['method']} "
                 f"{row['parameter_name']}={row['parameter_value']}")

        if not bool(row["feasible"]):
            emit(f"[INFEASIBLE] {label} — {row['infeasible_reason']}")
            summary_rows.append({
                "dataset": row["dataset"], "n": row["n"],
                "method": row["method"],
                "parameter": f"{row['parameter_name']}={row['parameter_value']}",
                "feasible": False, "check": "feasibility",
                "result": "INFEASIBLE", "detail": row["infeasible_reason"],
            })
            continue

        n_feasible += 1
        checks = validate_output(row, raw_cache)
        out_pass = all(c["result"] == "PASS" for c in checks)
        overall_pass = overall_pass and out_pass
        n_outputs_pass += int(out_pass)

        emit(f"[{'PASS' if out_pass else 'FAIL'}] {label}")
        for c in checks:
            emit(f"     {c['result']:<4} {c['check']:<32} {c['detail']}")
            summary_rows.append({
                "dataset": row["dataset"], "n": row["n"],
                "method": row["method"],
                "parameter": f"{row['parameter_name']}={row['parameter_value']}",
                "feasible": True, "check": c["check"],
                "result": c["result"], "detail": c["detail"],
            })
        emit()

    emit("=" * 74)
    emit(f"Feasible outputs validated: {n_outputs_pass}/{n_feasible} fully PASS")
    emit(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    emit("=" * 74)

    pd.DataFrame(summary_rows).to_csv(
        phase2_dir() / "validation_summary.csv", index=False)
    with open(phase2_dir() / "validation_report.txt", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    emit("")
    emit("Saved:")
    emit(f"  - {(phase2_dir() / 'validation_summary.csv').relative_to(project_root())}")
    emit(f"  - {(phase2_dir() / 'validation_report.txt').relative_to(project_root())}")

    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
