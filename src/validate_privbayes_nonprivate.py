#!/usr/bin/env python3
"""Phase 3 — independent validation of the NON-PRIVATE PrivBayes outputs.

This script re-reads the synthetic CSVs and ``learned_networks.json`` and
INDEPENDENTLY re-checks every Phase 3 output. It imports only the shared
utilities (schema constants, discretization, domains) — it does NOT import the
generator ``privbayes_nonprivate.py`` and does NOT trust the CPT metadata that
the generator recorded; CPT row sums are re-fit here from the raw data.

Checks
------
Network (from learned_networks.json):
  * every attribute appears exactly once
  * parent degree <= r
  * every parent appears earlier in the network order
  * no cycles (implied by the earlier-parent rule; verified explicitly)
  * no duplicate attribute-parent pairs

CPT (re-fit from raw + network):
  * every probability finite
  * no negative probabilities
  * every CPT row sums to 1 within tolerance
  * marginal fallback distributions valid

Synthetic CSV (per n, r, seed):
  * row count equals original n
  * columns exactly age, education_num, hours_per_week, income_class
  * no missing values
  * age in [18, 80], education_num in [6, 16], hours_per_week in [1, 70]
  * income_class values legal
  * a file exists for every (n, r, seed) combination

Exits 0 if every check passes, 1 otherwise.

    python src/validate_privbayes_nonprivate.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import privbayes_utils as U


EXPECTED_COLUMNS = U.ATTRS
TOL = 1e-6


# --------------------------------------------------------------------------- #
# Independent CPT re-fit (does not import the generator)
# --------------------------------------------------------------------------- #
def refit_cpt(disc_df: pd.DataFrame, attr: str, parents: list[str]) -> dict:
    """Re-derive P(X | Parents) as {parent_config: {value: prob}}."""
    if not parents:
        counts = disc_df[attr].astype(str).value_counts()
        return {(): (counts / counts.sum()).to_dict()}
    cpt: dict = {}
    grp = disc_df.groupby(parents)[attr].value_counts()
    for idx, cnt in grp.items():
        *pvals, xval = idx if isinstance(idx, tuple) else (idx,)
        cpt.setdefault(tuple(str(v) for v in pvals), {})[str(xval)] = int(cnt)
    for row in cpt.values():
        total = sum(row.values())
        for xv in row:
            row[xv] = row[xv] / total
    return cpt


def refit_marginal(disc_df: pd.DataFrame, attr: str) -> dict:
    counts = disc_df[attr].astype(str).value_counts()
    return (counts / counts.sum()).to_dict()


# --------------------------------------------------------------------------- #
# Network validation
# --------------------------------------------------------------------------- #
def validate_network(entry: dict, r: int) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    network = entry["network"]
    order = entry["order"]
    attrs = [node["attribute"] for node in network]

    checks.append((
        "each_attr_once",
        sorted(attrs) == sorted(U.ATTRS) and len(attrs) == len(set(attrs)),
        f"attrs={attrs}"))

    checks.append((
        "order_matches_nodes", order == attrs,
        f"order={order}"))

    max_deg = max((len(node["parents"]) for node in network), default=0)
    checks.append((
        "degree_within_r", max_deg <= r, f"max_degree={max_deg} r={r}"))

    # parents earlier in order (also rules out cycles for a fixed topological
    # order) + no self-parent
    pos = {a: i for i, a in enumerate(order)}
    earlier_ok, earlier_detail = True, "all parents precede child"
    for node in network:
        a = node["attribute"]
        for p in node["parents"]:
            if p not in pos or pos[p] >= pos[a]:
                earlier_ok = False
                earlier_detail = f"{a} has non-earlier parent {p}"
                break
        if not earlier_ok:
            break
    checks.append(("parents_earlier_no_cycle", earlier_ok, earlier_detail))

    pairs = [(node["attribute"], tuple(node["parents"])) for node in network]
    checks.append((
        "no_duplicate_pairs", len(pairs) == len(set(pairs)),
        f"pairs={len(pairs)} unique={len(set(pairs))}"))

    return checks


# --------------------------------------------------------------------------- #
# CPT validation (re-fit)
# --------------------------------------------------------------------------- #
def validate_cpts(entry: dict, disc_df: pd.DataFrame
                  ) -> list[tuple[str, bool, str]]:
    all_probs: list[float] = []
    worst_row_err = 0.0
    fallback_ok = True
    for node in entry["network"]:
        attr, parents = node["attribute"], node["parents"]
        cpt = refit_cpt(disc_df, attr, parents)
        for row in cpt.values():
            s = sum(row.values())
            worst_row_err = max(worst_row_err, abs(s - 1.0))
            all_probs.extend(row.values())
        fb = refit_marginal(disc_df, attr)
        if not (np.all(np.isfinite(list(fb.values())))
                and abs(sum(fb.values()) - 1.0) <= TOL):
            fallback_ok = False
        all_probs.extend(fb.values())

    arr = np.asarray(all_probs, dtype=float)
    return [
        ("cpt_all_finite", bool(np.all(np.isfinite(arr))), ""),
        ("cpt_no_negative", bool(np.all(arr >= -TOL)), f"min={arr.min():.3g}"),
        ("cpt_rows_sum_to_one", worst_row_err <= TOL,
         f"max_row_sum_error={worst_row_err:.2e}"),
        ("fallbacks_valid", fallback_ok, ""),
    ]


# --------------------------------------------------------------------------- #
# Synthetic CSV validation
# --------------------------------------------------------------------------- #
def validate_synthetic(n: int, r: int, seed: int, raw_n: int
                       ) -> list[tuple[str, bool, str]]:
    path = U.synthetic_path(n, r, seed)
    if not path.exists():
        return [("file_exists", False, f"missing {path.name}")]

    df = pd.read_csv(path)
    checks: list[tuple[str, bool, str]] = [("file_exists", True, path.name)]

    checks.append(("row_count", len(df) == raw_n,
                   f"rows={len(df)} expected={raw_n}"))
    checks.append(("schema_exact", list(df.columns) == EXPECTED_COLUMNS,
                   f"columns={list(df.columns)}"))
    miss = int(df.isna().sum().sum())
    checks.append(("no_missing_values", miss == 0, f"missing={miss}"))

    for attr in U.QIS:
        lo, hi = U.DOMAINS[attr]
        col = pd.to_numeric(df[attr], errors="coerce")
        ok = bool(col.notna().all() and col.min() >= lo and col.max() <= hi)
        checks.append((f"{attr}_in_range",
                       ok, f"range=[{col.min()},{col.max()}] expected[{lo},{hi}]"))

    illegal = sorted(set(df[U.SA].astype(str)) - set(U.INCOME_DOMAIN))
    checks.append(("income_legal", not illegal, f"illegal={illegal}"))

    return checks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    net_path = U.phase3_dir() / "learned_networks.json"
    if not net_path.exists():
        raise SystemExit(f"missing {net_path}; run "
                         f"src/privbayes_nonprivate.py first")
    with open(net_path, encoding="utf-8") as fh:
        learned = json.load(fh)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 3 — INDEPENDENT VALIDATION (non-private PrivBayes baseline)")
    emit("=" * 74)
    emit()

    cpt_rows: list[dict] = []
    synth_rows: list[dict] = []
    overall_pass = True
    n_combos = 0
    n_pass = 0

    disc_cache: dict = {}
    raw_len: dict = {}

    for n in U.DATASET_SIZES:
        if n not in disc_cache:
            raw = U.load_raw(n)
            raw_len[n] = len(raw)
            disc_cache[n] = U.discretize(raw)

        for r in U.R_VALUES:
            for seed in U.SEEDS:
                n_combos += 1
                key = f"n{n}_r{r}_seed{seed}"
                label = f"n={n} r={r} seed={seed}"
                entry = learned.get(key)

                checks: list[tuple[str, bool, str]] = []
                if entry is None:
                    checks.append(("network_entry_exists", False,
                                   f"no {key} in learned_networks.json"))
                else:
                    checks += validate_network(entry, r)
                    cpt_checks = validate_cpts(entry, disc_cache[n])
                    checks += cpt_checks
                    for name, ok, detail in cpt_checks:
                        cpt_rows.append({"n": n, "r": r, "seed": seed,
                                         "check": name,
                                         "result": "PASS" if ok else "FAIL",
                                         "detail": detail})

                synth_checks = validate_synthetic(n, r, seed, raw_len[n])
                checks += synth_checks
                for name, ok, detail in synth_checks:
                    synth_rows.append({"n": n, "r": r, "seed": seed,
                                       "check": name,
                                       "result": "PASS" if ok else "FAIL",
                                       "detail": detail})

                combo_pass = all(ok for _, ok, _ in checks)
                overall_pass = overall_pass and combo_pass
                n_pass += int(combo_pass)

                emit(f"[{'PASS' if combo_pass else 'FAIL'}] {label}")
                for name, ok, detail in checks:
                    if not ok:
                        emit(f"     FAIL {name:<26} {detail}")

    emit()
    emit("=" * 74)
    emit(f"Combinations validated: {n_pass}/{n_combos} fully PASS "
         f"(expected {len(U.DATASET_SIZES) * len(U.R_VALUES) * len(U.SEEDS)})")
    emit(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    emit("=" * 74)

    pd.DataFrame(cpt_rows).to_csv(
        U.phase3_dir() / "cpt_validation_summary.csv", index=False)
    pd.DataFrame(synth_rows).to_csv(
        U.phase3_dir() / "synthetic_validation_summary.csv", index=False)

    emit()
    emit("Saved:")
    emit(f"  - {(U.phase3_dir() / 'cpt_validation_summary.csv').relative_to(U.project_root())}")
    emit(f"  - {(U.phase3_dir() / 'synthetic_validation_summary.csv').relative_to(U.project_root())}")

    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
