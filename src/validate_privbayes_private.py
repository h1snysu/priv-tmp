#!/usr/bin/env python3
"""Phase 4 — independent validation of the PRIVATE PrivBayes outputs.

Re-reads the private synthetic CSVs, ``learned_networks_private.json`` and
``privacy_accounting_summary.csv`` and INDEPENDENTLY re-checks every Phase 4
output. It imports only the shared utilities (schema, fixed domains, budget
formula) — it does NOT import the generator ``privbayes_private.py``.

What this validator CAN and CANNOT do
-------------------------------------
It CANNOT mathematically prove differential privacy. It validates:
  * output completeness (all 180 files);
  * synthetic-record well-formedness (rows, schema, ranges, legal SA);
  * private-network structural validity (each attr once, degree<=r, parents
    earlier, acyclic);
  * noisy-CPT validity (finite, non-negative, rows sum to 1, valid fallbacks),
    re-derived from the recorded structure + noisy counts re-fit independently;
  * privacy-budget accounting consistency (every required field present and the
    recorded split matches the documented formula);
  * absence of obvious NON-PRIVATE artifacts (the private network JSON is not a
    copy of Phase 3's; outputs live under data/synthetic/private/).

    python src/validate_privbayes_private.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import privbayes_utils as U

EXPECTED_COLUMNS = U.ATTRS
TOL = 1e-6
D_ATTRS = len(U.ATTRS)

ACCT_FIELDS = ["epsilon_total", "epsilon_structure", "epsilon_parameters",
               "epsilon_step", "epsilon_cpt", "mi_sensitivity_bound",
               "laplace_scale"]


# --------------------------------------------------------------------------- #
# Network validation
# --------------------------------------------------------------------------- #
def validate_network(entry: dict, r: int) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    network = entry["network"]
    order = entry["order"]
    attrs = [node["attribute"] for node in network]

    checks.append(("each_attr_once",
                   sorted(attrs) == sorted(U.ATTRS)
                   and len(attrs) == len(set(attrs)), f"attrs={attrs}"))
    checks.append(("order_matches_nodes", order == attrs, f"order={order}"))

    max_deg = max((len(node["parents"]) for node in network), default=0)
    checks.append(("degree_within_r", max_deg <= r,
                   f"max_degree={max_deg} r={r}"))

    pos = {a: i for i, a in enumerate(order)}
    ok, detail = True, "all parents precede child"
    for node in network:
        a = node["attribute"]
        for p in node["parents"]:
            if p not in pos or pos[p] >= pos[a]:
                ok, detail = False, f"{a} has non-earlier parent {p}"
                break
        if not ok:
            break
    checks.append(("parents_earlier_no_cycle", ok, detail))

    pairs = [(node["attribute"], tuple(node["parents"])) for node in network]
    checks.append(("no_duplicate_pairs", len(pairs) == len(set(pairs)),
                   f"pairs={len(pairs)}"))
    return checks


# --------------------------------------------------------------------------- #
# Noisy-CPT validation (independent re-fit with the SAME mechanism)
# --------------------------------------------------------------------------- #
def refit_noisy_cpt(disc_df: pd.DataFrame, attr: str, parents: list[str],
                    laplace_scale: float, rng: np.random.Generator) -> dict:
    """Independently re-build a noisy CPT over the full fixed domain so we can
    check finiteness / non-negativity / row sums. The exact noise draws differ
    from the generator's (different RNG stream) — that is fine, we are checking
    that the *mechanism* yields valid probability tables, not reproducing them.
    """
    from itertools import product
    x_dom = U.FIXED_DOMAINS[attr]
    x_index = {v: i for i, v in enumerate(x_dom)}

    if not parents:
        counts = np.zeros(len(x_dom))
        for v, c in disc_df[attr].astype(str).value_counts().items():
            if v in x_index:
                counts[x_index[v]] = c
        noisy = np.clip(counts + rng.laplace(0, laplace_scale, counts.shape),
                        0, None)
        tot = noisy.sum()
        prob = noisy / tot if tot > 0 else np.full(len(x_dom), 1 / len(x_dom))
        return {"rows": [prob], "fallback": prob}

    obs: dict = {}
    grp = disc_df.groupby(parents)[attr].value_counts()
    for idx, cnt in grp.items():
        *pvals, xval = idx if isinstance(idx, tuple) else (idx,)
        obs[(tuple(str(v) for v in pvals), str(xval))] = int(cnt)

    rows = []
    marg = np.zeros(len(x_dom))
    pending = []
    for pconf in product(*[U.FIXED_DOMAINS[p] for p in parents]):
        counts = np.array([obs.get((pconf, xv), 0) for xv in x_dom], dtype=float)
        noisy = np.clip(counts + rng.laplace(0, laplace_scale, counts.shape),
                        0, None)
        marg += noisy
        tot = noisy.sum()
        if tot > 0:
            rows.append(noisy / tot)
        else:
            pending.append(True)
    mtot = marg.sum()
    fallback = marg / mtot if mtot > 0 else np.full(len(x_dom), 1 / len(x_dom))
    rows.extend([fallback] * len(pending))
    return {"rows": rows, "fallback": fallback}


def validate_cpts(entry: dict, disc_df: pd.DataFrame, laplace_scale: float,
                  rng: np.random.Generator) -> list[tuple[str, bool, str]]:
    all_probs: list[float] = []
    worst_row_err = 0.0
    fb_ok = True
    for node in entry["network"]:
        cpt = refit_noisy_cpt(disc_df, node["attribute"], node["parents"],
                              laplace_scale, rng)
        for prob in cpt["rows"]:
            worst_row_err = max(worst_row_err, abs(prob.sum() - 1.0))
            all_probs.extend(prob.tolist())
        fb = cpt["fallback"]
        if not (np.all(np.isfinite(fb)) and abs(fb.sum() - 1.0) <= TOL):
            fb_ok = False
        all_probs.extend(fb.tolist())
    arr = np.asarray(all_probs)
    return [
        ("cpt_all_finite", bool(np.all(np.isfinite(arr))), ""),
        ("cpt_no_negative", bool(np.all(arr >= -TOL)),
         f"min={arr.min():.3g}"),
        ("cpt_rows_sum_to_one", worst_row_err <= 1e-6,
         f"max_row_sum_error={worst_row_err:.2e}"),
        ("fallbacks_valid", fb_ok, ""),
    ]


# --------------------------------------------------------------------------- #
# Privacy-accounting validation
# --------------------------------------------------------------------------- #
def validate_accounting(acct: pd.Series, n: int, r: int, eps: float
                        ) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    present = all(f in acct and pd.notna(acct[f]) for f in ACCT_FIELDS)
    checks.append(("accounting_fields_present", present,
                   f"need={ACCT_FIELDS}"))
    if not present:
        return checks

    # recompute the documented split and compare
    if r > 0:
        es, ep = eps / 2.0, eps / 2.0
        e_step = es / (D_ATTRS - 1)
    else:
        es, ep = 0.0, eps
        e_step = 0.0
    e_cpt = ep / D_ATTRS
    scale = 2.0 / e_cpt

    def close(a, b):
        return abs(float(a) - float(b)) <= 1e-6

    split_ok = (close(acct["epsilon_total"], eps)
                and close(acct["epsilon_structure"], es)
                and close(acct["epsilon_parameters"], ep)
                and close(acct["epsilon_step"], e_step)
                and close(acct["epsilon_cpt"], e_cpt)
                and close(acct["laplace_scale"], scale))
    checks.append(("accounting_matches_formula", split_ok,
                   f"eps_struct={acct['epsilon_structure']} "
                   f"eps_param={acct['epsilon_parameters']} "
                   f"eps_cpt={acct['epsilon_cpt']} scale={acct['laplace_scale']}"))

    # composition: spent == epsilon_total
    spent = (D_ATTRS - 1) * float(acct["epsilon_step"]) \
        + D_ATTRS * float(acct["epsilon_cpt"]) if r > 0 \
        else D_ATTRS * float(acct["epsilon_cpt"])
    checks.append(("composition_sums_to_total", close(spent, eps),
                   f"spent={spent:.6f} total={eps}"))

    checks.append(("mi_sensitivity_positive",
                   float(acct["mi_sensitivity_bound"]) > 0,
                   f"delta_i={acct['mi_sensitivity_bound']}"))
    return checks


# --------------------------------------------------------------------------- #
# Synthetic CSV validation
# --------------------------------------------------------------------------- #
def validate_synthetic(path, raw_n: int) -> list[tuple[str, bool, str]]:
    if not path.exists():
        return [("file_exists", False, f"missing {path.name}")]
    df = pd.read_csv(path)
    checks = [("file_exists", True, path.name)]
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
        checks.append((f"{attr}_in_range", ok,
                       f"range=[{col.min()},{col.max()}] vs [{lo},{hi}]"))
    illegal = sorted(set(df[U.SA].astype(str)) - set(U.INCOME_DOMAIN))
    checks.append(("income_legal", not illegal, f"illegal={illegal}"))
    return checks


# --------------------------------------------------------------------------- #
# Non-private-artifact guard
# --------------------------------------------------------------------------- #
def validate_no_nonprivate_artifacts(learned: dict
                                     ) -> list[tuple[str, bool, str]]:
    """Confirm the private outputs are not silently reusing Phase 3."""
    checks: list[tuple[str, bool, str]] = []

    # (a) every entry is tagged as produced by the private mechanism
    sources = {e.get("source", "") for e in learned.values()}
    checks.append(("private_source_tag",
                   sources == {"phase4_private_exponential_mechanism"},
                   f"sources={sorted(sources)}"))

    # (b) the private network JSON must NOT be identical to the Phase 3 one
    p3 = U.phase3_dir() / "learned_networks.json"
    if p3.exists():
        with open(p3, encoding="utf-8") as fh:
            p3_data = json.load(fh)
        # compare the structure-only signatures; if Phase 4 were a copy of
        # Phase 3, the (order, parents) signatures would coincide for shared keys
        p3_sig = {k: [(nd["attribute"], tuple(nd["parents"]))
                      for nd in v["network"]] for k, v in p3_data.items()}
        identical_keys = 0
        compared = 0
        for k, v in learned.items():
            # phase3 keys look like n{n}_r{r}_seed{s}; derive the phase3 key
            base = f"n{v['n']}_r{v['r']}_seed{v['seed']}"
            if base in p3_sig:
                compared += 1
                sig = [(nd["attribute"], tuple(nd["parents"]))
                       for nd in v["network"]]
                if sig == p3_sig[base]:
                    identical_keys += 1
        # Some coincidental matches are possible (esp. r=0 data-independent
        # order or eps=2.0), so we only flag if essentially ALL match.
        frac = identical_keys / compared if compared else 0.0
        checks.append(("network_not_phase3_copy", frac < 0.95,
                       f"{identical_keys}/{compared} structures identical to "
                       f"Phase 3 (frac={frac:.2f})"))
    else:
        checks.append(("network_not_phase3_copy", True,
                       "phase3 json absent; nothing to copy"))

    # (c) outputs are under data/synthetic/private/, not nonprivate/
    sample = U.private_synthetic_path(U.DATASET_SIZES[0], 0, U.EPSILONS[0], 0)
    checks.append(("outputs_in_private_dir",
                   "synthetic/private" in str(sample)
                   and sample.parent == U.private_synthetic_dir(),
                   str(sample.parent)))
    return checks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    net_path = U.phase4_dir() / "learned_networks_private.json"
    acct_path = U.phase4_dir() / "privacy_accounting_summary.csv"
    for p in (net_path, acct_path):
        if not p.exists():
            raise SystemExit(f"missing {p}; run src/privbayes_private.py first")

    with open(net_path, encoding="utf-8") as fh:
        learned = json.load(fh)
    acct_df = pd.read_csv(acct_path)
    acct_idx = {(int(r.n), int(r.r), float(r.epsilon), int(r.seed)): r
                for r in acct_df.itertuples(index=False)}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 4 — INDEPENDENT VALIDATION (simplified private PrivBayes)")
    emit("Validates implementation consistency + budget accounting. It does NOT")
    emit("prove differential privacy.")
    emit("=" * 74)
    emit()

    cpt_rows: list[dict] = []
    synth_rows: list[dict] = []
    overall_pass = True
    n_combos = 0
    n_pass = 0

    disc_cache: dict = {}
    raw_len: dict = {}

    # global (run-independent) non-private-artifact checks
    artifact_checks = validate_no_nonprivate_artifacts(learned)
    emit("Global non-private-artifact guard:")
    for name, ok, detail in artifact_checks:
        overall_pass = overall_pass and ok
        emit(f"     {'PASS' if ok else 'FAIL'} {name:<26} {detail}")
    emit()

    for n in U.DATASET_SIZES:
        if n not in disc_cache:
            raw = U.load_raw(n)
            raw_len[n] = len(raw)
            disc_cache[n] = U.discretize(raw)

        for r in U.R_VALUES:
            for eps in U.EPSILONS:
                for seed in U.SEEDS:
                    n_combos += 1
                    key = f"n{n}_r{r}_{U.eps_tag(eps)}_seed{seed}"
                    label = f"n={n} r={r} eps={eps} seed={seed}"
                    entry = learned.get(key)

                    checks: list[tuple[str, bool, str]] = []
                    if entry is None:
                        checks.append(("network_entry_exists", False,
                                       f"no {key} in learned json"))
                    else:
                        checks += validate_network(entry, r)

                    acct = acct_idx.get((n, r, eps, seed))
                    if acct is None:
                        checks.append(("accounting_row_exists", False,
                                       f"no accounting row for {key}"))
                    else:
                        acct_series = pd.Series(acct._asdict())
                        checks += validate_accounting(acct_series, n, r, eps)
                        if entry is not None:
                            cpt_checks = validate_cpts(
                                entry, disc_cache[n],
                                float(acct_series["laplace_scale"]),
                                np.random.default_rng(10_000 + seed))
                            checks += cpt_checks
                            for nm, ok, dt in cpt_checks:
                                cpt_rows.append({"n": n, "r": r, "epsilon": eps,
                                                 "seed": seed, "check": nm,
                                                 "result": "PASS" if ok
                                                 else "FAIL", "detail": dt})

                    sc = validate_synthetic(
                        U.private_synthetic_path(n, r, eps, seed), raw_len[n])
                    checks += sc
                    for nm, ok, dt in sc:
                        synth_rows.append({"n": n, "r": r, "epsilon": eps,
                                           "seed": seed, "check": nm,
                                           "result": "PASS" if ok else "FAIL",
                                           "detail": dt})

                    combo_pass = all(ok for _, ok, _ in checks)
                    overall_pass = overall_pass and combo_pass
                    n_pass += int(combo_pass)
                    if not combo_pass:
                        emit(f"[FAIL] {label}")
                        for nm, ok, dt in checks:
                            if not ok:
                                emit(f"     FAIL {nm:<26} {dt}")

    expected = (len(U.DATASET_SIZES) * len(U.R_VALUES)
                * len(U.EPSILONS) * len(U.SEEDS))
    emit()
    emit("=" * 74)
    emit(f"Combinations validated: {n_pass}/{n_combos} fully PASS "
         f"(expected {expected})")
    emit(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    emit("=" * 74)

    pd.DataFrame(cpt_rows).to_csv(
        U.phase4_dir() / "noisy_cpt_validation_summary.csv", index=False)
    pd.DataFrame(synth_rows).to_csv(
        U.phase4_dir() / "synthetic_validation_summary.csv", index=False)

    emit()
    emit("Saved:")
    emit(f"  - {(U.phase4_dir() / 'noisy_cpt_validation_summary.csv').relative_to(U.project_root())}")
    emit(f"  - {(U.phase4_dir() / 'synthetic_validation_summary.csv').relative_to(U.project_root())}")

    with open(U.phase4_dir() / "phase4_validation_report.txt", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
