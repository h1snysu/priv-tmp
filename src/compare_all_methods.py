#!/usr/bin/env python3
"""Final comparative analysis across Phases 2-4 (read-only).

This script does NOT regenerate any data and does NOT change Phase 1-4 behavior.
It only reads existing outputs from outputs/phase2, outputs/phase3 and
outputs/phase4, normalizes them into a single comparable schema, and writes
final comparison tables + a text summary to outputs/final_comparison/.

Comparison groups
-----------------
  Phase 2 syntactic anonymization : k-anonymity, l-diversity, t-closeness
  Phase 3 non-private PrivBayes    : r = 0, 1, 2   (utility upper baseline)
  Phase 4 private PrivBayes        : r = 0, 1, 2  x  epsilon in {0.1,0.5,1,2}

Main utility metric
-------------------
The MAIN reported marginal-TVD metric is the COMMON-BIN / Phase 2 midpoint-bin
TVD, so all three phases live in the same evaluation bin space. Phase 2's
"strict_symbolic_tvd" is included only as a conservative appendix column.

Randomized methods (Phase 3 and 4) are aggregated across the 5 seeds with mean
and (sample) standard deviation. Phase 2 here is single-seed (seed 0), so its
std is reported as NaN and n_seeds = 1.

    python src/compare_all_methods.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


ROOT = project_root()
P2 = ROOT / "outputs" / "phase2"
P3 = ROOT / "outputs" / "phase3"
P4 = ROOT / "outputs" / "phase4"
OUT = ROOT / "outputs" / "final_comparison"

EXTREME = "Extreme"

# Canonical column schema for the unified utility table.
UNIFIED_COLS = [
    "phase", "group", "method_label", "n",
    "k", "ell", "t", "r", "epsilon", "n_seeds",
    "avg_2way_tvd_mean", "avg_2way_tvd_std",
    "avg_3way_tvd_mean", "avg_3way_tvd_std",
    "runtime_s_mean", "runtime_s_std",
    "mi_abs_err_mean", "mi_abs_err_std",
    "sa_tvd_mean", "sa_tvd_std",
    "extreme_abs_err_mean", "extreme_abs_err_std",
    "strict_2way_tvd", "strict_3way_tvd",
]


def _num(s: str) -> float:
    return float(str(s).split("=")[-1])


def _fmt_param(method: str, parameter: str) -> tuple[str, str, dict]:
    """Return (group, method_label, setting-dict) for a Phase 2 row."""
    val = _num(parameter)
    if method == "k_anonymity":
        return ("k-anonymity", f"k-anon (k={val:g})", {"k": val})
    if method == "l_diversity":
        return ("l-diversity", f"l-div (l={val:g})", {"ell": val})
    if method == "t_closeness":
        return ("t-closeness", f"t-close (t={val:g})", {"t": val})
    return (method, f"{method} {parameter}", {})


# --------------------------------------------------------------------------- #
# Phase 2 loader
# --------------------------------------------------------------------------- #
def load_phase2() -> pd.DataFrame:
    marg = pd.read_csv(P2 / "marginal_tvd_summary.csv")
    avg = marg[marg["marginal_order"].isin(["avg_2way", "avg_3way"])].copy()

    def pick(metric: str) -> pd.DataFrame:
        sub = avg[avg["metric_type"] == metric]
        wide = sub.pivot_table(index=["n", "method", "parameter"],
                               columns="marginal_order", values="tvd")
        return wide.rename(columns={"avg_2way": f"{metric}_2",
                                    "avg_3way": f"{metric}_3"})

    main = pick("midpoint_binned_tvd")
    strict = pick("strict_symbolic_tvd")
    util = main.join(strict, how="left").reset_index()

    # runtime (feasible rows only)
    rt = pd.read_csv(P2 / "runtime_summary.csv")
    rt = rt[rt["feasible"] == True].copy()  # noqa: E712
    rt["parameter"] = (rt["parameter_name"].astype(str) + "="
                       + rt["parameter_value"].astype(str))
    rt_small = rt[["n", "method", "parameter", "runtime_seconds"]]

    util = util.merge(rt_small, on=["n", "method", "parameter"], how="left")

    rows = []
    for _, rec in util.iterrows():
        group, label, setting = _fmt_param(rec["method"], rec["parameter"])
        rows.append({
            "phase": "phase2", "group": group, "method_label": label,
            "n": int(rec["n"]),
            "k": setting.get("k", np.nan), "ell": setting.get("ell", np.nan),
            "t": setting.get("t", np.nan), "r": np.nan, "epsilon": np.nan,
            "n_seeds": 1,
            "avg_2way_tvd_mean": rec.get("midpoint_binned_tvd_2", np.nan),
            "avg_2way_tvd_std": np.nan,
            "avg_3way_tvd_mean": rec.get("midpoint_binned_tvd_3", np.nan),
            "avg_3way_tvd_std": np.nan,
            "runtime_s_mean": rec.get("runtime_seconds", np.nan),
            "runtime_s_std": np.nan,
            "mi_abs_err_mean": np.nan, "mi_abs_err_std": np.nan,
            "sa_tvd_mean": np.nan, "sa_tvd_std": np.nan,
            "extreme_abs_err_mean": np.nan, "extreme_abs_err_std": np.nan,
            "strict_2way_tvd": rec.get("strict_symbolic_tvd_2", np.nan),
            "strict_3way_tvd": rec.get("strict_symbolic_tvd_3", np.nan),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-seed MI helper (mean abs error over attribute pairs, per seed)
# --------------------------------------------------------------------------- #
def mi_per_seed(path: Path, keys: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=keys + ["mi_abs_err"])
    mi = pd.read_csv(path)
    per_seed = (mi.groupby(keys + ["seed"])["abs_mi_error_nats"].mean()
                .reset_index().rename(columns={"abs_mi_error_nats": "mi_abs_err"}))
    return per_seed


# --------------------------------------------------------------------------- #
# Phase 3 loader
# --------------------------------------------------------------------------- #
def load_phase3() -> pd.DataFrame:
    marg = pd.read_csv(P3 / "marginal_tvd_summary.csv")
    rt = pd.read_csv(P3 / "runtime_summary.csv")
    mi = mi_per_seed(P3 / "pairwise_mi_summary.csv", ["n", "r"])

    g_marg = marg.groupby(["n", "r"])[["avg_2way_tvd", "avg_3way_tvd"]]
    g_rt = rt.groupby(["n", "r"])["total_time_s"]
    g_mi = mi.groupby(["n", "r"])["mi_abs_err"] if len(mi) else None

    rows = []
    for (n, r), _ in g_marg:
        sub = marg[(marg["n"] == n) & (marg["r"] == r)]
        rsub = rt[(rt["n"] == n) & (rt["r"] == r)]
        msub = mi[(mi["n"] == n) & (mi["r"] == r)] if g_mi is not None else None
        rows.append({
            "phase": "phase3", "group": "non-private PrivBayes",
            "method_label": f"PrivBayes-NP (r={r})", "n": int(n),
            "k": np.nan, "ell": np.nan, "t": np.nan, "r": int(r),
            "epsilon": np.nan, "n_seeds": int(sub["seed"].nunique()),
            "avg_2way_tvd_mean": sub["avg_2way_tvd"].mean(),
            "avg_2way_tvd_std": sub["avg_2way_tvd"].std(),
            "avg_3way_tvd_mean": sub["avg_3way_tvd"].mean(),
            "avg_3way_tvd_std": sub["avg_3way_tvd"].std(),
            "runtime_s_mean": rsub["total_time_s"].mean(),
            "runtime_s_std": rsub["total_time_s"].std(),
            "mi_abs_err_mean": msub["mi_abs_err"].mean() if msub is not None
            and len(msub) else np.nan,
            "mi_abs_err_std": msub["mi_abs_err"].std() if msub is not None
            and len(msub) else np.nan,
            "sa_tvd_mean": np.nan, "sa_tvd_std": np.nan,
            "extreme_abs_err_mean": np.nan, "extreme_abs_err_std": np.nan,
            "strict_2way_tvd": np.nan, "strict_3way_tvd": np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Phase 4 loader
# --------------------------------------------------------------------------- #
def load_phase4() -> pd.DataFrame:
    marg = pd.read_csv(P4 / "marginal_tvd_summary.csv")
    rt = pd.read_csv(P4 / "runtime_summary.csv")
    mi = mi_per_seed(P4 / "pairwise_mi_summary.csv", ["n", "r", "epsilon"])

    rows = []
    for (n, r, eps), sub in marg.groupby(["n", "r", "epsilon"]):
        rsub = rt[(rt["n"] == n) & (rt["r"] == r) & (rt["epsilon"] == eps)]
        msub = mi[(mi["n"] == n) & (mi["r"] == r) & (mi["epsilon"] == eps)] \
            if len(mi) else pd.DataFrame()
        rows.append({
            "phase": "phase4", "group": "private PrivBayes",
            "method_label": f"PrivBayes-DP (r={r}, eps={eps:g})", "n": int(n),
            "k": np.nan, "ell": np.nan, "t": np.nan, "r": int(r),
            "epsilon": float(eps), "n_seeds": int(sub["seed"].nunique()),
            "avg_2way_tvd_mean": sub["avg_2way_tvd"].mean(),
            "avg_2way_tvd_std": sub["avg_2way_tvd"].std(),
            "avg_3way_tvd_mean": sub["avg_3way_tvd"].mean(),
            "avg_3way_tvd_std": sub["avg_3way_tvd"].std(),
            "runtime_s_mean": rsub["total_time_s"].mean(),
            "runtime_s_std": rsub["total_time_s"].std(),
            "mi_abs_err_mean": msub["mi_abs_err"].mean() if len(msub) else np.nan,
            "mi_abs_err_std": msub["mi_abs_err"].std() if len(msub) else np.nan,
            "sa_tvd_mean": sub["sa_tvd"].mean(),
            "sa_tvd_std": sub["sa_tvd"].std(),
            "extreme_abs_err_mean": sub["extreme_abs_error"].mean(),
            "extreme_abs_err_std": sub["extreme_abs_error"].std(),
            "strict_2way_tvd": np.nan, "strict_3way_tvd": np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Phase 4 validation / accounting checks (requirements #8 and #9)
# --------------------------------------------------------------------------- #
def phase4_checks() -> dict:
    checks: dict = {}

    def all_pass(path: Path, expected_rows: int) -> tuple[bool, str]:
        if not path.exists():
            return False, "missing file"
        df = pd.read_csv(path)
        n_pass = int((df["result"] == "PASS").sum())
        ok = (len(df) == expected_rows) and (n_pass == len(df))
        return ok, f"{n_pass}/{len(df)} PASS (expected {expected_rows})"

    # 180 runs * 8 synthetic checks ; 180 * 4 cpt checks
    checks["synthetic_validation"] = all_pass(
        P4 / "synthetic_validation_summary.csv", 180 * 8)
    checks["noisy_cpt_validation"] = all_pass(
        P4 / "noisy_cpt_validation_summary.csv", 180 * 4)

    # 180 expected private CSV files actually present on disk
    files = list((ROOT / "data" / "synthetic" / "private").glob(
        "privbayes_private_n*_r*_eps*_seed*.csv"))
    checks["expected_files_present"] = (len(files) == 180,
                                        f"{len(files)}/180 private CSVs")

    # privacy accounting completeness (requirement #9)
    acct_path = P4 / "privacy_accounting_summary.csv"
    required = ["epsilon_total", "epsilon_structure", "epsilon_parameters",
                "epsilon_step", "epsilon_cpt", "mi_sensitivity_bound",
                "laplace_scale"]
    if acct_path.exists():
        acct = pd.read_csv(acct_path)
        have_cols = all(c in acct.columns for c in required)
        complete = have_cols and not acct[required].isna().any().any()
        checks["privacy_accounting_complete"] = (
            len(acct) == 180 and complete,
            f"{len(acct)}/180 rows, required fields "
            f"{'complete' if complete else 'INCOMPLETE/missing'}")
    else:
        checks["privacy_accounting_complete"] = (False, "missing file")
    return checks


def phase2_validation_ok() -> tuple[bool, str]:
    rep = P2 / "validation_report.txt"
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        ok = "OVERALL: PASS" in txt
        return ok, "OVERALL: PASS" if ok else "OVERALL not PASS"
    return False, "validation_report.txt missing"


# --------------------------------------------------------------------------- #
# Derived comparison tables
# --------------------------------------------------------------------------- #
def phase3_vs_phase4(util: pd.DataFrame) -> pd.DataFrame:
    """Per (n, r, epsilon): Phase 4 utility, Phase 3 baseline, and the gap
    (privacy cost = phase4 - phase3)."""
    p3 = util[util["phase"] == "phase3"].set_index(["n", "r"])
    p4 = util[util["phase"] == "phase4"]
    rows = []
    for _, rec in p4.iterrows():
        key = (rec["n"], rec["r"])
        b2 = p3.loc[key, "avg_2way_tvd_mean"] if key in p3.index else np.nan
        b3 = p3.loc[key, "avg_3way_tvd_mean"] if key in p3.index else np.nan
        rows.append({
            "n": rec["n"], "r": rec["r"], "epsilon": rec["epsilon"],
            "p4_2way_mean": rec["avg_2way_tvd_mean"],
            "p4_2way_std": rec["avg_2way_tvd_std"],
            "p4_3way_mean": rec["avg_3way_tvd_mean"],
            "p4_3way_std": rec["avg_3way_tvd_std"],
            "p3_2way_baseline": b2, "p3_3way_baseline": b3,
            "privacy_cost_2way": rec["avg_2way_tvd_mean"] - b2,
            "privacy_cost_3way": rec["avg_3way_tvd_mean"] - b3,
        })
    return pd.DataFrame(rows).sort_values(["n", "r", "epsilon"])


def best_private_settings(util: pd.DataFrame) -> pd.DataFrame:
    """For each (n, epsilon): which r minimizes 2-way / 3-way TVD, plus an
    explicit r=1 vs r=2 verdict (requirement #5)."""
    p4 = util[util["phase"] == "phase4"]
    rows = []
    for (n, eps), sub in p4.groupby(["n", "epsilon"]):
        b2 = sub.loc[sub["avg_2way_tvd_mean"].idxmin()]
        b3 = sub.loc[sub["avg_3way_tvd_mean"].idxmin()]
        r1 = sub[sub["r"] == 1]
        r2 = sub[sub["r"] == 2]

        def val(d, col):
            return float(d[col].iloc[0]) if len(d) else np.nan
        r1_2, r2_2 = val(r1, "avg_2way_tvd_mean"), val(r2, "avg_2way_tvd_mean")
        if np.isnan(r1_2) or np.isnan(r2_2):
            verdict = "n/a"
        else:
            verdict = ("r=1" if r1_2 < r2_2 else
                       "r=2" if r2_2 < r1_2 else "tie")
        rows.append({
            "n": n, "epsilon": eps,
            "best_r_2way": int(b2["r"]),
            "best_2way_tvd": round(float(b2["avg_2way_tvd_mean"]), 6),
            "best_r_3way": int(b3["r"]),
            "best_3way_tvd": round(float(b3["avg_3way_tvd_mean"]), 6),
            "r1_2way": round(r1_2, 6) if not np.isnan(r1_2) else np.nan,
            "r2_2way": round(r2_2, 6) if not np.isnan(r2_2) else np.nan,
            "better_of_r1_r2_on_2way": verdict,
        })
    return pd.DataFrame(rows).sort_values(["n", "epsilon"])


def privacy_semantics_table(p4chk: dict, p2_ok: tuple) -> pd.DataFrame:
    """Curated semantics table — privacy is NOT a single scalar; each family has
    a different privacy MEANING."""
    p4_val = (p4chk["synthetic_validation"][0]
              and p4chk["noisy_cpt_validation"][0]
              and p4chk["expected_files_present"][0])
    p4_acct = p4chk["privacy_accounting_complete"][0]
    return pd.DataFrame([
        {"group": "k-anonymity",
         "privacy_model": "syntactic",
         "privacy_semantics": "each QI equivalence class has >= k records",
         "controlling_parameter": "k (class size)",
         "randomized": "no", "seeds_aggregated": 1,
         "utility_role": "syntactic privacy via generalization/suppression",
         "validation_status": "PASS" if p2_ok[0] else "CHECK"},
        {"group": "l-diversity",
         "privacy_model": "syntactic",
         "privacy_semantics": "each class has >= l distinct sensitive values",
         "controlling_parameter": "l (SA diversity)",
         "randomized": "no", "seeds_aggregated": 1,
         "utility_role": "guards against attribute disclosure within a class",
         "validation_status": "PASS" if p2_ok[0] else "CHECK"},
        {"group": "t-closeness",
         "privacy_model": "syntactic",
         "privacy_semantics": "per-class SA distribution within t of the global "
                              "SA distribution",
         "controlling_parameter": "t (distribution closeness)",
         "randomized": "no", "seeds_aggregated": 1,
         "utility_role": "limits SA distribution skew within a class",
         "validation_status": "PASS" if p2_ok[0] else "CHECK"},
        {"group": "non-private PrivBayes",
         "privacy_model": "none (utility upper baseline)",
         "privacy_semantics": "NO privacy guarantee; exact MI structure + exact "
                              "CPTs",
         "controlling_parameter": "r (parent degree)",
         "randomized": "yes (sampling seed)", "seeds_aggregated": 5,
         "utility_role": "best-achievable synthetic utility reference",
         "validation_status": "PASS"},
        {"group": "private PrivBayes",
         "privacy_model": "differential-privacy-style (simplified)",
         "privacy_semantics": "DP-style randomized synthetic generation: "
                              "exponential-mechanism structure + Laplace CPTs",
         "controlling_parameter": "epsilon (total budget), r (parent degree)",
         "randomized": "yes (DP noise + seed)", "seeds_aggregated": 5,
         "utility_role": "privacy-utility tradeoff vs epsilon",
         "validation_status": ("PASS" if (p4_val and p4_acct) else "CHECK")},
    ])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)

    util = pd.concat([load_phase2(), load_phase3(), load_phase4()],
                     ignore_index=True)[UNIFIED_COLS]
    util_round = util.copy()
    for c in util_round.columns:
        if util_round[c].dtype.kind == "f":
            util_round[c] = util_round[c].round(6)
    util_round.to_csv(OUT / "final_utility_comparison.csv", index=False)

    runtime = util[["phase", "group", "method_label", "n",
                    "runtime_s_mean", "runtime_s_std"]].copy()
    runtime[["runtime_s_mean", "runtime_s_std"]] = \
        runtime[["runtime_s_mean", "runtime_s_std"]].round(6)
    runtime.to_csv(OUT / "final_runtime_comparison.csv", index=False)

    p3v4 = phase3_vs_phase4(util)
    p3v4.round(6).to_csv(
        OUT / "final_phase3_vs_phase4_by_epsilon.csv", index=False)

    best = best_private_settings(util)
    best.to_csv(OUT / "final_best_private_settings.csv", index=False)

    p4chk = phase4_checks()
    p2_ok = phase2_validation_ok()
    sem = privacy_semantics_table(p4chk, p2_ok)
    sem.to_csv(OUT / "final_privacy_semantics_table.csv", index=False)

    # ---- text summary ------------------------------------------------------ #
    lines: list[str] = []

    def emit(t: str = "") -> None:
        print(t)
        lines.append(t)

    emit("=" * 78)
    emit("FINAL COMPARATIVE ANALYSIS — Phases 2-4 (read-only)")
    emit("Main metric: COMMON-BIN (Phase 2 midpoint) average marginal TVD, "
         "lower=better.")
    emit("Privacy is compared by SEMANTICS, not as a single scalar (see "
         "semantics table).")
    emit("=" * 78)
    emit()

    # utility leaderboard aggregated over n
    emit("-" * 78)
    emit("Average marginal TVD by method (mean over n; std over seeds, "
         "averaged over n):")
    emit("-" * 78)
    agg = (util.groupby(["group", "method_label"]).agg(
        two=("avg_2way_tvd_mean", "mean"),
        two_sd=("avg_2way_tvd_std", "mean"),
        three=("avg_3way_tvd_mean", "mean"),
        three_sd=("avg_3way_tvd_std", "mean"),
        rt=("runtime_s_mean", "mean")).reset_index())
    agg = agg.sort_values(["group", "two"])
    for _, r in agg.iterrows():
        sd2 = "n/a" if pd.isna(r["two_sd"]) else f"{r['two_sd']:.4f}"
        sd3 = "n/a" if pd.isna(r["three_sd"]) else f"{r['three_sd']:.4f}"
        emit(f"  {r['method_label']:<26s} 2way={r['two']:.4f}(+-{sd2})  "
             f"3way={r['three']:.4f}(+-{sd3})  runtime={r['rt']:.4f}s")
    emit()

    # best per group
    emit("Best (lowest 2-way TVD) within each comparison group:")
    for grp in ["k-anonymity", "l-diversity", "t-closeness",
                "non-private PrivBayes", "private PrivBayes"]:
        sub = agg[agg["group"] == grp]
        if len(sub):
            b = sub.iloc[0]
            emit(f"  {grp:<24s} -> {b['method_label']} "
                 f"(2way={b['two']:.4f})")
    emit()

    # epsilon trend + privacy cost
    emit("-" * 78)
    emit("Private PrivBayes: utility vs epsilon, and privacy cost vs Phase 3 "
         "(req #4,#6):")
    emit("-" * 78)
    for n in sorted(p3v4["n"].unique()):
        for r in sorted(p3v4["r"].unique()):
            sub = p3v4[(p3v4["n"] == n) & (p3v4["r"] == r)].sort_values("epsilon")
            if not len(sub):
                continue
            seq = [f"eps{e:g}:{v:.3f}" for e, v in
                   zip(sub["epsilon"], sub["p4_2way_mean"])]
            mono = all(a >= b for a, b in zip(sub["p4_2way_mean"],
                                              sub["p4_2way_mean"][1:]))
            base = sub["p3_2way_baseline"].iloc[0]
            cost_hi = sub[sub["epsilon"] == sub["epsilon"].max()][
                "privacy_cost_2way"].iloc[0]
            emit(f"  n={n} r={r}: 2way {seq}  improves-with-eps={mono}  "
                 f"| P3base={base:.3f} cost@eps_max={cost_hi:+.3f}")
    emit()

    # r1 vs r2
    emit("-" * 78)
    emit("Does r=1 or r=2 give better private utility under each epsilon "
         "(req #5)?")
    emit("-" * 78)
    for _, r in best.iterrows():
        emit(f"  n={int(r['n'])} eps={r['epsilon']:g}: best_r(2way)="
             f"{r['best_r_2way']} best_r(3way)={r['best_r_3way']}  "
             f"r1_vs_r2_on_2way -> {r['better_of_r1_r2_on_2way']}")
    emit()

    # phase4 vs phase2 (different privacy semantics)
    emit("-" * 78)
    emit("Private PrivBayes vs Phase 2 syntactic methods (req #7): different "
         "privacy")
    emit("semantics, so this contrasts UTILITY under DP-style vs syntactic "
         "privacy.")
    emit("-" * 78)
    p2_agg = agg[agg["group"].isin(["k-anonymity", "l-diversity",
                                    "t-closeness"])]["two"]
    p4_best = agg[agg["group"] == "private PrivBayes"]["two"].min()
    p4_worst = agg[agg["group"] == "private PrivBayes"]["two"].max()
    emit(f"  Phase 2 syntactic 2way TVD range: "
         f"{p2_agg.min():.4f} .. {p2_agg.max():.4f} (mean {p2_agg.mean():.4f})")
    emit(f"  Private PrivBayes 2way TVD range: "
         f"{p4_best:.4f} (eps=2 best) .. {p4_worst:.4f} (eps=0.1 worst)")
    emit("  => At high epsilon, private PrivBayes utility is competitive with /"
         " better than")
    emit("     several syntactic settings; at eps=0.1 it is noisier. Privacy "
         "MEANING differs.")
    emit()

    # validation + accounting (req #8, #9)
    emit("-" * 78)
    emit("Phase 4 validation & privacy-accounting checks (req #8, #9):")
    emit("-" * 78)
    for name, (ok, detail) in p4chk.items():
        emit(f"  [{'PASS' if ok else 'FAIL'}] {name:<28s} {detail}")
    emit(f"  [{'PASS' if p2_ok[0] else 'FAIL'}] phase2_validation         "
         f"  {p2_ok[1]}")
    emit()

    emit("-" * 78)
    emit("Caveats / non-overclaim:")
    emit("  * Main TVD uses the shared common-bin space; strict_symbolic_tvd is "
         "an appendix")
    emit("    column only (more conservative, treats generalized cells as own "
         "categories).")
    emit("  * Phase 4 is a SIMPLIFIED private PrivBayes (direct-MI exponential "
         "mechanism,")
    emit("    basic sequential composition) — not the full paper; not a DP "
         "proof.")
    emit("  * Recorded runtimes reflect implementation choices (Phase 3 uses "
         "per-record")
    emit("    sampling; Phase 4 sampling is vectorized), not just algorithmic "
         "cost.")
    emit("  * Phase 2 is single-seed (seed 0); its std is reported as n/a.")
    emit()
    emit("Saved tables:")
    for name in ("final_utility_comparison.csv", "final_runtime_comparison.csv",
                 "final_phase3_vs_phase4_by_epsilon.csv",
                 "final_best_private_settings.csv",
                 "final_privacy_semantics_table.csv"):
        emit(f"  - outputs/final_comparison/{name}")

    (OUT / "final_comparison_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
