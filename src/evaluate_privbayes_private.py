#!/usr/bin/env python3
"""Phase 4 — utility evaluation of the PRIVATE PrivBayes synthetic data.

Measures how faithfully the *private* baseline reproduces the raw distributions,
and how much utility is lost relative to the Phase 3 non-private upper baseline.

Metrics (raw vs synthetic), aggregated by (n, r, epsilon) with mean+std over the
5 seeds:

  * average 2-way / 3-way marginal TVD, in the SHARED Phase 2 evaluation bin
    space (privbayes_utils.eval_bin_frame) so Phase 2/3/4 numbers are
    comparable;
  * pairwise mutual information preservation: raw MI, synthetic MI, |error|
    (nats, on the fixed PrivBayes discretization of each frame);
  * income_class (SA) distribution TVD, plus the absolute error on the rare
    "Extreme" class specifically.

Comparisons:
  * Phase 3 non-private PrivBayes at the same (n, r);
  * Phase 2 syntactic anonymization aggregate (midpoint-binned TVD), for context.

    python src/evaluate_privbayes_private.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import privbayes_utils as U

EXTREME = "Extreme"


def raw_eval_marginals(raw: pd.DataFrame) -> dict:
    binned = U.eval_bin_frame(raw)
    return {tuple(attrs): U.joint_distribution(binned, attrs)
            for _, attrs in U.marginal_subsets()}


def sa_distribution(df: pd.DataFrame) -> dict:
    counts = df[U.SA].astype(str).value_counts()
    total = counts.sum()
    return {v: counts.get(v, 0) / total for v in U.INCOME_DOMAIN}


def main() -> None:
    runtime_path = U.phase4_dir() / "runtime_summary.csv"
    if not runtime_path.exists():
        raise SystemExit(f"missing {runtime_path}; run "
                         f"src/privbayes_private.py first")
    runtime = pd.read_csv(runtime_path)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 4 — UTILITY EVALUATION (simplified private PrivBayes)")
    emit("Private utility is expected to be WORSE than Phase 3 (non-private) and"
         " to")
    emit("improve as epsilon grows. This reflects the cost of privacy, not a "
         "bug.")
    emit("=" * 74)
    emit()

    raw_cache: dict = {}
    raw_marg_cache: dict = {}
    raw_disc_cache: dict = {}
    raw_mi_cache: dict = {}
    raw_sa_cache: dict = {}

    tvd_rows: list[dict] = []
    mi_rows: list[dict] = []
    sa_rows: list[dict] = []

    for _, row in runtime.iterrows():
        n, r = int(row["n"]), int(row["r"])
        eps, seed = float(row["epsilon"]), int(row["seed"])
        path = U.private_synthetic_path(n, r, eps, seed)
        if not path.exists():
            emit(f"[SKIP] missing {path.name}")
            continue

        if n not in raw_cache:
            raw = U.load_raw(n)
            raw_cache[n] = raw
            raw_marg_cache[n] = raw_eval_marginals(raw)
            raw_disc_cache[n] = U.discretize(raw)
            raw_mi_cache[n] = U.pairwise_mutual_information(raw_disc_cache[n])
            raw_sa_cache[n] = sa_distribution(raw)

        synth = pd.read_csv(path)

        # ---- marginal TVD (shared eval bins) --------------------------- #
        synth_binned = U.eval_bin_frame(synth)
        two, three = [], []
        for order, attrs in U.marginal_subsets():
            sdist = U.joint_distribution(synth_binned, attrs)
            tvd = U.tvd_between(raw_marg_cache[n][tuple(attrs)], sdist)
            (two if order == 2 else three).append(tvd)
        tvd_rows.append({"n": n, "r": r, "epsilon": eps, "seed": seed,
                         "avg_2way_tvd": round(float(np.mean(two)), 6),
                         "avg_3way_tvd": round(float(np.mean(three)), 6)})

        # ---- pairwise MI preservation ---------------------------------- #
        smi = U.pairwise_mutual_information(U.discretize(synth))
        for (a, b), rmi in raw_mi_cache[n].items():
            mi_rows.append({"n": n, "r": r, "epsilon": eps, "seed": seed,
                            "var_a": a, "var_b": b,
                            "raw_mi_nats": round(rmi, 6),
                            "synthetic_mi_nats": round(smi[(a, b)], 6),
                            "abs_mi_error_nats": round(abs(rmi - smi[(a, b)]),
                                                       6)})

        # ---- SA distribution TVD + Extreme error ----------------------- #
        s_sa = sa_distribution(synth)
        r_sa = raw_sa_cache[n]
        sa_tvd = 0.5 * sum(abs(s_sa[v] - r_sa[v]) for v in U.INCOME_DOMAIN)
        sa_rows.append({"n": n, "r": r, "epsilon": eps, "seed": seed,
                        "sa_tvd": round(sa_tvd, 6),
                        "extreme_raw_p": round(r_sa[EXTREME], 6),
                        "extreme_synth_p": round(s_sa[EXTREME], 6),
                        "extreme_abs_error": round(abs(r_sa[EXTREME]
                                                       - s_sa[EXTREME]), 6)})

    tvd_df = pd.DataFrame(tvd_rows)
    mi_df = pd.DataFrame(mi_rows)
    sa_df = pd.DataFrame(sa_rows)

    # ---- aggregate by (n, r, epsilon): mean + std over seeds ----------- #
    def agg(df, cols):
        g = df.groupby(["n", "r", "epsilon"])[cols]
        out = g.mean().round(6)
        std = g.std(ddof=0).round(6)
        out.columns = [f"{c}_mean" for c in cols]
        for c in cols:
            out[f"{c}_std"] = std[c]
        return out.reset_index()

    tvd_agg = agg(tvd_df, ["avg_2way_tvd", "avg_3way_tvd"])
    sa_agg = agg(sa_df, ["sa_tvd", "extreme_abs_error"])
    mi_agg = (mi_df.groupby(["n", "r", "epsilon"])["abs_mi_error_nats"]
              .agg(["mean", "std"]).round(6)
              .rename(columns={"mean": "abs_mi_error_mean",
                               "std": "abs_mi_error_std"}).reset_index())

    # merge per-run with their (n,r,eps) aggregates for the saved summary
    tvd_out = tvd_df.merge(tvd_agg, on=["n", "r", "epsilon"])
    tvd_out = tvd_out.merge(sa_df[["n", "r", "epsilon", "seed", "sa_tvd",
                                   "extreme_abs_error"]],
                            on=["n", "r", "epsilon", "seed"])
    tvd_out.to_csv(U.phase4_dir() / "marginal_tvd_summary.csv", index=False)
    mi_df.to_csv(U.phase4_dir() / "pairwise_mi_summary.csv", index=False)

    # ---- Phase 3 non-private baseline (same n, r) ---------------------- #
    p3_path = U.phase3_dir() / "marginal_tvd_summary.csv"
    p3 = None
    if p3_path.exists():
        p3raw = pd.read_csv(p3_path)
        p3 = (p3raw.groupby(["n", "r"])[["avg_2way_tvd", "avg_3way_tvd"]]
              .mean().round(6).reset_index()
              .rename(columns={"avg_2way_tvd": "p3_2way",
                               "avg_3way_tvd": "p3_3way"}))

    # ---- report -------------------------------------------------------- #
    emit("Marginal TVD vs raw (shared eval bins) — mean over seeds "
         "[lower=better]:")
    show = tvd_agg.copy()
    if p3 is not None:
        show = show.merge(p3, on=["n", "r"], how="left")
    emit(show.to_string(index=False))
    emit()

    emit("Trend check — does TVD improve (decrease) as epsilon grows? "
         "(fixed n, r)")
    for n in U.DATASET_SIZES:
        for r in U.R_VALUES:
            sub = tvd_agg[(tvd_agg["n"] == n) & (tvd_agg["r"] == r)] \
                .sort_values("epsilon")
            if len(sub) == len(U.EPSILONS):
                seq = sub["avg_2way_tvd_mean"].tolist()
                mono = "yes" if all(seq[i] >= seq[i + 1]
                                    for i in range(len(seq) - 1)) else "no"
                p3v = ""
                if p3 is not None:
                    m = p3[(p3["n"] == n) & (p3["r"] == r)]
                    if len(m):
                        p3v = f" | phase3_2way={m['p3_2way'].iloc[0]:.4f}"
                emit(f"  n={n} r={r}: 2way over eps"
                     f"{U.EPSILONS} = "
                     f"{[round(s, 4) for s in seq]} monotone↓? {mono}{p3v}")
    emit()

    emit("SA (income_class) distribution TVD + rare 'Extreme' abs error — "
         "mean over seeds:")
    emit(sa_agg.to_string(index=False))
    emit()

    emit("Pairwise MI absolute error (nats) — mean over seeds, by (n, r, eps):")
    emit(mi_agg.to_string(index=False))
    emit()

    # Phase 2 context (optional): mean midpoint-binned avg TVD by method.
    p2_path = U.phase2_dir() if hasattr(U, "phase2_dir") else None
    p2_file = U.project_root() / "outputs" / "phase2" / "marginal_tvd_summary.csv"
    if p2_file.exists():
        p2 = pd.read_csv(p2_file)
        p2m = p2[(p2["metric_type"] == "midpoint_binned_tvd")
                 & (p2["marginal_order"].isin(["avg_2way", "avg_3way"]))]
        piv = (p2m.pivot_table(index="method", columns="marginal_order",
                               values="tvd", aggfunc="mean").round(4))
        emit("Phase 2 context — mean midpoint-binned TVD by syntactic method:")
        emit(piv.to_string())
        emit()

    emit("Saved:")
    emit(f"  - {(U.phase4_dir() / 'marginal_tvd_summary.csv').relative_to(U.project_root())}")
    emit(f"  - {(U.phase4_dir() / 'pairwise_mi_summary.csv').relative_to(U.project_root())}")

    summary_path = U.phase4_dir() / "phase4_summary.txt"
    mode = "a" if summary_path.exists() else "w"
    with open(summary_path, mode, encoding="utf-8") as fh:
        if mode == "a":
            fh.write("\n")
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
