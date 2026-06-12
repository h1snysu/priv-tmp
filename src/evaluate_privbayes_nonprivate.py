#!/usr/bin/env python3
"""Phase 3 — utility evaluation of the NON-PRIVATE PrivBayes synthetic data.

This measures how faithfully the non-private baseline reproduces the raw data's
distributions. It is a *utility upper baseline*: because no differential privacy
is applied, these numbers are the best fidelity a private PrivBayes (Phase 4)
could hope to approach, not a privacy guarantee.

Metrics (raw vs synthetic, per n / r / seed, then aggregated by n and r):

  * average 2-way marginal TVD
  * average 3-way marginal TVD
      Both raw and synthetic are mapped into the SHARED Phase 2 evaluation bin
      space (see privbayes_utils.eval_bin_frame) so the comparison is fair —
      raw exact values and synthetic midpoint values live in the same category
      space — and so these TVDs are directly comparable to the Phase 2 table.

  * pairwise mutual information preservation: raw MI, synthetic MI, |error|.
      MI is computed in nats on the PrivBayes discretization of each frame, so
      raw and synthetic are discretised identically before MI.

  * runtime summary (read back from runtime_summary.csv).

Run from the project root with:

    python src/evaluate_privbayes_nonprivate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import privbayes_utils as U


def raw_eval_marginals(raw: pd.DataFrame) -> dict:
    """Raw joint distributions over the shared Phase 2 eval bins, per subset."""
    binned = U.eval_bin_frame(raw)
    return {tuple(attrs): U.joint_distribution(binned, attrs)
            for _, attrs in U.marginal_subsets()}


def main() -> None:
    runtime_path = U.phase3_dir() / "runtime_summary.csv"
    if not runtime_path.exists():
        raise SystemExit(f"missing {runtime_path}; run "
                         f"src/privbayes_nonprivate.py first")
    runtime = pd.read_csv(runtime_path)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 3 — UTILITY EVALUATION (non-private PrivBayes upper baseline)")
    emit("NOTE: high fidelity here is expected; it reflects NO privacy cost.")
    emit("=" * 74)
    emit()

    # raw caches keyed by n
    raw_cache: dict = {}
    raw_marg_cache: dict = {}
    raw_disc_cache: dict = {}
    raw_mi_cache: dict = {}

    tvd_rows: list[dict] = []
    mi_rows: list[dict] = []

    for _, row in runtime.iterrows():
        n, r, seed = int(row["n"]), int(row["r"]), int(row["seed"])
        synth_path = U.synthetic_path(n, r, seed)
        if not synth_path.exists():
            emit(f"[SKIP] missing {synth_path.name}")
            continue

        if n not in raw_cache:
            raw = U.load_raw(n)
            raw_cache[n] = raw
            raw_marg_cache[n] = raw_eval_marginals(raw)
            raw_disc_cache[n] = U.discretize(raw)
            raw_mi_cache[n] = U.pairwise_mutual_information(raw_disc_cache[n])

        synth = pd.read_csv(synth_path)

        # ---- marginal TVD in shared Phase 2 eval bins ------------------- #
        synth_binned = U.eval_bin_frame(synth)
        two, three = [], []
        for order, attrs in U.marginal_subsets():
            synth_dist = U.joint_distribution(synth_binned, attrs)
            tvd = U.tvd_between(raw_marg_cache[n][tuple(attrs)], synth_dist)
            (two if order == 2 else three).append(tvd)
        avg2 = float(np.mean(two))
        avg3 = float(np.mean(three))
        tvd_rows.append({"n": n, "r": r, "seed": seed,
                         "avg_2way_tvd": round(avg2, 6),
                         "avg_3way_tvd": round(avg3, 6)})

        # ---- pairwise MI preservation (nats, on PrivBayes discretisation) #
        synth_disc = U.discretize(synth)
        synth_mi = U.pairwise_mutual_information(synth_disc)
        for (a, b), raw_mi in raw_mi_cache[n].items():
            s_mi = synth_mi[(a, b)]
            mi_rows.append({"n": n, "r": r, "seed": seed,
                            "var_a": a, "var_b": b,
                            "raw_mi_nats": round(raw_mi, 6),
                            "synthetic_mi_nats": round(s_mi, 6),
                            "abs_mi_error_nats": round(abs(raw_mi - s_mi), 6)})

    tvd_df = pd.DataFrame(tvd_rows)
    mi_df = pd.DataFrame(mi_rows)

    # ---- aggregate by (n, r): mean over seeds --------------------------- #
    tvd_agg = (tvd_df.groupby(["n", "r"])[["avg_2way_tvd", "avg_3way_tvd"]]
               .mean().round(6).reset_index())
    mi_agg = (mi_df.groupby(["n", "r"])[
        ["raw_mi_nats", "synthetic_mi_nats", "abs_mi_error_nats"]]
        .mean().round(6).reset_index())

    # attach runtime means by (n, r)
    rt_agg = (runtime.groupby(["n", "r"])[
        ["discretization_time_s", "structure_learning_time_s",
         "cpt_fitting_time_s", "sampling_time_s", "total_time_s"]]
        .mean().round(6).reset_index())

    tvd_out = tvd_df.merge(tvd_agg, on=["n", "r"], suffixes=("", "_mean_by_nr"))
    tvd_out.to_csv(U.phase3_dir() / "marginal_tvd_summary.csv", index=False)
    mi_df.to_csv(U.phase3_dir() / "pairwise_mi_summary.csv", index=False)

    # ---- report -------------------------------------------------------- #
    emit("Marginal TVD vs raw — mean over seeds, by (n, r) "
         "[lower is better]:")
    emit(tvd_agg.to_string(index=False))
    emit()
    emit("Sanity-check expectation: avg TVD should DECREASE as r grows "
         "(0 -> 1 -> 2),")
    emit("most clearly for 3-way marginals (r=0 ignores all dependencies).")
    emit()
    for n in U.DATASET_SIZES:
        sub = tvd_agg[tvd_agg["n"] == n].sort_values("r")
        if len(sub) == 3:
            t2 = sub["avg_2way_tvd"].tolist()
            t3 = sub["avg_3way_tvd"].tolist()
            mono2 = "yes" if t2[0] >= t2[1] >= t2[2] else "no"
            mono3 = "yes" if t3[0] >= t3[1] >= t3[2] else "no"
            emit(f"  n={n}: 2way r0>r1>r2 monotone? {mono2}  "
                 f"({t2[0]:.4f}->{t2[1]:.4f}->{t2[2]:.4f}); "
                 f"3way? {mono3} "
                 f"({t3[0]:.4f}->{t3[1]:.4f}->{t3[2]:.4f})")
    emit()
    emit("Pairwise mutual information preservation — mean over seeds, "
         "by (n, r) [nats]:")
    emit(mi_agg.to_string(index=False))
    emit()
    emit("Runtime breakdown — mean seconds over seeds, by (n, r):")
    emit(rt_agg.to_string(index=False))
    emit()
    emit("Saved:")
    emit(f"  - {(U.phase3_dir() / 'marginal_tvd_summary.csv').relative_to(U.project_root())}")
    emit(f"  - {(U.phase3_dir() / 'pairwise_mi_summary.csv').relative_to(U.project_root())}")

    # append evaluation section to the phase 3 summary
    summary_path = U.phase3_dir() / "phase3_summary.txt"
    mode = "a" if summary_path.exists() else "w"
    with open(summary_path, mode, encoding="utf-8") as fh:
        if mode == "a":
            fh.write("\n")
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
