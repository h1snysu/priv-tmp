#!/usr/bin/env python3
"""Final comparison figures (read-only).

Reads the tables produced by src/compare_all_methods.py from
outputs/final_comparison/ and renders six PNG figures into
outputs/final_comparison/figures/. It regenerates no data and changes no
Phase 1-4 behavior.

    python src/visualize_results.py     # run compare_all_methods.py first
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (used via matplotlib.ticker.ScalarFormatter)
import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


OUT = project_root() / "outputs" / "final_comparison"
FIG = OUT / "figures"

GROUP_ORDER = ["k-anonymity", "l-diversity", "t-closeness",
               "non-private PrivBayes", "private PrivBayes"]
GROUP_COLOR = {
    "k-anonymity": "#4e79a7", "l-diversity": "#59a14f",
    "t-closeness": "#9c755f", "non-private PrivBayes": "#f28e2b",
    "private PrivBayes": "#e15759",
}
R_COLOR = {0: "#4e79a7", 1: "#59a14f", 2: "#e15759"}


def _require(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}; run src/compare_all_methods.py first")
    return pd.read_csv(path)


def _method_sort_key(row: pd.Series) -> tuple:
    g = GROUP_ORDER.index(row["group"]) if row["group"] in GROUP_ORDER else 9
    return (g, row.get("r", 99) if pd.notna(row.get("r")) else 99,
            row.get("epsilon", 99) if pd.notna(row.get("epsilon")) else -1,
            row["method_label"])


# --------------------------------------------------------------------------- #
# Bar chart: TVD by method (aggregated over n)
# --------------------------------------------------------------------------- #
def bar_tvd_by_method(util: pd.DataFrame, order: int, fname: str) -> None:
    mean_col = f"avg_{order}way_tvd_mean"
    std_col = f"avg_{order}way_tvd_std"
    agg = (util.groupby(["group", "method_label"], sort=False)
           .agg(mean=(mean_col, "mean"), std=(std_col, "mean"),
                r=("r", "first"), epsilon=("epsilon", "first"))
           .reset_index())
    agg["key"] = agg.apply(_method_sort_key, axis=1)
    agg = agg.sort_values("key").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(agg))
    colors = [GROUP_COLOR.get(g, "#777") for g in agg["group"]]
    err = agg["std"].fillna(0.0).to_numpy()
    ax.bar(x, agg["mean"], yerr=err, color=colors, capsize=3, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["method_label"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(f"avg {order}-way marginal TVD  (lower = better)")
    ax.set_title(f"Average {order}-way marginal TVD by method "
                 f"(common-bin; mean over n, error = seed std)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g])
               for g in GROUP_ORDER]
    ax.legend(handles, GROUP_ORDER, fontsize=8, title="comparison group")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Line chart: private TVD vs epsilon (per r, averaged over n)
# --------------------------------------------------------------------------- #
def line_private_vs_epsilon(p3v4: pd.DataFrame, order: int, fname: str) -> None:
    p4col = f"p4_{order}way_mean"
    basecol = f"p3_{order}way_baseline"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in sorted(p3v4["r"].unique()):
        sub = p3v4[p3v4["r"] == r]
        g = sub.groupby("epsilon")[p4col].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["epsilon"], g["mean"],
                    yerr=g["std"].fillna(0.0), marker="o",
                    color=R_COLOR.get(int(r)), capsize=3,
                    label=f"private r={int(r)}")
        base = sub.groupby("epsilon")[basecol].mean().mean()
        ax.axhline(base, ls="--", lw=1, color=R_COLOR.get(int(r)), alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(sorted(p3v4["epsilon"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("epsilon (log scale)")
    ax.set_ylabel(f"avg {order}-way marginal TVD  (lower = better)")
    ax.set_title(f"Private PrivBayes {order}-way TVD vs epsilon "
                 f"(mean over n)\ndashed = Phase 3 non-private baseline per r")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Line chart: privacy cost (phase4 - phase3) vs epsilon
# --------------------------------------------------------------------------- #
def line_privacy_gap(p3v4: pd.DataFrame, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in sorted(p3v4["r"].unique()):
        sub = p3v4[p3v4["r"] == r]
        g = sub.groupby("epsilon")["privacy_cost_2way"].mean().reset_index()
        ax.plot(g["epsilon"], g["privacy_cost_2way"], marker="o",
                color=R_COLOR.get(int(r)), label=f"r={int(r)}")
    ax.axhline(0.0, ls="--", color="black", lw=1, alpha=0.6,
               label="non-private baseline")
    ax.set_xscale("log")
    ax.set_xticks(sorted(p3v4["epsilon"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("epsilon (log scale)")
    ax.set_ylabel("privacy cost = TVD(private) - TVD(non-private), 2-way")
    ax.set_title("Privacy cost of Phase 4 vs Phase 3 non-private baseline\n"
                 "(closer to 0 = smaller utility loss; shrinks as epsilon grows)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Bar chart: runtime by method (log scale)
# --------------------------------------------------------------------------- #
def bar_runtime(runtime: pd.DataFrame, util: pd.DataFrame, fname: str) -> None:
    agg = (runtime.groupby(["group", "method_label"], sort=False)
           .agg(rt=("runtime_s_mean", "mean")).reset_index())
    meta = util.groupby("method_label").agg(
        r=("r", "first"), epsilon=("epsilon", "first")).reset_index()
    agg = agg.merge(meta, on="method_label", how="left")
    agg["key"] = agg.apply(_method_sort_key, axis=1)
    agg = agg.sort_values("key").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(agg))
    colors = [GROUP_COLOR.get(g, "#777") for g in agg["group"]]
    ax.bar(x, agg["rt"].clip(lower=1e-6), color=colors, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["method_label"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("mean total runtime per dataset (s, log scale)")
    ax.set_title("Runtime by method (mean over n; recorded times reflect "
                 "implementation,\ne.g. Phase 3 per-record vs Phase 4 vectorized "
                 "sampling)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLOR[g])
               for g in GROUP_ORDER]
    ax.legend(handles, GROUP_ORDER, fontsize=8, title="comparison group")
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    util = _require(OUT / "final_utility_comparison.csv")
    runtime = _require(OUT / "final_runtime_comparison.csv")
    p3v4 = _require(OUT / "final_phase3_vs_phase4_by_epsilon.csv")

    bar_tvd_by_method(util, 2, "tvd_2way_by_method.png")
    bar_tvd_by_method(util, 3, "tvd_3way_by_method.png")
    line_private_vs_epsilon(p3v4, 2, "private_tvd_by_epsilon_2way.png")
    line_private_vs_epsilon(p3v4, 3, "private_tvd_by_epsilon_3way.png")
    line_privacy_gap(p3v4, "phase3_vs_phase4_gap.png")
    bar_runtime(runtime, util, "runtime_by_method.png")

    print("Saved figures to outputs/final_comparison/figures/:")
    for name in ("tvd_2way_by_method.png", "tvd_3way_by_method.png",
                 "private_tvd_by_epsilon_2way.png",
                 "private_tvd_by_epsilon_3way.png",
                 "phase3_vs_phase4_gap.png", "runtime_by_method.png"):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
