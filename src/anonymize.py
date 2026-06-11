#!/usr/bin/env python3
"""Phase 2 — k-anonymity, l-diversity, and t-closeness via global recoding.

This script anonymizes the Phase 1 simulated datasets using *global*
generalization (full-domain recoding): every value of a quasi-identifier is
generalized to the same level across the whole dataset. For each privacy
setting we exhaustively search all 5^3 combinations of QI generalization
levels, keep every combination that satisfies the privacy target, and pick the
feasible one with minimum information loss. Suppression is only used as a
fallback when no global recoding can satisfy the target.

The sensitive attribute (income_class) is never modified.

Run from the project root with:

    python src/anonymize.py
"""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
QIS = ["age", "education_num", "hours_per_week"]
SA = "income_class"
INCOME_DOMAIN = ["Low", "LowerMiddle", "UpperMiddle", "High", "VeryHigh",
                 "Extreme"]

DATASET_NS = [3000, 5000, 10000]
SEED = 0

# Numeric QI domains (inclusive).
DOMAINS = {
    "age": (18, 80),
    "education_num": (6, 16),
    "hours_per_week": (1, 70),
}

# education_num broad categories (level 3).
EDU_CATS = {"LowEdu": (6, 9), "MidEdu": (10, 13), "HighEdu": (14, 16)}

MAX_LEVEL = 4
LEVELS = list(range(MAX_LEVEL + 1))

# Interval widths per generalization level for the purely interval-based QIs.
AGE_WIDTHS = {1: 5, 2: 10, 3: 20}
HOURS_WIDTHS = {1: 5, 2: 10, 3: 20}
EDU_WIDTHS = {1: 2, 2: 4}

# Parameter grid.
K_VALUES = [5, 10, 20]
L_VALUES = [2, 3, 4]
T_VALUES = [0.2, 0.3, 0.4]
FIXED_K_FOR_L = 10
FIXED_K_FOR_T = 10

# Maximum fraction of rows that may be suppressed (record suppression) so that
# a more balanced generalization can be retained. Set to 0.0 to disable
# suppression entirely (pure global generalization). Equivalence classes that
# still violate the privacy target after generalization are suppressed (their
# rows dropped); a candidate is only feasible if total suppression stays within
# this budget.
MAX_SUPPRESSION_FRAC = 0.05


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Return the project root (parent of the src/ directory)."""
    return Path(__file__).resolve().parent.parent


def raw_dir() -> Path:
    return project_root() / "data" / "raw"


def anonymized_dir() -> Path:
    return project_root() / "data" / "anonymized"


def phase2_dir() -> Path:
    return project_root() / "outputs" / "phase2"


def load_raw_dataset(n: int) -> pd.DataFrame:
    """Load one Phase 1 raw dataset by size, with QIs as plain integers."""
    path = raw_dir() / f"simulated_adult_like_n{n}_seed{SEED}.csv"
    df = pd.read_csv(path)
    for col in QIS:
        df[col] = df[col].astype(int)
    df[SA] = df[SA].astype(str)
    return df


# --------------------------------------------------------------------------- #
# Generalization functions
# --------------------------------------------------------------------------- #
def _interval(value: int, width: int, dmin: int, dmax: int) -> str:
    """Globally consistent fixed-grid interval label, clamped to the domain."""
    lo = (value // width) * width
    hi = lo + width - 1
    lo = max(lo, dmin)
    hi = min(hi, dmax)
    return f"[{lo}-{hi}]"


def generalize_age(value: int, level: int) -> str:
    """Generalize a single age value to the requested level."""
    if level == 0:
        return str(int(value))
    if level == MAX_LEVEL:
        return "*"
    dmin, dmax = DOMAINS["age"]
    return _interval(int(value), AGE_WIDTHS[level], dmin, dmax)


def generalize_education_num(value: int, level: int) -> str:
    """Generalize a single education_num value to the requested level."""
    v = int(value)
    if level == 0:
        return str(v)
    if level == MAX_LEVEL:
        return "*"
    if level == 3:
        for name, (lo, hi) in EDU_CATS.items():
            if lo <= v <= hi:
                return name
        return "*"
    dmin, dmax = DOMAINS["education_num"]
    return _interval(v, EDU_WIDTHS[level], dmin, dmax)


def generalize_hours_per_week(value: int, level: int) -> str:
    """Generalize a single hours_per_week value to the requested level."""
    if level == 0:
        return str(int(value))
    if level == MAX_LEVEL:
        return "*"
    dmin, dmax = DOMAINS["hours_per_week"]
    return _interval(int(value), HOURS_WIDTHS[level], dmin, dmax)


GEN_FUNCS = {
    "age": generalize_age,
    "education_num": generalize_education_num,
    "hours_per_week": generalize_hours_per_week,
}


def generalize_dataframe(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """Return a copy of df with each QI generalized to its level in `levels`.

    `levels` maps QI name -> generalization level (0..4). The sensitive
    attribute is copied through unchanged.
    """
    out = pd.DataFrame(index=df.index)
    for attr in QIS:
        func = GEN_FUNCS[attr]
        lvl = levels[attr]
        out[attr] = df[attr].map(lambda v, f=func, l=lvl: f(v, l)).astype(str)
    out[SA] = df[SA].astype(str).values
    return out


# --------------------------------------------------------------------------- #
# Interval parsing (midpoint / width) for information-loss evaluation
# --------------------------------------------------------------------------- #
def parse_bounds(token: str, attr: str) -> tuple[int, int]:
    """Parse a generalized QI cell into (low, high) integer bounds.

    Handles exact integers, "[lo-hi]" intervals, the education categories,
    and "*" (which maps to the full attribute domain).
    """
    dmin, dmax = DOMAINS[attr]
    if token == "*":
        return dmin, dmax
    if token in EDU_CATS:
        return EDU_CATS[token]
    if token.startswith("[") and token.endswith("]"):
        lo_s, hi_s = token[1:-1].split("-")
        return int(lo_s), int(hi_s)
    return int(token), int(token)


def cell_midpoint(token: str, attr: str) -> float:
    lo, hi = parse_bounds(token, attr)
    return (lo + hi) / 2.0


def cell_ncp(token: str, attr: str) -> float:
    """Normalized certainty penalty for one generalized cell: interval span
    divided by the attribute's full domain span."""
    dmin, dmax = DOMAINS[attr]
    lo, hi = parse_bounds(token, attr)
    span = dmax - dmin
    if span <= 0:
        return 0.0
    return (hi - lo) / span


# --------------------------------------------------------------------------- #
# Information-loss measures
# --------------------------------------------------------------------------- #
def level_information_loss(levels: dict) -> float:
    """Data-independent loss: mean of (level / MAX_LEVEL) across the QIs."""
    return float(np.mean([levels[a] / MAX_LEVEL for a in QIS]))


def numeric_ncp_information_loss(raw_df: pd.DataFrame,
                                 anon_df: pd.DataFrame) -> float:
    """Data-dependent loss: mean per-cell NCP over all QI cells of anon_df.

    raw_df is accepted for signature symmetry/independence; the NCP is fully
    determined by the generalized intervals in anon_df.
    """
    totals = []
    for attr in QIS:
        ncp = anon_df[attr].map(lambda tok, a=attr: cell_ncp(tok, a))
        totals.append(ncp.to_numpy())
    return float(np.mean(np.concatenate(totals))) if totals else 0.0


# --------------------------------------------------------------------------- #
# Equivalence-class statistics
# --------------------------------------------------------------------------- #
def global_sa_distribution(df: pd.DataFrame) -> pd.Series:
    """Normalized SA distribution over the full INCOME_DOMAIN."""
    counts = df[SA].value_counts()
    probs = counts.reindex(INCOME_DOMAIN, fill_value=0).astype(float)
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs


def class_tvd_to_global(class_sa: pd.Series, global_dist: pd.Series) -> float:
    """Total variation distance between a class SA distribution and global.

    `class_sa` may be counts or probabilities indexed by income class.
    """
    p = class_sa.reindex(INCOME_DOMAIN, fill_value=0).astype(float)
    total = p.sum()
    if total > 0:
        p = p / total
    return float(0.5 * (p - global_dist).abs().sum())


def equivalence_class_stats(df: pd.DataFrame,
                            global_dist: pd.Series | None = None) -> dict:
    """Compute equivalence-class statistics for a generalized dataframe.

    Returns sizes, distinct-SA counts and (if a global distribution is given)
    per-class TVD to the global SA distribution.
    """
    grouped = df.groupby(QIS, sort=False)
    sizes = grouped.size()
    distinct = grouped[SA].nunique()

    stats = {
        "num_classes": int(len(sizes)),
        "min_class_size": int(sizes.min()) if len(sizes) else 0,
        "max_class_size": int(sizes.max()) if len(sizes) else 0,
        "min_distinct_sa": int(distinct.min()) if len(distinct) else 0,
        "max_class_tvd": np.nan,
    }

    if global_dist is not None and len(sizes):
        ct = (df.groupby(QIS + [SA], sort=False).size()
                .unstack(SA, fill_value=0)
                .reindex(columns=INCOME_DOMAIN, fill_value=0))
        prob = ct.div(ct.sum(axis=1), axis=0)
        tvd = 0.5 * prob.sub(global_dist, axis=1).abs().sum(axis=1)
        stats["max_class_tvd"] = float(tvd.max())
    return stats


# --------------------------------------------------------------------------- #
# Feasibility checks
# --------------------------------------------------------------------------- #
def check_k(df: pd.DataFrame, k: int) -> bool:
    """True iff every equivalence class has size >= k."""
    sizes = df.groupby(QIS, sort=False).size()
    return bool(len(sizes) and sizes.min() >= k)


def check_l_diversity(df: pd.DataFrame, k: int, ell: int) -> bool:
    """True iff k-anonymous AND every class has >= ell distinct SA values."""
    grouped = df.groupby(QIS, sort=False)
    sizes = grouped.size()
    if not len(sizes) or sizes.min() < k:
        return False
    distinct = grouped[SA].nunique()
    return bool(distinct.min() >= ell)


def check_t_closeness(df: pd.DataFrame, k: int, t: float,
                      global_dist: pd.Series | None = None) -> bool:
    """True iff k-anonymous AND every class TVD to global SA dist is <= t."""
    sizes = df.groupby(QIS, sort=False).size()
    if not len(sizes) or sizes.min() < k:
        return False
    if global_dist is None:
        global_dist = global_sa_distribution(df)
    stats = equivalence_class_stats(df, global_dist)
    return bool(stats["max_class_tvd"] <= t + 1e-12)


# --------------------------------------------------------------------------- #
# Global-recoding search
# --------------------------------------------------------------------------- #
def _precompute_generalizations(df: pd.DataFrame) -> tuple[dict, dict]:
    """Precompute each QI generalized at every level once (search speed-up).

    Returns (gen, ncp_row) where ``gen[attr][level]`` is the generalized column
    and ``ncp_row[attr][level]`` is the per-row NCP array for that column.
    """
    gen = {attr: {} for attr in QIS}
    ncp_row = {attr: {} for attr in QIS}
    for attr in QIS:
        func = GEN_FUNCS[attr]
        for level in LEVELS:
            col = df[attr].map(lambda v, f=func, l=level: f(v, l)).astype(str)
            gen[attr][level] = col
            ncp_row[attr][level] = col.map(
                lambda tok, a=attr: cell_ncp(tok, a)).to_numpy()
    return gen, ncp_row


def _assemble(gen: dict, levels: dict, sa_series: pd.Series) -> pd.DataFrame:
    cand = pd.DataFrame({
        "age": gen["age"][levels["age"]],
        "education_num": gen["education_num"][levels["education_num"]],
        "hours_per_week": gen["hours_per_week"][levels["hours_per_week"]],
    })
    cand[SA] = sa_series.values
    return cand


def _violating_classes(cand: pd.DataFrame, method: str, params: dict,
                       global_dist: pd.Series) -> pd.Series:
    """Boolean Series (indexed by equivalence-class key) marking classes that
    violate the privacy target after generalization.

    A class violates if it is smaller than k, or (l-diversity) has fewer than
    ell distinct SA values, or (t-closeness) has SA-distribution TVD above t.
    """
    grouped = cand.groupby(QIS, sort=False)
    sizes = grouped.size()
    violate = sizes < params["k"]

    if method == "l_diversity":
        distinct = grouped[SA].nunique()
        violate = violate | (distinct < params["ell"])
    elif method == "t_closeness":
        ct = (cand.groupby(QIS + [SA], sort=False).size()
                  .unstack(SA, fill_value=0)
                  .reindex(columns=INCOME_DOMAIN, fill_value=0))
        prob = ct.div(ct.sum(axis=1), axis=0)
        tvd = 0.5 * prob.sub(global_dist, axis=1).abs().sum(axis=1)
        violate = violate | (tvd > params["t"] + 1e-12)
    elif method != "k_anonymity":
        raise ValueError(f"unknown method: {method}")
    return violate


def _evaluate_candidate(cand: pd.DataFrame, levels: dict, ncp_row: dict,
                        method: str, params: dict, global_dist: pd.Series,
                        n: int, budget_rows: int) -> dict | None:
    """Generalize -> suppress violating classes -> score.

    Returns None if suppression would exceed the budget or nothing survives.
    Otherwise returns the kept dataframe, suppression count, surviving class
    count, and the combined information loss used for ranking.
    """
    violate = _violating_classes(cand, method, params, global_dist)
    n_classes_total = int(len(violate))

    if violate.any():
        viol_df = violate.rename("__viol__").reset_index()
        merged = cand.merge(viol_df, on=QIS, how="left", sort=False)
        kept_mask = ~merged["__viol__"].to_numpy()
    else:
        kept_mask = np.ones(len(cand), dtype=bool)

    suppressed = int((~kept_mask).sum())
    if suppressed > budget_rows:
        return None
    num_classes_kept = int(n_classes_total - int(violate.sum()))
    if num_classes_kept <= 0:
        return None

    # Combined information loss over ALL original rows: kept rows incur their
    # generalization NCP; suppressed rows count as fully lost (1.0).
    la, le, lh = levels["age"], levels["education_num"], levels["hours_per_week"]
    per_row_ncp = (ncp_row["age"][la] + ncp_row["education_num"][le]
                   + ncp_row["hours_per_week"][lh]) / 3.0
    combined_loss = float((per_row_ncp[kept_mask].sum() + suppressed) / n)

    return {
        "kept_mask": kept_mask,
        "suppressed": suppressed,
        "num_classes_kept": num_classes_kept,
        "combined_loss": combined_loss,
    }


def _selection_rank(combined_loss: float, levels: dict, suppressed: int,
                    num_classes: int) -> tuple:
    """Lexicographic utility ranking key (smaller is better).

    Order: (1) lowest combined information loss (generalization + suppression),
    (2) fewest fully-suppressed QIs (level 4), (3) lowest maximum generalization
    level, (4) lowest total level sum, (5) fewest suppressed rows, (6) most
    surviving equivalence classes. This favors balanced generalization over
    collapsing whole columns to "*", while keeping suppression small.
    """
    level_vals = [levels[a] for a in QIS]
    num_level4 = sum(1 for v in level_vals if v == MAX_LEVEL)
    return (
        round(combined_loss, 9),
        num_level4,
        max(level_vals),
        sum(level_vals),
        suppressed,
        -num_classes,
    )


def _format_score(rank: tuple) -> str:
    """Human-readable serialization of a selection rank tuple."""
    loss, num_level4, max_lvl, sum_lvl, suppressed, neg_classes = rank
    return (f"loss={loss:.6f};n4={num_level4};maxlvl={max_lvl};"
            f"sumlvl={sum_lvl};suppressed={suppressed};"
            f"nclasses={-neg_classes}")


def find_best_global_recoding(raw_df: pd.DataFrame, method: str,
                              params: dict) -> dict:
    """Exhaustively search all 5^3 level combinations for the best feasible
    recoding under the lexicographic utility ranking, allowing up to
    MAX_SUPPRESSION_FRAC record suppression to retain balanced generalization.

    Returns a dict describing the outcome (feasible / infeasible, chosen
    levels, anonymized dataframe, stats). Infeasible settings are reported
    explicitly rather than failing silently.
    """
    gen, ncp_row = _precompute_generalizations(raw_df)
    global_dist = global_sa_distribution(raw_df)
    sa_series = raw_df[SA]
    n = len(raw_df)
    budget_rows = int(np.floor(MAX_SUPPRESSION_FRAC * n))

    best = None  # (rank, levels, kept_df, suppressed, combined_loss)
    n_feasible = 0
    for la, le, lh in product(LEVELS, LEVELS, LEVELS):
        levels = {"age": la, "education_num": le, "hours_per_week": lh}
        cand = _assemble(gen, levels, sa_series)
        ev = _evaluate_candidate(cand, levels, ncp_row, method, params,
                                 global_dist, n, budget_rows)
        if ev is None:
            continue
        n_feasible += 1
        rank = _selection_rank(ev["combined_loss"], levels, ev["suppressed"],
                               ev["num_classes_kept"])
        if best is None or rank < best[0]:
            kept_df = cand.loc[ev["kept_mask"]].reset_index(drop=True)
            best = (rank, levels, kept_df, ev["suppressed"], ev["combined_loss"])

    if best is not None:
        rank, levels, anon_df, suppressed, combined_loss = best
        stats = equivalence_class_stats(anon_df, global_dist)
        num_level4 = sum(1 for a in QIS if levels[a] == MAX_LEVEL)
        return {
            "feasible": True,
            "infeasible_reason": "",
            "levels": levels,
            "anon_df": anon_df,
            "stats": stats,
            "n_feasible_candidates": n_feasible,
            "numeric_ncp": numeric_ncp_information_loss(raw_df, anon_df),
            "num_fully_suppressed_qis": num_level4,
            "selection_score": _format_score(rank),
            "rows_suppressed": suppressed,
            "cell_suppression_used": False,
            "row_suppression_used": suppressed > 0,
        }

    # ---- Suppression fallback ------------------------------------------
    # No global recoding satisfies the target. This is only reachable when
    # even the fully-generalized ("*","*","*") single class cannot meet the
    # constraint (e.g. ell greater than the number of distinct SA values in
    # the whole dataset). Such a target is genuinely infeasible; we log it.
    reason = _infeasible_reason(raw_df, method, params)
    return {
        "feasible": False,
        "infeasible_reason": reason,
        "levels": {"age": MAX_LEVEL, "education_num": MAX_LEVEL,
                   "hours_per_week": MAX_LEVEL},
        "anon_df": None,
        "stats": None,
        "n_feasible_candidates": 0,
        "numeric_ncp": np.nan,
        "num_fully_suppressed_qis": np.nan,
        "selection_score": "",
        "rows_suppressed": 0,
        "cell_suppression_used": False,
        "row_suppression_used": False,
    }


def _infeasible_reason(raw_df: pd.DataFrame, method: str, params: dict) -> str:
    if method == "l_diversity":
        distinct = raw_df[SA].nunique()
        if params["ell"] > distinct:
            return (f"ell={params['ell']} exceeds {distinct} distinct SA "
                    f"values in the dataset")
    if method == "t_closeness":
        return (f"no recoding (incl. full generalization) keeps every class "
                f"TVD <= t={params['t']}")
    return "no feasible global recoding under target"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _settings():
    """Yield (method, parameter_name, parameter_value, k, ell, t, params)."""
    for k in K_VALUES:
        yield ("k_anonymity", "k", k, k, None, None, {"k": k})
    for ell in L_VALUES:
        yield ("l_diversity", "ell", ell, FIXED_K_FOR_L, ell, None,
               {"k": FIXED_K_FOR_L, "ell": ell})
    for t in T_VALUES:
        yield ("t_closeness", "t", t, FIXED_K_FOR_T, None, t,
               {"k": FIXED_K_FOR_T, "t": t})


def _output_filename(n: int, method: str, params: dict) -> str:
    base = f"simulated_adult_like_n{n}_seed{SEED}"
    if method == "k_anonymity":
        return f"{base}_kanon_k{params['k']}.csv"
    if method == "l_diversity":
        return f"{base}_ldiv_k{params['k']}_l{params['ell']}.csv"
    if method == "t_closeness":
        t_tag = str(params["t"]).replace(".", "p")
        return f"{base}_tclose_k{params['k']}_t{t_tag}.csv"
    raise ValueError(method)


def run_all() -> pd.DataFrame:
    """Run every (dataset x setting) combination and return the runtime table."""
    anonymized_dir().mkdir(parents=True, exist_ok=True)
    phase2_dir().mkdir(parents=True, exist_ok=True)

    rows = []
    for n in DATASET_NS:
        raw_df = load_raw_dataset(n)
        print(f"\n=== dataset n={n} ({len(raw_df)} rows) ===")
        for (method, pname, pval, k, ell, t, params) in _settings():
            start = time.perf_counter()
            result = find_best_global_recoding(raw_df, method, params)
            runtime = time.perf_counter() - start

            levels = result["levels"]
            output_path = ""
            rows_out = 0
            if result["feasible"]:
                anon_df = result["anon_df"]
                out_name = _output_filename(n, method, params)
                out_path = anonymized_dir() / out_name
                anon_df.to_csv(out_path, index=False)
                output_path = str(out_path.relative_to(project_root()))
                rows_out = len(anon_df)
                stats = result["stats"]
                status = "FEASIBLE"
            else:
                stats = {"num_classes": np.nan, "min_class_size": np.nan,
                         "max_class_size": np.nan, "min_distinct_sa": np.nan,
                         "max_class_tvd": np.nan}
                status = f"INFEASIBLE ({result['infeasible_reason']})"

            print(f"  [{status:>10s}] {method:<12s} {pname}={pval}  "
                  f"levels=({levels['age']},{levels['education_num']},"
                  f"{levels['hours_per_week']})  {runtime:.3f}s")

            rows.append({
                "dataset": f"simulated_adult_like_n{n}_seed{SEED}",
                "n": n,
                "method": method,
                "parameter_name": pname,
                "parameter_value": pval,
                "k": k,
                "ell": ell if ell is not None else "",
                "t": t if t is not None else "",
                "feasible": result["feasible"],
                "infeasible_reason": result["infeasible_reason"],
                "output_path": output_path,
                "runtime_seconds": round(runtime, 6),
                "age_level": levels["age"],
                "education_num_level": levels["education_num"],
                "hours_per_week_level": levels["hours_per_week"],
                "num_fully_suppressed_qis": result["num_fully_suppressed_qis"],
                "selection_score": result["selection_score"],
                "num_equivalence_classes": stats["num_classes"],
                "min_class_size": stats["min_class_size"],
                "max_class_size": stats["max_class_size"],
                "min_distinct_sa": stats["min_distinct_sa"],
                "max_class_tvd": stats["max_class_tvd"],
                "rows_in": len(raw_df),
                "rows_out": rows_out,
                "rows_suppressed": result["rows_suppressed"],
                "cell_suppression_used": result["cell_suppression_used"],
                "row_suppression_used": result["row_suppression_used"],
            })

    return pd.DataFrame(rows)


def main() -> None:
    runtime_df = run_all()
    out_csv = phase2_dir() / "runtime_summary.csv"
    runtime_df.to_csv(out_csv, index=False)

    n_feasible = int(runtime_df["feasible"].sum())
    n_total = len(runtime_df)
    print(f"\nWrote {out_csv.relative_to(project_root())}")
    print(f"Feasible settings: {n_feasible}/{n_total}")
    infeasible = runtime_df[~runtime_df["feasible"]]
    if len(infeasible):
        print("Infeasible settings logged:")
        for _, r in infeasible.iterrows():
            print(f"  - {r['dataset']} {r['method']} "
                  f"{r['parameter_name']}={r['parameter_value']}: "
                  f"{r['infeasible_reason']}")


if __name__ == "__main__":
    main()
