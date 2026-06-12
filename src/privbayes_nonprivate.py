#!/usr/bin/env python3
"""Phase 3 — NON-PRIVATE simplified PrivBayes synthetic data generation.

This is a **non-private utility upper baseline** (no differential privacy /
noise of any kind). It learns a Bayesian network with GreedyBayes using exact
empirical mutual information, fits exact conditional probability tables, and
samples synthetic datasets. Phase 4 will later add differential privacy.

Pipeline per (n, r, seed):
  1. discretize the raw data into the PrivBayes discrete domain;
  2. GreedyBayes structure learning (max parent degree r, seeded random root);
  3. fit exact CPTs P(X | Parents) (+ marginal fallback P(X) per attribute);
  4. ancestral sampling of n synthetic records;
  5. decode bins -> rounded midpoints, restore canonical column order, save CSV.

Parameter grid: n in {3000, 5000, 10000} x r in {0, 1, 2} x seed in {0..4}
              = 45 synthetic datasets.

Run from the project root with:

    python src/privbayes_nonprivate.py
"""

from __future__ import annotations

import json
import time
from itertools import combinations

import numpy as np
import pandas as pd

import privbayes_utils as U


# --------------------------------------------------------------------------- #
# GreedyBayes structure learning
# --------------------------------------------------------------------------- #
def greedy_bayes(disc_df: pd.DataFrame, r: int, rng: np.random.Generator
                 ) -> tuple[list[tuple[str, tuple[str, ...]]], list[dict]]:
    """Simplified non-private GreedyBayes.

    Returns
    -------
    network : ordered list of (attribute, parents) pairs in BN order.
    trace   : per-step diagnostics (chosen MI + best alternatives) for logging.

    Algorithm
    ---------
    * Randomly pick the first attribute X1 (seeded) with empty parents.
    * Repeatedly, over every unselected X and every parent set Pi drawn from the
      already-selected attributes V with |Pi| = min(r, |V|), compute I(X; Pi)
      and add the (X, Pi) with maximum MI.
    * Deterministic tie-break: (-MI, attribute_name, parent_tuple) so equal-MI
      candidates resolve reproducibly. This makes r=0 (all MI == 0) fully
      deterministic given the random root.
    """
    attrs = list(disc_df.columns)
    remaining = set(attrs)

    first = attrs[int(rng.integers(len(attrs)))]
    network: list[tuple[str, tuple[str, ...]]] = [(first, ())]
    selected: list[str] = [first]
    remaining.remove(first)
    trace: list[dict] = [{"step": 0, "attribute": first, "parents": [],
                          "mutual_information_nats": 0.0,
                          "note": "random root"}]

    step = 1
    while remaining:
        deg = min(r, len(selected))
        candidates: list[tuple[float, str, tuple[str, ...]]] = []
        for x in remaining:
            if deg == 0:
                parent_sets = [()]
            else:
                parent_sets = [tuple(sorted(p))
                               for p in combinations(selected, deg)]
            for parents in parent_sets:
                mi = U.mutual_information(disc_df, x, parents)
                # sort key: larger MI first, then attribute name, then parents
                candidates.append((mi, x, parents))

        # max MI, tie-break by (name, parent tuple) ascending
        best = min(candidates, key=lambda c: (-c[0], c[1], c[2]))
        best_mi, best_x, best_parents = best
        network.append((best_x, best_parents))
        selected.append(best_x)
        remaining.remove(best_x)
        trace.append({"step": step, "attribute": best_x,
                      "parents": list(best_parents),
                      "mutual_information_nats": round(best_mi, 6),
                      "degree": len(best_parents)})
        step += 1

    return network, trace


# --------------------------------------------------------------------------- #
# CPT fitting (exact, non-private)
# --------------------------------------------------------------------------- #
def fit_marginal(disc_df: pd.DataFrame, attr: str) -> dict:
    """Exact marginal P(X) as {value: prob}. Used directly for roots and as the
    fallback for unseen / zero-sum parent configurations during sampling."""
    counts = disc_df[attr].astype(str).value_counts()
    probs = counts / counts.sum()
    return probs.to_dict()


def fit_cpt(disc_df: pd.DataFrame, attr: str, parents: tuple[str, ...]) -> dict:
    """Exact CPT P(X | Parents) as {parent_config_tuple: {value: prob}}.

    parent_config_tuple is the tuple of parent values in ``parents`` order.
    For an empty parent set the single key () maps to the marginal P(X).
    """
    if not parents:
        return {(): fit_marginal(disc_df, attr)}

    cpt: dict = {}
    grp = disc_df.groupby(list(parents))[attr].value_counts()
    # grp is a Series indexed by (*parent_values, attr_value)
    for idx, cnt in grp.items():
        *pvals, xval = idx if isinstance(idx, tuple) else (idx,)
        key = tuple(str(v) for v in pvals)
        cpt.setdefault(key, {})[str(xval)] = int(cnt)
    # normalise each parent-config row to a probability distribution
    for key, row in cpt.items():
        total = sum(row.values())
        for xval in row:
            row[xval] = row[xval] / total
    return cpt


def fit_network(disc_df: pd.DataFrame,
                network: list[tuple[str, tuple[str, ...]]]
                ) -> tuple[dict, dict]:
    """Fit CPTs and marginal fallbacks for every node in the network."""
    cpts = {attr: fit_cpt(disc_df, attr, parents) for attr, parents in network}
    fallbacks = {attr: fit_marginal(disc_df, attr) for attr, _ in network}
    return cpts, fallbacks


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def _sample_from(dist: dict, rng: np.random.Generator) -> str:
    vals = list(dist.keys())
    probs = np.asarray([dist[v] for v in vals], dtype=float)
    probs = probs / probs.sum()  # guard against tiny float drift
    return vals[int(rng.choice(len(vals), p=probs))]


def sample(network: list[tuple[str, tuple[str, ...]]], cpts: dict,
           fallbacks: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Ancestral sampling of ``n`` discrete records, then column reorder.

    For each attribute we look up its parent configuration among already-sampled
    values; an unseen / zero-sum row falls back to the marginal P(X). The result
    is a discrete frame (bin labels) in canonical ATTRS column order.
    """
    columns = [attr for attr, _ in network]
    records = {attr: [] for attr in columns}

    for _ in range(n):
        row: dict = {}
        for attr, parents in network:
            cpt = cpts[attr]
            key = tuple(row[p] for p in parents)
            dist = cpt.get(key)
            if not dist or sum(dist.values()) <= 0.0:
                dist = fallbacks[attr]  # marginal fallback (unseen config)
            row[attr] = _sample_from(dist, rng)
            records[attr].append(row[attr])

    disc = pd.DataFrame(records)[columns]
    return disc[U.ATTRS]  # canonical schema order


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
# Note: cpt_validation_summary.csv is produced INDEPENDENTLY by
# src/validate_privbayes_nonprivate.py (the validator re-fits the CPTs and
# checks finiteness / non-negativity / row sums). The generator only records the
# learned structure, runtime, and pairwise MI here.
def main() -> None:
    U.synthetic_dir().mkdir(parents=True, exist_ok=True)
    U.phase3_dir().mkdir(parents=True, exist_ok=True)

    runtime_rows: list[dict] = []
    network_rows: list[dict] = []
    mi_rows: list[dict] = []
    learned: dict = {}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 3 — NON-PRIVATE SIMPLIFIED PRIVBAYES (utility upper baseline)")
    emit("NOTE: no differential privacy is applied; this is NOT a privacy "
         "method.")
    emit("=" * 74)
    emit()

    raw_cache: dict = {}
    for n in U.DATASET_SIZES:
        if n not in raw_cache:
            raw_cache[n] = U.load_raw(n)
        raw = raw_cache[n]

        for r in U.R_VALUES:
            for seed in U.SEEDS:
                rng = np.random.default_rng(seed)

                t0 = time.perf_counter()
                disc = U.discretize(raw)
                t1 = time.perf_counter()

                network, trace = greedy_bayes(disc, r, rng)
                t2 = time.perf_counter()

                cpts, fallbacks = fit_network(disc, network)
                t3 = time.perf_counter()

                synth_disc = sample(network, cpts, fallbacks, len(raw), rng)
                t4 = time.perf_counter()

                synth = U.decode(synth_disc)
                out_path = U.synthetic_path(n, r, seed)
                synth.to_csv(out_path, index=False)

                disc_t = t1 - t0
                struct_t = t2 - t1
                cpt_t = t3 - t2
                sample_t = t4 - t3
                total_t = t4 - t0

                runtime_rows.append({
                    "n": n, "r": r, "seed": seed,
                    "discretization_time_s": round(disc_t, 6),
                    "structure_learning_time_s": round(struct_t, 6),
                    "cpt_fitting_time_s": round(cpt_t, 6),
                    "sampling_time_s": round(sample_t, 6),
                    "total_time_s": round(total_t, 6),
                    "output_path": str(out_path.relative_to(U.project_root())),
                })

                # network summary (one row per node)
                order = [a for a, _ in network]
                for pos, (attr, parents) in enumerate(network):
                    network_rows.append({
                        "n": n, "r": r, "seed": seed, "order_index": pos,
                        "attribute": attr, "parents": "|".join(parents),
                        "degree": len(parents),
                    })

                # pairwise MI on the discretized RAW data (depends only on n)
                if seed == 0 and r == 0:
                    for (a, b), mi in U.pairwise_mutual_information(disc).items():
                        mi_rows.append({"n": n, "var_a": a, "var_b": b,
                                        "raw_mi_nats": round(mi, 6)})

                learned[f"n{n}_r{r}_seed{seed}"] = {
                    "n": n, "r": r, "seed": seed,
                    "order": order,
                    "network": [{"attribute": a, "parents": list(p)}
                                for a, p in network],
                    "trace": trace,
                }

                emit(f"n={n:<5d} r={r} seed={seed} | "
                     f"order={'>'.join(order)} | "
                     f"total={total_t:.3f}s")

    # ---- Save outputs ---------------------------------------------------- #
    pd.DataFrame(runtime_rows).to_csv(
        U.phase3_dir() / "runtime_summary.csv", index=False)
    pd.DataFrame(network_rows).to_csv(
        U.phase3_dir() / "network_summary.csv", index=False)
    pd.DataFrame(mi_rows).to_csv(
        U.phase3_dir() / "pairwise_mi_summary.csv", index=False)
    with open(U.phase3_dir() / "learned_networks.json", "w",
              encoding="utf-8") as fh:
        json.dump(learned, fh, indent=2)

    emit()
    emit("-" * 74)
    emit(f"Generated {len(runtime_rows)} synthetic datasets "
         f"(expected 3 x 3 x 5 = 45).")
    emit("Most-frequently learned edges (attribute <- parents), by frequency:")
    edge_counts: dict = {}
    for row in network_rows:
        if row["parents"]:
            key = f"{row['attribute']} <- {row['parents']}"
            edge_counts[key] = edge_counts.get(key, 0) + 1
    for key, cnt in sorted(edge_counts.items(), key=lambda kv: -kv[1])[:10]:
        emit(f"    {cnt:3d}x  {key}")
    emit()
    emit("Saved:")
    for name in ("runtime_summary.csv", "network_summary.csv",
                 "learned_networks.json", "pairwise_mi_summary.csv"):
        emit(f"  - {(U.phase3_dir() / name).relative_to(U.project_root())}")
    emit(f"  - {U.synthetic_dir().relative_to(U.project_root())}/ "
         f"({len(runtime_rows)} CSV files)")

    with open(U.phase3_dir() / "phase3_summary.txt", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
