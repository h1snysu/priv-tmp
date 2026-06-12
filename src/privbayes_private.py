#!/usr/bin/env python3
"""Phase 4 — PRIVATE simplified PrivBayes synthetic data generation.

This is a **simplified differentially-private PrivBayes-style** generator. It

  1. privately learns the Bayesian-network structure with the EXPONENTIAL
     mechanism (quality function = empirical mutual information, in nats);
  2. privately learns noisy conditional probability tables with the LAPLACE
     mechanism over full-domain joint count tables;
  3. samples synthetic data using ONLY the private network + noisy CPTs +
     fallback distributions + fixed public domains (never the raw data).

IMPORTANT — scope / honesty
---------------------------
This is a *simplified* private PrivBayes, NOT the full algorithm from
Zhang et al. ("PrivBayes", SIGMOD 2014 / TODS 2017):

  * It scores candidate edges with the DIRECT mutual information I(X; Pi) and a
    conservative MI sensitivity bound, rather than the paper's improved
    low-sensitivity surrogate score F.
  * Privacy budget is composed with BASIC sequential composition (no advanced
    composition / zCDP), under the replace-one-tuple neighbouring relation.

The privacy accounting below is exact for THIS mechanism; we do not claim it
matches the paper's privacy/utility. We never reuse the Phase 3 (non-private)
learned networks — that would leak raw-data structure.

Privacy accounting (d = 4 attributes, replace-one-tuple neighbours)
-------------------------------------------------------------------
  epsilon_total = epsilon_structure + epsilon_parameters
    r > 0:  epsilon_structure = eps/2 , epsilon_parameters = eps/2
    r = 0:  epsilon_structure = 0     , epsilon_parameters = eps   (structure
            is data-independent and spends no budget)

  Exponential mechanism (structure), per step:
    epsilon_step = epsilon_structure / (d - 1)        # d-1 = 3 EM steps
    Pr(candidate) proportional to exp( epsilon_step * I(X;Pi) / (2 * Delta_I) )
    Delta_I(n) = (2/n) ln((n+1)/2) + ((n-1)/n) ln((n+1)/(n-1))   [nats]
  The root node is chosen uniformly at random (data-independent, 0 budget).

  Laplace mechanism (parameters), one noisy joint table per attribute:
    epsilon_cpt   = epsilon_parameters / d
    hist sensitivity = 2 (replace-one changes a count histogram by L1 = 2)
    laplace_scale = 2 / epsilon_cpt

  Composition check: (d-1)*epsilon_step + d*epsilon_cpt
                   = epsilon_structure + epsilon_parameters = epsilon_total.

Run from the project root with:

    python src/privbayes_private.py
"""

from __future__ import annotations

import json
import time
import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd

import privbayes_utils as U


D_ATTRS = len(U.ATTRS)  # 4


# --------------------------------------------------------------------------- #
# Privacy primitives
# --------------------------------------------------------------------------- #
def mi_sensitivity_bound(n: int) -> float:
    """Conservative sensitivity of the mutual-information quality function in
    NATS, under the replace-one-tuple neighbouring relation.

        Delta_I(n) = (2/n) ln((n+1)/2) + ((n-1)/n) ln((n+1)/(n-1))

    This is the PrivBayes (Zhang et al., 2014) *first-cut* MI sensitivity — the
    direct-MI bound, NOT the paper's lower-sensitivity surrogate F. Documented
    here so nothing is silently hardcoded.
    """
    n = float(n)
    return (2.0 / n) * np.log((n + 1.0) / 2.0) \
        + ((n - 1.0) / n) * np.log((n + 1.0) / (n - 1.0))


def exponential_mechanism(scores: np.ndarray, eps_step: float,
                          sensitivity: float, rng: np.random.Generator
                          ) -> tuple[int, bool]:
    """Sample one index proportional to exp(eps_step * score / (2*sensitivity)).

    Numerically stable: log-weights minus their max before exponentiating. If
    every weight underflows / is invalid, fall back to UNIFORM sampling and
    return used_uniform=True so the caller can log a warning.
    """
    if eps_step <= 0.0 or sensitivity <= 0.0:
        return int(rng.integers(len(scores))), True

    log_w = (eps_step * scores) / (2.0 * sensitivity)
    log_w = log_w - np.max(log_w)
    weights = np.exp(log_w)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.integers(len(scores))), True
    probs = weights / total
    return int(rng.choice(len(scores), p=probs)), False


# --------------------------------------------------------------------------- #
# Private structure learning (private GreedyBayes)
# --------------------------------------------------------------------------- #
def private_greedy_bayes(disc_df: pd.DataFrame, r: int, eps_structure: float,
                         n: int, rng: np.random.Generator
                         ) -> tuple[list[tuple[str, tuple[str, ...]]], list[dict]]:
    """Exponential-mechanism GreedyBayes.

    Returns (network, trace). The root is a uniformly-random attribute (0 budget,
    data-independent). Each remaining attribute is added via one exponential-
    mechanism draw over all (X, parent-set) candidates with |Pi| = min(r, |V|).

    For r = 0 there is no MI-based selection (all empty parent sets, MI == 0):
    we use a seed-controlled random order, which is data-independent and spends
    no privacy budget.
    """
    attrs = list(disc_df.columns)
    remaining = set(attrs)

    delta_i = mi_sensitivity_bound(n)
    eps_step = eps_structure / (D_ATTRS - 1) if eps_structure > 0 else 0.0

    # --- root: uniformly random, data-independent, 0 budget ---
    first = attrs[int(rng.integers(len(attrs)))]
    network: list[tuple[str, tuple[str, ...]]] = [(first, ())]
    selected: list[str] = [first]
    remaining.remove(first)
    trace: list[dict] = [{"step": 0, "attribute": first, "parents": [],
                          "mechanism": "random_root", "epsilon_step": 0.0,
                          "used_uniform_fallback": False}]

    step = 1
    while remaining:
        deg = min(r, len(selected))

        if deg == 0:
            # r = 0 (or no selected parents yet): data-independent random pick.
            rem_sorted = sorted(remaining)
            pick = rem_sorted[int(rng.integers(len(rem_sorted)))]
            network.append((pick, ()))
            selected.append(pick)
            remaining.remove(pick)
            trace.append({"step": step, "attribute": pick, "parents": [],
                          "mechanism": "random_order", "epsilon_step": 0.0,
                          "used_uniform_fallback": False})
            step += 1
            continue

        # Build candidate (X, parent-set) pairs and their MI scores.
        cand: list[tuple[str, tuple[str, ...]]] = []
        scores: list[float] = []
        for x in sorted(remaining):
            for parents in (tuple(sorted(p))
                            for p in combinations(selected, deg)):
                cand.append((x, parents))
                scores.append(U.mutual_information(disc_df, x, parents))

        idx, used_uniform = exponential_mechanism(
            np.asarray(scores, dtype=float), eps_step, delta_i, rng)
        pick_x, pick_parents = cand[idx]
        network.append((pick_x, pick_parents))
        selected.append(pick_x)
        remaining.remove(pick_x)
        # NOTE: we deliberately do NOT persist the selected candidate's raw MI
        # score. The exponential mechanism privatizes the *choice* of candidate,
        # not the exact data-dependent score; saving the raw score would be an
        # additional, unaccounted release. delta_i depends only on the public n.
        trace.append({"step": step, "attribute": pick_x,
                      "parents": list(pick_parents),
                      "mechanism": "exponential", "epsilon_step": eps_step,
                      "delta_i": round(float(delta_i), 8),
                      "used_uniform_fallback": used_uniform})
        if used_uniform:
            warnings.warn(f"exponential mechanism underflow at step {step}; "
                          f"used uniform fallback (eps_step={eps_step:.4g})")
        step += 1

    return network, trace


# --------------------------------------------------------------------------- #
# Private parameter learning (noisy CPTs via Laplace on full-domain counts)
# --------------------------------------------------------------------------- #
def noisy_cpt(disc_df: pd.DataFrame, attr: str, parents: tuple[str, ...],
              eps_cpt: float, rng: np.random.Generator) -> dict:
    """Noisy CPT P*(X | Parents) over the FULL public Cartesian domain.

    Steps: materialise the full joint count table over (X, Parents) including
    every zero-count cell -> add Laplace(scale = 2/eps_cpt) to every cell ->
    clip negatives to 0 -> normalise each parent-row to probabilities. A parent
    row that is all-zero after clipping falls back to the noisy marginal P*(X)
    (the noisy joint summed over parents), or uniform if that is also zero.

    Returns
    -------
    {
      "x_domain":  list[str],                       # column order of probs
      "table":     {parent_config_tuple: np.ndarray},  # prob vector per row
      "fallback":  np.ndarray,                       # noisy marginal P*(X)
    }
    """
    x_dom = U.FIXED_DOMAINS[attr]
    x_index = {v: i for i, v in enumerate(x_dom)}
    scale = 2.0 / eps_cpt

    if not parents:
        counts = np.zeros(len(x_dom), dtype=float)
        vc = disc_df[attr].astype(str).value_counts()
        for v, c in vc.items():
            if v in x_index:
                counts[x_index[v]] += c
        noisy = np.clip(counts + rng.laplace(0.0, scale, size=counts.shape),
                        0.0, None)
        total = noisy.sum()
        prob = noisy / total if total > 0 else np.full(len(x_dom),
                                                       1.0 / len(x_dom))
        return {"x_domain": list(x_dom), "table": {(): prob}, "fallback": prob}

    parent_doms = [U.FIXED_DOMAINS[p] for p in parents]
    # observed counts: (parent_config, x) -> count
    obs: dict = {}
    grp = disc_df.groupby(list(parents))[attr].value_counts()
    for idx, cnt in grp.items():
        *pvals, xval = idx if isinstance(idx, tuple) else (idx,)
        obs[(tuple(str(v) for v in pvals), str(xval))] = int(cnt)

    table: dict = {}
    marginal_noisy = np.zeros(len(x_dom), dtype=float)
    for pconf in product(*parent_doms):
        counts = np.zeros(len(x_dom), dtype=float)
        for xi, xv in enumerate(x_dom):
            counts[xi] = obs.get((pconf, xv), 0)
        noisy = np.clip(counts + rng.laplace(0.0, scale, size=counts.shape),
                        0.0, None)
        marginal_noisy += noisy
        total = noisy.sum()
        table[pconf] = (noisy / total) if total > 0 else None  # fill later

    mtot = marginal_noisy.sum()
    fallback = (marginal_noisy / mtot) if mtot > 0 \
        else np.full(len(x_dom), 1.0 / len(x_dom))

    # zero-sum rows -> noisy marginal fallback
    for pconf, prob in table.items():
        if prob is None:
            table[pconf] = fallback

    return {"x_domain": list(x_dom), "table": table, "fallback": fallback}


def build_private_model(disc_df: pd.DataFrame,
                        network: list[tuple[str, tuple[str, ...]]],
                        eps_cpt: float, rng: np.random.Generator) -> dict:
    """Fit a noisy CPT for every node. The returned model is self-contained:
    sampling needs only this object + fixed domains (no raw data)."""
    cpts = {attr: noisy_cpt(disc_df, attr, parents, eps_cpt, rng)
            for attr, parents in network}
    return {"network": network, "cpts": cpts}


# --------------------------------------------------------------------------- #
# Sampling (vectorised ancestral sampling; raw data is NOT accessed)
# --------------------------------------------------------------------------- #
def sample_private(model: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Vectorised ancestral sampling from the private model -> discrete frame in
    canonical ATTRS column order."""
    network = model["network"]
    cpts = model["cpts"]
    out: dict[str, np.ndarray] = {}

    for attr, parents in network:
        node = cpts[attr]
        x_dom = np.asarray(node["x_domain"], dtype=object)
        col = np.empty(n, dtype=object)

        if not parents:
            prob = node["table"][()]
            picks = rng.choice(len(x_dom), size=n, p=prob)
            col = x_dom[picks]
        else:
            parent_cols = [out[p] for p in parents]
            # group rows by parent configuration, sample each group in a batch
            keys = list(zip(*parent_cols))
            uniq, inverse = np.unique(
                np.array([str(k) for k in keys]), return_inverse=True)
            # map each unique string key back to a tuple to look up the CPT row
            rep_tuple: dict[str, tuple] = {}
            for k in keys:
                rep_tuple.setdefault(str(k), tuple(str(v) for v in k))
            for u_i, u_key in enumerate(uniq):
                mask = inverse == u_i
                pconf = rep_tuple[u_key]
                prob = node["table"].get(pconf, node["fallback"])
                m = int(mask.sum())
                picks = rng.choice(len(x_dom), size=m, p=prob)
                col[mask] = x_dom[picks]
        out[attr] = col

    disc = pd.DataFrame({a: out[a] for a, _ in network})
    return disc[U.ATTRS]


# --------------------------------------------------------------------------- #
# Budget bookkeeping
# --------------------------------------------------------------------------- #
def budget_split(eps: float, r: int) -> dict:
    if r > 0:
        eps_struct = eps / 2.0
        eps_param = eps / 2.0
    else:
        eps_struct = 0.0
        eps_param = eps
    num_steps = (D_ATTRS - 1) if r > 0 else 0
    eps_step = (eps_struct / (D_ATTRS - 1)) if r > 0 else 0.0
    eps_cpt = eps_param / D_ATTRS
    return {
        "epsilon_total": eps, "epsilon_structure": eps_struct,
        "epsilon_parameters": eps_param, "num_structure_steps": num_steps,
        "epsilon_step": eps_step, "epsilon_cpt": eps_cpt,
        "hist_sensitivity": 2.0, "laplace_scale": 2.0 / eps_cpt,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    U.private_synthetic_dir().mkdir(parents=True, exist_ok=True)
    U.phase4_dir().mkdir(parents=True, exist_ok=True)

    runtime_rows: list[dict] = []
    network_rows: list[dict] = []
    acct_rows: list[dict] = []
    learned: dict = {}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 74)
    emit("PHASE 4 — SIMPLIFIED *PRIVATE* PRIVBAYES")
    emit("Exponential-mechanism structure + Laplace noisy CPTs. Simplified: "
         "direct MI")
    emit("score (not surrogate F) and basic sequential composition. NOT the "
         "full paper.")
    emit("=" * 74)
    emit()
    emit("SCOPE / NON-OVERCLAIM ----------------------------------------------")
    emit("We implement a SIMPLIFIED private PrivBayes-style mechanism. Unlike")
    emit("the full PrivBayes paper, we use direct mutual information as the")
    emit("exponential-mechanism quality score instead of the paper's lower-")
    emit("sensitivity surrogate function F, we use basic sequential")
    emit("composition, and a fixed parent-degree grid r in {0,1,2}.")
    emit("")
    emit("ARTIFACT CLASSIFICATION --------------------------------------------")
    emit("PRIVATE RELEASE (consumes the epsilon budget below):")
    emit("  - synthetic datasets under data/synthetic/private/")
    emit("  - the private network STRUCTURE in learned_networks_private.json")
    emit("    (the exponential-mechanism output: chosen attribute + parents).")
    emit("  We deliberately do NOT persist the selected candidates' raw MI")
    emit("  scores: the exponential mechanism privatizes the CHOICE, not the")
    emit("  exact data-dependent score, so saving it would be an unaccounted")
    emit("  release. privacy_accounting_summary.csv documents the full budget.")
    emit("")
    emit("TRUSTED CURATOR-SIDE EVALUATION (NOT part of the private release):")
    emit("  - marginal_tvd_summary.csv, pairwise_mi_summary.csv (incl. RAW MI),")
    emit("    and all raw-vs-synthetic utility numbers are computed with full")
    emit("    raw-data access for analysis only. They must NOT be published as")
    emit("    if they were differentially private outputs.")
    emit("=" * 74)
    emit()

    raw_cache: dict = {}
    for n in U.DATASET_SIZES:
        if n not in raw_cache:
            raw_cache[n] = U.load_raw(n)
        raw = raw_cache[n]
        delta_i = mi_sensitivity_bound(n)

        # raw discretised view is reused across r/eps/seed for this n
        disc_n = U.discretize(raw)

        for r in U.R_VALUES:
            for eps in U.EPSILONS:
                bud = budget_split(eps, r)
                for seed in U.SEEDS:
                    rng = np.random.default_rng(seed)

                    t0 = time.perf_counter()
                    disc = disc_n  # discretization is fixed/public
                    t1 = time.perf_counter()

                    network, trace = private_greedy_bayes(
                        disc, r, bud["epsilon_structure"], n, rng)
                    t2 = time.perf_counter()

                    model = build_private_model(
                        disc, network, bud["epsilon_cpt"], rng)
                    t3 = time.perf_counter()

                    synth_disc = sample_private(model, len(raw), rng)
                    t4 = time.perf_counter()

                    synth = U.decode(synth_disc)
                    out_path = U.private_synthetic_path(n, r, eps, seed)
                    synth.to_csv(out_path, index=False)

                    runtime_rows.append({
                        "n": n, "r": r, "epsilon": eps, "seed": seed,
                        "discretization_time_s": round(t1 - t0, 6),
                        "structure_learning_time_s": round(t2 - t1, 6),
                        "cpt_fitting_time_s": round(t3 - t2, 6),
                        "sampling_time_s": round(t4 - t3, 6),
                        "total_time_s": round(t4 - t0, 6),
                        "output_path": str(
                            out_path.relative_to(U.project_root())),
                    })

                    order = [a for a, _ in network]
                    for pos, (attr, parents) in enumerate(network):
                        network_rows.append({
                            "n": n, "r": r, "epsilon": eps, "seed": seed,
                            "order_index": pos, "attribute": attr,
                            "parents": "|".join(parents), "degree": len(parents),
                        })

                    acct_rows.append({
                        "n": n, "r": r, "epsilon": eps, "seed": seed,
                        **{k: round(v, 8) for k, v in bud.items()},
                        "mi_sensitivity_bound": round(float(delta_i), 8),
                        "neighbor_model": "replace_one_tuple",
                    })

                    key = (f"n{n}_r{r}_{U.eps_tag(eps)}_seed{seed}")
                    learned[key] = {
                        "n": n, "r": r, "epsilon": eps, "seed": seed,
                        "order": order,
                        "network": [{"attribute": a, "parents": list(p)}
                                    for a, p in network],
                        "trace": trace,
                        "source": "phase4_private_exponential_mechanism",
                    }

        emit(f"n={n}: generated {len(U.R_VALUES) * len(U.EPSILONS) * len(U.SEEDS)} "
             f"private datasets (Delta_I={delta_i:.6g} nats)")

    # ---- Save outputs ---------------------------------------------------- #
    pd.DataFrame(runtime_rows).to_csv(
        U.phase4_dir() / "runtime_summary.csv", index=False)
    pd.DataFrame(network_rows).to_csv(
        U.phase4_dir() / "network_summary.csv", index=False)
    pd.DataFrame(acct_rows).to_csv(
        U.phase4_dir() / "privacy_accounting_summary.csv", index=False)
    with open(U.phase4_dir() / "learned_networks_private.json", "w",
              encoding="utf-8") as fh:
        json.dump(learned, fh, indent=2)

    emit()
    emit("-" * 74)
    emit(f"Generated {len(runtime_rows)} private synthetic datasets "
         f"(expected 3 x 3 x 4 x 5 = 180).")
    emit("Privacy: simplified private PrivBayes (direct-MI exponential mechanism"
         " + Laplace CPTs,")
    emit("basic sequential composition, replace-one-tuple). NOT a privacy proof"
         " of the full paper.")
    emit()
    emit("Saved:")
    for name in ("runtime_summary.csv", "network_summary.csv",
                 "privacy_accounting_summary.csv",
                 "learned_networks_private.json"):
        emit(f"  - {(U.phase4_dir() / name).relative_to(U.project_root())}")
    emit(f"  - {U.private_synthetic_dir().relative_to(U.project_root())}/ "
         f"({len(runtime_rows)} CSV files)")

    with open(U.phase4_dir() / "phase4_summary.txt", "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
