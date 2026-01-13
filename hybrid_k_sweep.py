"""
Run hybrid composition with varying top-k across all setups/specs/seeds.

Outputs a CSV with average reward and time (including decomposition time) per k.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from collections import defaultdict

import numpy as np

# Ensure local imports work when executed outside the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DOUBLE_POLICIES,
    GRIDWORLD_AVAILABLE_ACTIONS,
    TRIPLE_POLICIES,
    TRIVIAL_POLICIES,
)
from full_experiment import (
    DOUBLE_EXPERIMENTS,
    TRIPLE_EXPERIMENTS,
    TRIVIAL_EXPERIMENTS,
    combine_q_tables_list,
    decompose_query_with_retry,
    embedding_from_qtable,
    greedy_eval,
    infer_grid_and_canonical,
    init_env_from_run,
    load_canonical_states,
    load_q_table_from_metadata,
    normalize_seed,
)
from search_faiss_policies import PolicyRetriever


def find_latest_run_dir(base_dir: str, spec: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    candidates = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(f"{spec}_")
    ]
    if not candidates:
        return None
    return os.path.join(base_dir, sorted(candidates)[-1])


def seed_dir_exists(states_root: str, setup: str, seed: str) -> bool:
    return os.path.isdir(os.path.join(states_root, setup, f"seed_{seed}"))


def group_candidates_for_subqueries(
    retriever: PolicyRetriever,
    sub_queries: list[str],
    seed: str,
    similarity_threshold: float = 0.7,
    search_k: int = 5,
) -> tuple[list[list[dict]], float]:
    grouped = []
    total_search_time = 0.0

    for sq in sub_queries:
        result_dict, timing = retriever.vdb.search_similar_policies(
            sq, k=search_k, policy_seed=seed
        )
        if isinstance(timing, dict):
            total_search_time += timing.get("total_time", 0.0)
        else:
            total_search_time += timing

        results = result_dict.get("results", [])
        results = [r for r in results if r.get("score", 0) > similarity_threshold]
        scored = []
        for r in results:
            emb = r.get("policy_embedding")
            if emb is None:
                continue
            if isinstance(emb, list):
                emb = np.array(emb)
            if retriever.regressor_model:
                expected = getattr(retriever.regressor_model, "n_features_in_", None)
                if expected is not None and emb.shape[0] != expected:
                    continue
                pred = float(retriever.regressor_model.predict(emb.reshape(1, -1))[0])
            else:
                pred = 0.0
            r["regressor_score"] = pred
            scored.append(r)

        scored = sorted(
            scored, key=lambda x: x.get("regressor_score", -1), reverse=True
        )
        grouped.append(scored)

    return grouped, total_search_time


def best_hybrid_from_groups(
    retriever: PolicyRetriever,
    grouped_candidates: list[list[dict]],
    top_k: int,
    canonical_states,
    env,
    seed: str,
) -> tuple[np.ndarray | None, float]:
    start = time.time()

    top_groups = [g[:top_k] for g in grouped_candidates]
    if any(len(g) == 0 for g in top_groups):
        return None, time.time() - start

    best_pred = -float("inf")
    best_q = None

    for combo in itertools.product(*top_groups):
        q_tables = []
        missing = []
        for p in combo:
            q = load_q_table_from_metadata(p)
            if q is None:
                missing.append(p.get("policy_name", "unknown"))
                continue
            q_tables.append(q)
        if missing or len(q_tables) != len(combo):
            continue

        try:
            seed_val = int(seed)
        except (TypeError, ValueError):
            seed_val = int(combo[0].get("policy_seed", 0))
        expected_states = env.grid_length * env.grid_width
        if any(q.shape[0] != expected_states for q in q_tables):
            continue

        q_combined = combine_q_tables_list(q_tables)
        embedding = embedding_from_qtable(
            env, q_combined, canonical_states, seed_val
        )
        pred = (
            float(
                retriever.regressor_model.predict(
                    np.array(embedding).reshape(1, -1)
                )[0]
            )
            if retriever.regressor_model
            else 0.0
        )
        if pred > best_pred:
            best_pred = pred
            best_q = q_combined

    elapsed = time.time() - start
    return best_q, elapsed


def build_experiment_groups():
    return [
        ("trivial", TRIVIAL_EXPERIMENTS, TRIVIAL_POLICIES),
        ("double", DOUBLE_EXPERIMENTS, DOUBLE_POLICIES),
        ("triple", TRIPLE_EXPERIMENTS, TRIPLE_POLICIES),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep hybrid top-k across all setups/specs/seeds."
    )
    parser.add_argument(
        "--state-runs-dir",
        type=str,
        default="state_runs",
        help="Directory containing X1/X5/X10 runs (e.g., state_runs)",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=["X1", "X5", "X10"],
        help="Spec labels to include (e.g., X1 X5 X10)",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Optional seed list override",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/hybrid_k_sweep.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum k to sweep (default: 2)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Optional max k override (default: global min candidates - 1)",
    )
    parser.add_argument(
        "--search-k",
        type=int,
        default=5,
        help="Top-k retrieved per sub-query (default: 5)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index/policy.index",
        help="FAISS index path to use",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="faiss_index/metadata.pkl",
        help="FAISS metadata path to use",
    )
    parser.add_argument(
        "--regressor-model-path",
        type=str,
        default="models/reward_regressor.pkl",
        help="Regressor model path to use",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for candidate filtering (default: 0.7)",
    )
    args = parser.parse_args()

    retriever = PolicyRetriever(
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        regressor_model_path=args.regressor_model_path,
        application_name="Grid World",
        available_actions=GRIDWORLD_AVAILABLE_ACTIONS,
    )

    # Resolve run directories and canonical states per spec.
    run_dirs = {}
    canonical_by_spec = {}
    grid_by_spec = {}
    for spec in args.specs:
        run_dir = find_latest_run_dir(args.state_runs_dir, spec)
        if not run_dir:
            print(f"Warning: no run_dir found for {spec} in {args.state_runs_dir}")
            continue
        run_dirs[spec] = run_dir
        grid_size, canonical_count = infer_grid_and_canonical(run_dir)
        canonical_by_spec[spec] = load_canonical_states(
            states_folder=run_dir,
            canonical_states=canonical_count,
            run_dir=run_dir,
        )
        grid_by_spec[spec] = grid_size

    if not run_dirs:
        raise SystemExit("No valid spec run directories found.")

    seeds = args.seeds
    if seeds is None:
        seeds = sorted(
            {
                str(m.get("policy_seed"))
                for m in retriever.vdb.metadata
                if m.get("policy_seed") is not None
            }
        )

    experiment_groups = build_experiment_groups()
    decomp_cache = {}
    exp_subqueries = {}
    for mode, experiments, policy_list in experiment_groups:
        for exp in experiments:
            setup = exp["setup"]
            query = exp["query"]
            expected_count = exp.get("expected_count")
            cache_key = (mode, query, expected_count)
            if cache_key not in decomp_cache:
                print(
                    f"Decomposing query for mode={mode}, setup={setup} (expected {expected_count})..."
                )
                sub_queries, decomp_time = decompose_query_with_retry(
                    retriever,
                    query,
                    max_attempts=3,
                    expected_count=expected_count,
                    policy_list=policy_list,
                )
                decomp_cache[cache_key] = (sub_queries, decomp_time)
            exp_subqueries[(mode, setup)] = decomp_cache[cache_key]

    run_cache = {}
    global_min_candidates = None
    total_runs = 0
    skipped_runs = 0

    for spec, run_dir in run_dirs.items():
        canonical_states = canonical_by_spec[spec]
        grid_size = grid_by_spec[spec]
        for mode, experiments, _ in experiment_groups:
            for exp in experiments:
                setup = exp["setup"]
                sub_queries, decomp_time = exp_subqueries[(mode, setup)]
                for seed in seeds:
                    seed_name = normalize_seed(seed)
                    if not seed_dir_exists(run_dir, setup, seed_name):
                        skipped_runs += 1
                        continue
                    total_runs += 1
                    env = init_env_from_run(run_dir, setup, seed_name, grid_size)
                    grouped, search_time = group_candidates_for_subqueries(
                        retriever,
                        sub_queries,
                        seed_name,
                        similarity_threshold=args.similarity_threshold,
                        search_k=args.search_k,
                    )
                    if not grouped or any(len(g) == 0 for g in grouped):
                        skipped_runs += 1
                        continue
                    min_len = min(len(g) for g in grouped)
                    if global_min_candidates is None:
                        global_min_candidates = min_len
                    else:
                        global_min_candidates = min(global_min_candidates, min_len)

                    run_cache[(spec, mode, setup, seed_name)] = {
                        "groups": grouped,
                        "search_time": search_time,
                        "env": env,
                        "canonical_states": canonical_states,
                        "decomp_time": decomp_time,
                    }

    if not run_cache:
        raise SystemExit("No valid runs found after candidate filtering.")

    if global_min_candidates is None or global_min_candidates <= 2:
        raise SystemExit(
            "Insufficient candidates to sweep (need at least 3 per sub-query)."
        )

    max_k = global_min_candidates - 1
    if args.max_k is not None:
        max_k = min(max_k, args.max_k)

    if args.min_k > max_k:
        raise SystemExit(
            f"Invalid k range: min_k={args.min_k} > max_k={max_k}."
        )

    k_values = list(range(args.min_k, max_k + 1))
    results = []

    for k in k_values:
        agg = defaultdict(float)
        count = 0
        for (spec, mode, setup, seed_name), cache in run_cache.items():
            groups = cache["groups"]
            if any(len(g) < k for g in groups):
                continue
            q_best, combo_time = best_hybrid_from_groups(
                retriever,
                groups,
                k,
                cache["canonical_states"],
                cache["env"],
                seed_name,
            )
            if q_best is None:
                continue
            reward = greedy_eval(cache["env"], q_best)
            hybrid_time = cache["search_time"] + combo_time
            total_time = hybrid_time + cache["decomp_time"]

            agg["reward"] += reward
            agg["hybrid_time"] += hybrid_time
            agg["total_time"] += total_time
            agg["decomp_time"] += cache["decomp_time"]
            agg["search_time"] += cache["search_time"]
            agg["combo_time"] += combo_time
            count += 1

        if count == 0:
            continue
        results.append(
            {
                "k": k,
                "avg_reward": agg["reward"] / count,
                "avg_time_s": agg["total_time"] / count,
                "avg_hybrid_time_s": agg["hybrid_time"] / count,
                "avg_decomp_time_s": agg["decomp_time"] / count,
                "avg_search_time_s": agg["search_time"] / count,
                "avg_combo_time_s": agg["combo_time"] / count,
                "runs_used": count,
                "runs_total": len(run_cache),
                "min_candidates": global_min_candidates,
            }
        )
        print(
            f"k={k}: avg_reward={results[-1]['avg_reward']:.2f}, "
            f"avg_time={results[-1]['avg_time_s']:.2f}s (n={count})"
        )

    output_path = args.output
    if output_path.endswith(os.sep) or os.path.isdir(output_path):
        output_path = os.path.join(output_path, "hybrid_k_sweep.csv")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "avg_reward",
                "avg_time_s",
                "avg_hybrid_time_s",
                "avg_decomp_time_s",
                "avg_search_time_s",
                "avg_combo_time_s",
                "runs_used",
                "runs_total",
                "min_candidates",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote sweep results to {output_path}")


if __name__ == "__main__":
    main()
