"""
Preparation script for pi2vec framework.

This script ensures all required models are trained:
1. Successor models (if regressor training data doesn't exist)
2. Regressor model (if not already trained)

After training, confirms that the framework is ready for use.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pi2vec.train_regressor import train_regressor_variants
from pi2vec.train_successor import main as train_successor_main

BASE_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever"]
COMBINED_REWARD_SYSTEMS = [
    "hazard-lever",
    "path-gold",
    "path-gold-hazard",
    "path-gold-hazard-lever",
]

# Policy sets for each experiment type (to prevent retrieval leakage)
# Trivial: only base policies (for composing primitives)
TRIVIAL_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever"]
# Double: base + pair policies (path-gold, hazard-lever) for 2-policy composition
DOUBLE_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever", "path-gold", "hazard-lever"]
# Triple: base + triple policy (path-gold-hazard) for 3-policy composition
TRIPLE_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever", "path-gold-hazard"]


def _with_combined_rewards(reward_systems: list[str] | None) -> list[str]:
    if reward_systems is None:
        return BASE_REWARD_SYSTEMS + COMBINED_REWARD_SYSTEMS
    merged: list[str] = []
    for entry in reward_systems + COMBINED_REWARD_SYSTEMS:
        if entry not in merged:
            merged.append(entry)
    return merged


def check_successor_trained(
    training_data_path: str = "data/regressor_training_data.json",
):
    """
    Check if successor models have been trained.

    Returns:
        bool: True if training data exists, False otherwise
    """
    return os.path.exists(training_data_path)


def check_regressor_trained(model_path: str = "models/reward_regressor_base.pkl"):
    """
    Check if regressor model has been trained.

    Returns:
        bool: True if model exists, False otherwise
    """
    return os.path.exists(model_path)


def main(
    states_folder: str = "states_16",
    canonical_states: int = 128,
    run_dir: str | None = None,
    reward_systems: list[str] | None = None,
    regressor_data_path: str = "data/regressor_training_data.json",
    regressor_model_path: str = "models/reward_regressor_base.pkl",
    regressor_plot_path: str = "plots/regression_plot_base.jpeg",
):
    """
    Main preparation function.

    Args:
        states_folder: Name of the folder containing states (e.g., "states_16").
        canonical_states: Total number of canonical states to collect (default: 128)
        run_dir: Path to a state_runs folder; if provided, overrides states_folder.
        reward_systems: Optional list of reward systems to include.
        regressor_data_path: Output path for regressor training data.
        regressor_model_path: Output path for the regressor model.
    """
    print("=" * 80)
    print("pi2vec Framework Preparation")
    print("=" * 80)
    print()

    # Check successor models
    print("Step 1: Checking successor models...")
    if check_successor_trained(training_data_path=regressor_data_path):
        print("✓ Successor models already trained")
        print(f"  Training data found: {regressor_data_path}")
    else:
        print("⚠️  Successor models not found")
        print("  Training successor models...")
        print()
        try:
            train_successor_main(
                states_folder=states_folder,
                canonical_states=canonical_states,
                run_dir=run_dir,
                reward_systems=reward_systems,
                regressor_data_path=regressor_data_path,
            )
            print()
            print("✓ Successor models training completed")
        except Exception as e:
            print(f"❌ Error training successor models: {e}")
            print("  Please check the error above and try again.")
            return
    print()

    # Check regressor model
    print("Step 2: Checking regressor model...")
    if check_regressor_trained(model_path=regressor_model_path):
        print("✓ Regressor model already trained")
        print(f"  Model found: {regressor_model_path}")
    else:
        print("⚠️  Regressor model not found")
        print("  Training regressor models (base/pair/trip)...")
        print()
        try:
            train_regressor_variants(
                source_json_path=regressor_data_path,
                output_json_paths={
                    "base": "data/regressor_training_data_base.json",
                    "pair": "data/regressor_training_data_pair.json",
                    "trip": "data/regressor_training_data_trip.json",
                },
                output_model_paths={
                    "base": "models/reward_regressor_base.pkl",
                    "pair": "models/reward_regressor_pair.pkl",
                    "trip": "models/reward_regressor_trip.pkl",
                },
                output_plot_paths={
                    "base": "plots/regression_plot_base.jpeg",
                    "pair": "plots/regression_plot_pair.jpeg",
                    "trip": "plots/regression_plot_trip.jpeg",
                },
            )
            print()
            print("✓ Regressor models training completed")
        except Exception as e:
            print(f"❌ Error training regressor model: {e}")
            print("  Please check the error above and try again.")
            return
    print()

    # Final verification
    print("Step 3: Verifying framework readiness...")
    successor_ready = check_successor_trained(training_data_path=regressor_data_path)
    regressor_ready = check_regressor_trained(model_path=regressor_model_path)

    if successor_ready and regressor_ready:
        print("=" * 80)
        print("✅ Framework is ready!")
        print("=" * 80)
        print()
        print("All required models have been trained and are ready for use.")
        print()
        print("You can now use the framework via CLI:")
        print(
            '  python search_faiss_policies.py "your query here" [--seed SEED] [--filter-energy]'
        )
        print()
        print("Example:")
        print(
            '  python search_faiss_policies.py "collect gold efficiently" --seed 0003 --filter-energy'
        )
        print()
    else:
        print("=" * 80)
        print("❌ Framework is not ready")
        print("=" * 80)
        print()
        if not successor_ready:
            print("  - Successor models are missing")
        if not regressor_ready:
            print("  - Regressor model is missing")
        print()
        print("Please check the errors above and ensure all models are trained.")
        return


def _find_latest_run_dir(base_dir: str, spec: str) -> str | None:
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


def _format_spec_path(path: str | None, spec: str) -> str | None:
    if not path:
        return path
    if "{spec}" in path:
        return path.format(spec=spec)
    base_dir = os.path.dirname(path)
    base_name = os.path.basename(path)
    if base_dir:
        return os.path.join(base_dir, spec, base_name)
    return os.path.join(spec, base_name)


def _suffix_with_spec(path: str, spec: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_{spec}{ext}"


def _remove_path(path: str):
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"Warning: failed to remove {path}: {e}")


def _cleanup_run_artifacts(run_dir: str):
    run_name = os.path.basename(os.path.normpath(run_dir))
    data_dir = os.path.join(os.getcwd(), "data")
    _remove_path(os.path.join(data_dir, f"processed_states_{run_name}.csv"))
    _remove_path(os.path.join(data_dir, f"processed_states_transitions_{run_name}.pkl"))
    for fname in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
        if fname.startswith(f"canonical_states_{run_name}") and fname.endswith(".npy"):
            _remove_path(os.path.join(data_dir, fname))


def _get_experiment_index_paths(index_path: str, metadata_path: str, experiment: str):
    """Get index and metadata paths for a specific experiment type."""
    index_dir = os.path.dirname(index_path) or "faiss_index"
    index_name = os.path.basename(index_path)
    metadata_name = os.path.basename(metadata_path)
    exp_dir = os.path.join(index_dir, experiment)
    return os.path.join(exp_dir, index_name), os.path.join(exp_dir, metadata_name)


def _create_experiment_indices(
    run_dirs: list[tuple[str, str]],
    canonical_states: int,
    index_path: str,
    metadata_path: str,
    reset_index: bool = True,
):
    """
    Create separate FAISS indices for each experiment type (trivial, double, triple).

    This prevents retrieval leakage where a task can retrieve policies that already
    solve the target task (e.g., 'path-gold' task retrieving 'path-gold' policy).
    """
    experiment_configs = {
        "trivial": TRIVIAL_REWARD_SYSTEMS,
        "double": DOUBLE_REWARD_SYSTEMS,
        "triple": TRIPLE_REWARD_SYSTEMS,
    }

    for experiment, reward_systems in experiment_configs.items():
        exp_index_path, exp_metadata_path = _get_experiment_index_paths(
            index_path, metadata_path, experiment
        )

        # Reset experiment-specific index if requested
        if reset_index:
            for path in (exp_index_path, exp_metadata_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        print(f"Warning: failed to remove {path}: {e}")

        # Create directory for experiment index
        exp_dir = os.path.dirname(exp_index_path)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n[{experiment}] Creating index with policies: {reward_systems}")

        for idx, (spec, run_dir) in enumerate(run_dirs):
            print(f"  [{spec}] Adding policies to {experiment} index...")
            train_successor_main(
                states_folder=os.path.basename(run_dir),
                canonical_states=canonical_states,
                run_dir=run_dir,
                reward_systems=reward_systems,
                regressor_data_path="/dev/null",  # Don't collect regressor data
                append_regressor_data=False,
                collect_regressor_data=False,
                index_path=exp_index_path,
                metadata_path=exp_metadata_path,
            )
        print(f"[{experiment}] Index created at {exp_index_path}")


def prepare_from_state_runs(
    base_dir: str = "state_runs",
    specs: tuple[str, ...] = ("X1", "X5", "X10"),
    canonical_states: int = 128,
    reward_systems: list[str] | None = None,
    include_combined_rewards: bool = False,
    index_path: str = "faiss_index/policy.index",
    metadata_path: str = "faiss_index/metadata.pkl",
    regressor_data_path: str = "data/regressor_training_data.json",
    regressor_model_path: str = "models/reward_regressor_base.pkl",
    regressor_plot_path: str = "plots/regression_plot_base.jpeg",
    regressor_base_path: str | None = None,
    regressor_pair_path: str | None = None,
    regressor_trip_path: str | None = None,
    normalize_regressor: bool = False,
    regressor_random_search: bool = False,
    regressor_random_search_iters: int = 25,
    regressor_random_search_cv: int = 10,
    regressor_random_search_seed: int = 42,
    split_regressor_by_spec: bool = False,
    split_index_by_experiment: bool = False,
    overwrite: bool = False,
    reset_index: bool = True,
    reset_regressor: bool = True,
    train_regressor: bool = True,
):
    if overwrite:
        reset_index = True
        reset_regressor = True
        train_regressor = True
    regressor_base_path = regressor_base_path or regressor_model_path
    regressor_pair_path = regressor_pair_path or "models/reward_regressor_pair.pkl"
    regressor_trip_path = regressor_trip_path or "models/reward_regressor_trip.pkl"
    # Reset artifacts to avoid mixing old embeddings with new feature dimensions.
    if reset_index:
        for path in (index_path, metadata_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Warning: failed to remove {path}: {e}")
        # Also reset experiment-specific indices if split_index_by_experiment is enabled
        if split_index_by_experiment:
            for experiment in ("trivial", "double", "triple"):
                exp_index, exp_meta = _get_experiment_index_paths(
                    index_path, metadata_path, experiment
                )
                for path in (exp_index, exp_meta):
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError as e:
                            print(f"Warning: failed to remove {path}: {e}")
    if reset_regressor:
        extra_regressor_paths = (
            "data/regressor_training_data_base.json",
            "data/regressor_training_data_pair.json",
            "data/regressor_training_data_trip.json",
            regressor_base_path,
            regressor_pair_path,
            regressor_trip_path,
            "plots/regression_plot_base.jpeg",
            "plots/regression_plot_pair.jpeg",
            "plots/regression_plot_trip.jpeg",
        )
        for path in (regressor_data_path, regressor_model_path, *extra_regressor_paths):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Warning: failed to remove {path}: {e}")

    run_dirs = []
    for spec in specs:
        run_dir = _find_latest_run_dir(base_dir, spec)
        if run_dir:
            run_dirs.append((spec, run_dir))
        else:
            print(f"Warning: no run_dir found for {spec} in {base_dir}")

    if not run_dirs:
        print(f"❌ No runs found under {base_dir}")
        return

    print("=" * 80)
    print("pi2vec Framework Preparation (state_runs)")
    print("=" * 80)
    print("Runs:")
    for spec, run_dir in run_dirs:
        print(f"- {spec}: {run_dir}")
    print()

    if overwrite:
        for _, run_dir in run_dirs:
            _cleanup_run_artifacts(run_dir)
        for spec, _ in run_dirs:
            _remove_path(_suffix_with_spec(regressor_data_path, spec))
            _remove_path(
                _suffix_with_spec("data/regressor_training_data_base.json", spec)
            )
            _remove_path(
                _suffix_with_spec("data/regressor_training_data_pair.json", spec)
            )
            _remove_path(
                _suffix_with_spec("data/regressor_training_data_trip.json", spec)
            )
            _remove_path(_format_spec_path(regressor_base_path, spec))
            _remove_path(_format_spec_path(regressor_pair_path, spec))
            _remove_path(_format_spec_path(regressor_trip_path, spec))
            _remove_path(_suffix_with_spec("plots/regression_plot_base.jpeg", spec))
            _remove_path(_suffix_with_spec("plots/regression_plot_pair.jpeg", spec))
            _remove_path(_suffix_with_spec("plots/regression_plot_trip.jpeg", spec))

    if reward_systems is None:
        reward_systems = BASE_REWARD_SYSTEMS
    regressor_reward_systems = (
        _with_combined_rewards(reward_systems)
        if include_combined_rewards
        else reward_systems
    )
    combined_only = [
        r for r in COMBINED_REWARD_SYSTEMS if r not in regressor_reward_systems
    ]

    for idx, (spec, run_dir) in enumerate(run_dirs):
        print(f"[{spec}] Training successor models...")
        train_successor_main(
            states_folder=os.path.basename(run_dir),
            canonical_states=canonical_states,
            run_dir=run_dir,
            reward_systems=regressor_reward_systems,
            regressor_data_path=regressor_data_path,
            append_regressor_data=idx > 0,
            index_path=index_path,
            metadata_path=metadata_path,
        )
        if combined_only:
            print(f"[{spec}] Adding combined policies to VDB...")
            train_successor_main(
                states_folder=os.path.basename(run_dir),
                canonical_states=canonical_states,
                run_dir=run_dir,
                reward_systems=combined_only,
                regressor_data_path=regressor_data_path,
                append_regressor_data=False,
                collect_regressor_data=False,
                index_path=index_path,
                metadata_path=metadata_path,
            )
        print(f"[{spec}] Successor models completed.")

    # Create experiment-specific indices to prevent retrieval leakage
    if split_index_by_experiment:
        print("\n" + "=" * 80)
        print("Creating experiment-specific indices (trivial, double, triple)")
        print("=" * 80)
        _create_experiment_indices(
            run_dirs=run_dirs,
            canonical_states=canonical_states,
            index_path=index_path,
            metadata_path=metadata_path,
            reset_index=False,  # Already reset above
        )
        print("\n✓ Experiment-specific indices created")

    if train_regressor:
        if split_regressor_by_spec:
            if not os.path.exists(regressor_data_path):
                raise FileNotFoundError(
                    f"Regressor training data not found: {regressor_data_path}"
                )
            with open(regressor_data_path, "r") as f:
                full_data = json.load(f)
            spec_list = full_data.get("spec")
            if not spec_list:
                raise ValueError(
                    "Regressor training data missing 'spec'; cannot split by spec."
                )
            spec_list = [str(s) if s is not None else "unknown" for s in spec_list]
            total = len(full_data.get("reward", []))
            if total == 0:
                raise ValueError("No regressor training samples found.")

            keys = [
                "policy_embedding",
                "policy_embedding_base",
                "policy_embedding_pair",
                "policy_embedding_trip",
                "reward",
                "policy_target",
                "spec",
            ]
            print("Training regressor models per spec (base/pair/trip)...")
            for spec in specs:
                indices = [i for i, s in enumerate(spec_list) if s == str(spec)]
                if not indices:
                    print(f"Warning: no regressor samples found for spec {spec}")
                    continue
                spec_data = {}
                for key in keys:
                    values = full_data.get(key)
                    if values is None:
                        continue
                    if len(values) != total:
                        raise ValueError(
                            f"Regressor data key '{key}' length mismatch (expected {total})."
                        )
                    spec_data[key] = [values[i] for i in indices]

                spec_source_path = _suffix_with_spec(regressor_data_path, spec)
                os.makedirs(os.path.dirname(spec_source_path) or ".", exist_ok=True)
                with open(spec_source_path, "w") as f:
                    json.dump(spec_data, f, indent=2)

                base_plot = _suffix_with_spec("plots/regression_plot_base.jpeg", spec)
                pair_plot = _suffix_with_spec("plots/regression_plot_pair.jpeg", spec)
                trip_plot = _suffix_with_spec("plots/regression_plot_trip.jpeg", spec)

                train_regressor_variants(
                    source_json_path=spec_source_path,
                    output_json_paths={
                        "base": _suffix_with_spec(
                            "data/regressor_training_data_base.json", spec
                        ),
                        "pair": _suffix_with_spec(
                            "data/regressor_training_data_pair.json", spec
                        ),
                        "trip": _suffix_with_spec(
                            "data/regressor_training_data_trip.json", spec
                        ),
                    },
                    output_model_paths={
                        "base": _format_spec_path(regressor_base_path, spec),
                        "pair": _format_spec_path(regressor_pair_path, spec),
                        "trip": _format_spec_path(regressor_trip_path, spec),
                    },
                    output_plot_paths={
                        "base": base_plot,
                        "pair": pair_plot,
                        "trip": trip_plot,
                    },
                    embedding_key_by_variant={
                        "base": "policy_embedding_base",
                        "pair": "policy_embedding_pair",
                        "trip": "policy_embedding_trip",
                    },
                    normalize_embeddings=normalize_regressor,
                    random_search=regressor_random_search,
                    random_search_iters=regressor_random_search_iters,
                    random_search_cv=regressor_random_search_cv,
                    random_search_seed=regressor_random_search_seed,
                )
            print("✓ Regressor models trained per spec")
        else:
            print("Training regressor models (base/pair/trip)...")
            train_regressor_variants(
                source_json_path=regressor_data_path,
                output_json_paths={
                    "base": "data/regressor_training_data_base.json",
                    "pair": "data/regressor_training_data_pair.json",
                    "trip": "data/regressor_training_data_trip.json",
                },
                output_model_paths={
                    "base": regressor_base_path,
                    "pair": regressor_pair_path,
                    "trip": regressor_trip_path,
                },
                output_plot_paths={
                    "base": "plots/regression_plot_base.jpeg",
                    "pair": "plots/regression_plot_pair.jpeg",
                    "trip": "plots/regression_plot_trip.jpeg",
                },
                embedding_key_by_variant={
                    "base": "policy_embedding_base",
                    "pair": "policy_embedding_pair",
                    "trip": "policy_embedding_trip",
                },
                normalize_embeddings=normalize_regressor,
                random_search=regressor_random_search,
                random_search_iters=regressor_random_search_iters,
                random_search_cv=regressor_random_search_cv,
                random_search_seed=regressor_random_search_seed,
            )
            print("✓ Regressor models trained")
    else:
        print("Skipping regressor training (using existing model).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare pi2vec assets from state_runs (successor + FAISS + regressor)."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="state_runs",
        help="Base directory containing X1/X5/X10 runs",
    )
    parser.add_argument(
        "--specs",
        nargs="*",
        default=["X1", "X5", "X10"],
        help="Spec labels to include (e.g., X1 X5 X10)",
    )
    parser.add_argument(
        "--canonical-states",
        type=int,
        default=128,
        help="Number of canonical states to sample",
    )
    parser.add_argument(
        "--reward-systems",
        nargs="*",
        default=None,
        help="Optional reward systems to include (default: path gold hazard lever)",
    )
    parser.add_argument(
        "--include-combined-rewards",
        action="store_true",
        help="Include composite reward systems (e.g., path-gold-*) in regressor training",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index/policy.index",
        help="FAISS index output path",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="faiss_index/metadata.pkl",
        help="FAISS metadata output path",
    )
    parser.add_argument(
        "--regressor-data-path",
        type=str,
        default="data/regressor_training_data.json",
        help="Regressor training data output path",
    )
    parser.add_argument(
        "--regressor-model-path",
        type=str,
        default="models/reward_regressor_base.pkl",
        help="Regressor model output path",
    )
    parser.add_argument(
        "--regressor-plot-path",
        type=str,
        default="plots/regression_plot_base.jpeg",
        help="Regressor plot output path",
    )
    parser.add_argument(
        "--regressor-base-path",
        type=str,
        default=None,
        help="Regressor base model output path (overrides --regressor-model-path)",
    )
    parser.add_argument(
        "--regressor-pair-path",
        type=str,
        default=None,
        help="Regressor pair model output path",
    )
    parser.add_argument(
        "--regressor-trip-path",
        type=str,
        default=None,
        help="Regressor trip model output path",
    )
    parser.add_argument(
        "--normalize-regressor",
        action="store_true",
        help="Normalize embeddings (zero mean, unit variance) before regressor fit",
    )
    parser.add_argument(
        "--regressor-random-search",
        action="store_true",
        help="Run RandomizedSearchCV for the regressor before fitting",
    )
    parser.add_argument(
        "--regressor-random-search-iters",
        type=int,
        default=25,
        help="Number of RandomizedSearchCV iterations",
    )
    parser.add_argument(
        "--regressor-random-search-cv",
        type=int,
        default=10,
        help="Number of CV folds for RandomizedSearchCV",
    )
    parser.add_argument(
        "--regressor-random-search-seed",
        type=int,
        default=42,
        help="Random seed for RandomizedSearchCV",
    )
    parser.add_argument(
        "--split-regressor-by-spec",
        action="store_true",
        help="Train separate regressors per spec (X1/X5/X10) using spec-specific data",
    )
    parser.add_argument(
        "--split-index-by-experiment",
        action="store_true",
        help="Create separate FAISS indices for trivial/double/triple experiments to prevent retrieval leakage",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Skip existing checks and overwrite cached artifacts/models",
    )
    parser.add_argument(
        "--no-reset-index",
        action="store_true",
        help="Do not remove existing FAISS index files",
    )
    parser.add_argument(
        "--no-reset-regressor",
        action="store_true",
        help="Do not remove existing regressor artifacts",
    )
    parser.add_argument(
        "--skip-regressor",
        action="store_true",
        help="Skip regressor training (use existing model)",
    )
    args = parser.parse_args()

    reset_index = not args.no_reset_index
    reset_regressor = not args.no_reset_regressor
    train_regressor = not args.skip_regressor
    if args.skip_regressor:
        reset_regressor = False

    prepare_from_state_runs(
        base_dir=args.base_dir,
        specs=tuple(args.specs),
        canonical_states=args.canonical_states,
        reward_systems=args.reward_systems,
        include_combined_rewards=args.include_combined_rewards,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        regressor_data_path=args.regressor_data_path,
        regressor_model_path=args.regressor_model_path,
        regressor_plot_path=args.regressor_plot_path,
        regressor_base_path=args.regressor_base_path,
        regressor_pair_path=args.regressor_pair_path,
        regressor_trip_path=args.regressor_trip_path,
        normalize_regressor=args.normalize_regressor,
        regressor_random_search=args.regressor_random_search,
        regressor_random_search_iters=args.regressor_random_search_iters,
        regressor_random_search_cv=args.regressor_random_search_cv,
        regressor_random_search_seed=args.regressor_random_search_seed,
        split_regressor_by_spec=args.split_regressor_by_spec,
        split_index_by_experiment=args.split_index_by_experiment,
        overwrite=args.overwrite,
        reset_index=reset_index,
        reset_regressor=reset_regressor,
        train_regressor=train_regressor,
    )
