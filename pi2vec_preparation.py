"""
Preparation script for pi2vec framework.

This script ensures all required models are trained:
1. Successor models (if regressor training data doesn't exist)
2. Regressor model (if not already trained)

After training, confirms that the framework is ready for use.
"""

import argparse
import os

from pi2vec.train_regressor import main as train_regressor_main
from pi2vec.train_successor import main as train_successor_main


def check_successor_trained(
    training_data_path: str = "data/regressor_training_data.json",
):
    """
    Check if successor models have been trained.

    Returns:
        bool: True if training data exists, False otherwise
    """
    return os.path.exists(training_data_path)


def check_regressor_trained(model_path: str = "models/reward_regressor.pkl"):
    """
    Check if regressor model has been trained.

    Returns:
        bool: True if model exists, False otherwise
    """
    return os.path.exists(model_path)


def main(
    states_folder: str = "states_16",
    canonical_states: int = 64,
    run_dir: str | None = None,
    reward_systems: list[str] | None = None,
    regressor_data_path: str = "data/regressor_training_data.json",
    regressor_model_path: str = "models/reward_regressor.pkl",
    regressor_plot_path: str = "plots/regression_plot.jpeg",
):
    """
    Main preparation function.

    Args:
        states_folder: Name of the folder containing states (e.g., "states_16").
        canonical_states: Total number of canonical states to collect (default: 64)
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
        print("  Training regressor model...")
        print()
        try:
            train_regressor_main(
                training_data_path=regressor_data_path,
                model_path=regressor_model_path,
                plot_path=regressor_plot_path,
            )
            print()
            print("✓ Regressor model training completed")
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


def prepare_from_state_runs(
    base_dir: str = "state_runs",
    specs: tuple[str, ...] = ("X1", "X5", "X10"),
    canonical_states: int = 128,
    reward_systems: list[str] | None = None,
    index_path: str = "faiss_index/policy.index",
    metadata_path: str = "faiss_index/metadata.pkl",
    regressor_data_path: str = "data/regressor_training_data.json",
    regressor_model_path: str = "models/reward_regressor.pkl",
    regressor_plot_path: str = "plots/regression_plot.jpeg",
    reset_index: bool = True,
    reset_regressor: bool = True,
    train_regressor: bool = True,
):
    # Reset artifacts to avoid mixing old embeddings with new feature dimensions.
    if reset_index:
        for path in (index_path, metadata_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Warning: failed to remove {path}: {e}")
    if reset_regressor:
        for path in (regressor_data_path, regressor_model_path):
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

    for idx, (spec, run_dir) in enumerate(run_dirs):
        print(f"[{spec}] Training successor models...")
        train_successor_main(
            states_folder=os.path.basename(run_dir),
            canonical_states=canonical_states,
            run_dir=run_dir,
            reward_systems=reward_systems,
            regressor_data_path=regressor_data_path,
            append_regressor_data=idx > 0,
            index_path=index_path,
            metadata_path=metadata_path,
        )
        print(f"[{spec}] Successor models completed.")

    if train_regressor:
        print("Training regressor model...")
        train_regressor_main(
            training_data_path=regressor_data_path,
            model_path=regressor_model_path,
            plot_path=regressor_plot_path,
        )
        print("✓ Regressor model training completed")
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
        help="Optional reward systems to include (default: all found in run_dir)",
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
        default="models/reward_regressor.pkl",
        help="Regressor model output path",
    )
    parser.add_argument(
        "--regressor-plot-path",
        type=str,
        default="plots/regression_plot.jpeg",
        help="Regressor plot output path",
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
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        regressor_data_path=args.regressor_data_path,
        regressor_model_path=args.regressor_model_path,
        regressor_plot_path=args.regressor_plot_path,
        reset_index=reset_index,
        reset_regressor=reset_regressor,
        train_regressor=train_regressor,
    )
