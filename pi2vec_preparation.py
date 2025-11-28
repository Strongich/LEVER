"""
Preparation script for pi2vec framework.

This script ensures all required models are trained:
1. Successor models (if regressor training data doesn't exist)
2. Regressor model (if not already trained)

After training, confirms that the framework is ready for use.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pi2vec.train_regressor import main as train_regressor_main
from pi2vec.train_successor import main as train_successor_main


def check_successor_trained():
    """
    Check if successor models have been trained.

    Returns:
        bool: True if training data exists, False otherwise
    """
    training_data_path = "data/regressor_training_data.json"
    return os.path.exists(training_data_path)


def check_regressor_trained():
    """
    Check if regressor model has been trained.

    Returns:
        bool: True if model exists, False otherwise
    """
    model_path = "models/reward_regressor.pkl"
    return os.path.exists(model_path)


def main():
    """Main preparation function."""
    print("=" * 80)
    print("pi2vec Framework Preparation")
    print("=" * 80)
    print()

    # Check successor models
    print("Step 1: Checking successor models...")
    if check_successor_trained():
        print("✓ Successor models already trained")
        print("  Training data found: data/regressor_training_data.json")
    else:
        print("⚠️  Successor models not found")
        print("  Training successor models...")
        print()
        try:
            train_successor_main()
            print()
            print("✓ Successor models training completed")
        except Exception as e:
            print(f"❌ Error training successor models: {e}")
            print("  Please check the error above and try again.")
            return
    print()

    # Check regressor model
    print("Step 2: Checking regressor model...")
    if check_regressor_trained():
        print("✓ Regressor model already trained")
        print("  Model found: models/reward_regressor.pkl")
    else:
        print("⚠️  Regressor model not found")
        print("  Training regressor model...")
        print()
        try:
            train_regressor_main()
            print()
            print("✓ Regressor model training completed")
        except Exception as e:
            print(f"❌ Error training regressor model: {e}")
            print("  Please check the error above and try again.")
            return
    print()

    # Final verification
    print("Step 3: Verifying framework readiness...")
    successor_ready = check_successor_trained()
    regressor_ready = check_regressor_trained()

    if successor_ready and regressor_ready:
        print("=" * 80)
        print("✅ Framework is ready!")
        print("=" * 80)
        print()
        print("All required models have been trained and are ready for use.")
        print()
        print("You can now use the framework via CLI:")
        print(
            '  python search_faiss_policies.py "your query here" [--seed SEED] [--filter-energy] [--show-all]'
        )
        print()
        print("Example:")
        print(
            '  python search_faiss_policies.py "collect gold efficiently" --seed " --filter-energy'
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


if __name__ == "__main__":
    main()
