"""
Reset Framework Script

This script is created to easily reset the LEVER framework in case you want to use
different training data or start fresh. It will delete:

1. All data from the data/ folder
2. The trained regressor model (models/reward_regressor.pkl)
3. The Faiss vector database (faiss_index/ folder)
4. The successor feature models (psi_models/ folder)

WARNING: This will permanently delete all trained models and processed data.
Make sure you have backups if needed before running this script.

Usage:
    python reset_framework.py
"""

import os
import shutil
from pathlib import Path


def delete_directory(path):
    """Delete a directory if it exists."""
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"✓ Deleted: {path}")
            return True
        except Exception as e:
            print(f"✗ Error deleting {path}: {e}")
            return False
    else:
        print(f"⚠️  Not found (skipping): {path}")
        return False


def delete_file(path):
    """Delete a file if it exists."""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"✓ Deleted: {path}")
            return True
        except Exception as e:
            print(f"✗ Error deleting {path}: {e}")
            return False
    else:
        print(f"⚠️  Not found (skipping): {path}")
        return False


def delete_data_folder_contents():
    """Delete all contents of the data/ folder but keep the folder itself."""
    data_dir = Path("data")
    if not data_dir.exists():
        print("⚠️  data/ folder does not exist (skipping)")
        return False

    deleted_any = False
    for item in data_dir.iterdir():
        try:
            if item.is_file():
                item.unlink()
                print(f"✓ Deleted file: {item}")
                deleted_any = True
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"✓ Deleted directory: {item}")
                deleted_any = True
        except Exception as e:
            print(f"✗ Error deleting {item}: {e}")

    if not deleted_any:
        print("⚠️  data/ folder is empty (nothing to delete)")

    return deleted_any


def main():
    """Main function to reset the framework."""
    print("=" * 80)
    print("LEVER Framework Reset")
    print("=" * 80)
    print()
    print("This will delete:")
    print("  1. All data from data/ folder")
    print("  2. models/reward_regressor.pkl")
    print("  3. faiss_index/ folder")
    print("  4. psi_models/ folder")
    print()
    print("WARNING: This action cannot be undone!")
    print()

    # Ask for confirmation
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Reset cancelled.")
        return

    print()
    print("Starting reset...")
    print("-" * 80)

    # 1. Delete all data from data/ folder
    print("\n1. Deleting data/ folder contents...")
    delete_data_folder_contents()

    # 2. Delete regressor model
    print("\n2. Deleting regressor model...")
    delete_file("models/reward_regressor.pkl")

    # 3. Delete faiss_index folder
    print("\n3. Deleting Faiss index...")
    delete_directory("faiss_index")

    # 4. Delete psi_models folder
    print("\n4. Deleting successor feature models...")
    delete_directory("psi_models")

    print()
    print("=" * 80)
    print("Reset complete!")
    print("=" * 80)
    print()
    print("You can now run the framework preparation again:")
    print("  python pi2vec_preparation.py")
    print()


if __name__ == "__main__":
    main()

