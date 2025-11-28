"""
View all policies in the Faiss Vector Database.

This script displays:
1. All policies in the database
2. Metadata for each policy
3. Index statistics
"""

from setup_faiss_vdb import FaissVectorDB


def view_database(
    index_path="faiss_index/policy.index", metadata_path="faiss_index/metadata.pkl"
):
    """Display all policies and statistics in the Faiss VDB."""
    try:
        vdb = FaissVectorDB(index_path=index_path, metadata_path=metadata_path)
        vdb.load()
        print("✓ Loaded Faiss database")
    except FileNotFoundError as e:
        print(f"✗ Database not found: {e}")
        print("\nPlease run 'python setup_faiss_vdb.py' first to create the database.")
        return
    except Exception as e:
        print(f"✗ Failed to load database: {e}")
        return

    print("\n" + "=" * 80)
    print("FAISS VECTOR DATABASE CONTENTS")
    print("=" * 80)

    # Get index statistics
    stats = vdb.get_index_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Display all policies
    print("\n" + "=" * 80)
    print("POLICY METADATA")
    print("=" * 80)

    if not vdb.metadata:
        print("\n⚠️  No policies found in database")
        return

    print(f"\nFound {len(vdb.metadata)} policies:\n")

    for i, policy in enumerate(vdb.metadata, 1):
        # Safely get policy_id with multiple fallbacks
        if not isinstance(policy, dict):
            print(f"{i}. Policy ID: N/A (Invalid metadata format)")
            print("-" * 80)
            continue
        policy_id = (
            policy.get("policy_id") or policy.get("policy_name") or f"policy_{i}"
        )
        print(f"{i}. Policy ID: {policy_id}")
        print("-" * 80)
        # Display only the metadata fields that are actually available
        if policy.get("policy_name"):
            print(f"   {'Policy Name':<20} : {policy.get('policy_name')}")
        if policy.get("policy_target"):
            print(f"   {'Policy Target':<20} : {policy.get('policy_target')}")
        if policy.get("policy_seed"):
            print(f"   {'Policy Seed':<20} : {policy.get('policy_seed')}")
        if policy.get("description"):
            desc_preview = (
                policy.get("description").strip()[:100] + "..."
                if len(policy.get("description").strip()) > 100
                else policy.get("description").strip()
            )
            print(f"   {'Description':<20} : {desc_preview}")
        if policy.get("reward") is not None:
            print(f"   {'Reward':<20} : {policy.get('reward')}")
        if policy.get("energy_consumption") is not None:
            print(f"   {'Energy Consumption':<20} : {policy.get('energy_consumption')}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total policies: {len(vdb.metadata)}")

    # Get all unique policy targets (if available)
    policy_targets = set(
        policy.get("policy_target")
        for policy in vdb.metadata
        if policy.get("policy_target")
    )
    if policy_targets:
        print(f"Unique policy targets: {', '.join(sorted(policy_targets))}")

    # Files info
    print("\nFiles:")
    print(f"  Index file: {index_path}")
    print(f"  Metadata file: {metadata_path}")
    print()


if __name__ == "__main__":
    view_database()
