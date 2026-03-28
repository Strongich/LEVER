"""
Preparation script for DQN policies using the pi2vec framework.

Analogous to tabular/pi2vec_preparation.py but for deep RL (DQN) trained policies
stored under deeprl_runs_dqn_try6/8/.

Pipeline:
1. For each reward system folder, read episode_rewards.csv and select:
   - 5 VDB snapshots at quantiles 0.2/0.4/0.6/0.8/1.0 (reward>0, reward_det>0)
   - 60% of remaining positive snapshots for regressor-only training
2. Encode transitions via state_to_vector with minigrid_ids mapping.
3. For each of three setups (base/pair/trip), create 128 canonical states,
   train successor models, populate a FAISS VDB, and train a regressor.
"""

import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pi2vec.pi2vec_utils import _build_minigrid_value_map, state_to_vector
from pi2vec.psimodel import SuccessorFeatureModelDeep
from pi2vec.train_regressor import train_regressor_variants
from pi2vec.train_successor import train_and_save_successor_model_variants

# Setup definitions (base=trivial, pair=double, trip=triple)
BASE_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever"]
PAIR_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever", "path-gold", "hazard-lever"]
TRIP_REWARD_SYSTEMS = ["path", "gold", "hazard", "lever", "path-gold-hazard"]

SETUP_CONFIGS = {
    "base": BASE_REWARD_SYSTEMS,
    "pair": PAIR_REWARD_SYSTEMS,
    "trip": TRIP_REWARD_SYSTEMS,
}

QUANTILE_LABELS = {0.2: 20, 0.4: 40, 0.6: 60, 0.8: 80, 1.0: 100}


def select_policy_snapshots(
    rewards_csv_path: str,
    regressor_only_fraction: float = 0.6,
    seed: int = 42,
    prefilter: str = "v1",
) -> tuple[list[dict], list[dict]]:
    """
    Select policy snapshots from episode_rewards.csv.

    prefilter variants:
      v1 – filter reward > 0 AND reward_det > 0, sort by [reward, reward_det] asc.
      v2 – no positive filter; sort by [reward_det, success_rate] asc.

    Picks 5 at quantiles 0.2/0.4/0.6/0.8/1.0 (quantile 1.0 = best).
    From remaining rows, samples 60% for regressor-only use.

    Returns:
        vdb_snapshots: 5 snapshots for VDB + regressor training
        regressor_only_snapshots: additional snapshots for regressor training only
    """
    df = pd.read_csv(rewards_csv_path)
    for col in ("reward", "reward_det", "timesteps", "success_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["reward", "reward_det", "timesteps"])

    if prefilter == "v1":
        candidates = df[(df["reward"] > 0) & (df["reward_det"] > 0)].copy()
        sort_cols = ["reward", "reward_det"]
    elif prefilter == "v2":
        candidates = df.copy()
        sort_cols = ["reward_det", "success_rate"]
    else:
        raise ValueError(f"Unknown prefilter: {prefilter}")

    if candidates.empty:
        return [], []

    # Sort ascending so quantile 1.0 = best
    candidates = candidates.sort_values(
        sort_cols, ascending=[True] * len(sort_cols)
    ).reset_index(drop=True)

    n = len(candidates)
    vdb_indices: set[int] = set()
    vdb_snapshots: list[dict] = []

    for quantile, label in sorted(QUANTILE_LABELS.items()):
        idx = min(int(round(quantile * (n - 1))), n - 1)
        # Resolve collisions by scanning forward then backward
        orig_idx = idx
        while idx in vdb_indices and idx < n - 1:
            idx += 1
        if idx in vdb_indices:
            idx = orig_idx
            while idx in vdb_indices and idx > 0:
                idx -= 1
        if idx in vdb_indices:
            continue
        vdb_indices.add(idx)
        row = candidates.iloc[idx]
        vdb_snapshots.append(
            {
                "timesteps": int(row["timesteps"]),
                "reward": float(row["reward"]),
                "reward_det": float(row["reward_det"]),
                "label": label,
                "include_in_vdb": True,
            }
        )

    # Remaining rows: sample 60% for regressor only
    remaining = candidates[~candidates.index.isin(vdb_indices)]
    sample_count = max(0, int(len(remaining) * regressor_only_fraction))
    regressor_only_snapshots: list[dict] = []
    if sample_count > 0:
        sampled = remaining.sample(n=sample_count, random_state=seed)
        for _, row in sampled.iterrows():
            regressor_only_snapshots.append(
                {
                    "timesteps": int(row["timesteps"]),
                    "reward": float(row["reward"]),
                    "reward_det": float(row["reward_det"]),
                    "label": f"ep{int(row['timesteps'])}",
                    "include_in_vdb": False,
                }
            )

    return vdb_snapshots, regressor_only_snapshots


def encode_transitions(
    transitions_path: str,
    minigrid_ids: dict,
    minigrid_value_map: dict[int, int],
    show_progress: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load eval_transitions.pkl and encode states via state_to_vector."""
    with open(transitions_path, "rb") as f:
        raw_transitions = pickle.load(f)

    encoded: list[tuple[np.ndarray, np.ndarray]] = []
    iterator = raw_transitions
    if show_progress:
        iterator = tqdm(raw_transitions, desc="Encoding", leave=False)

    for item in iterator:
        state = item.get("state", {})
        next_state = item.get("next_state", {})
        try:
            s_vec = state_to_vector(
                state,
                minigrid_ids=minigrid_ids,
                minigrid_value_map=minigrid_value_map,
            )
            s_next_vec = state_to_vector(
                next_state,
                minigrid_ids=minigrid_ids,
                minigrid_value_map=minigrid_value_map,
            )
            encoded.append((s_vec, s_next_vec))
        except Exception as e:
            continue

    return encoded


def sample_canonical_states(
    all_encoded: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    reward_systems: list[str],
    count: int = 128,
) -> np.ndarray:
    """
    Reservoir-sample canonical state vectors from encoded transitions,
    considering only policies whose reward system is in the given list.

    Keys in all_encoded are like "gold_20", "path-gold_ep500000", etc.
    """
    reservoir: list[np.ndarray] = []
    seen = 0

    for policy_key, transitions in all_encoded.items():
        # Extract reward system: everything before the last underscore
        parts = policy_key.rsplit("_", 1)
        reward_system = parts[0] if len(parts) == 2 else policy_key
        if reward_system not in reward_systems:
            continue
        for s_vec, s_next_vec in transitions:
            for vec in (s_vec, s_next_vec):
                seen += 1
                if len(reservoir) < count:
                    reservoir.append(vec)
                else:
                    idx = random.randrange(seen)
                    if idx < count:
                        reservoir[idx] = vec

    if not reservoir:
        return np.empty((0,), dtype=np.float32)
    return np.array(reservoir, dtype=np.float32)


DESC_MAP = {
    "path": "Find the shortest path to the exit.",
    "gold": "Collect as much gold as possible before exiting.",
    "lever": "Activate the lever before reaching the exit.",
    "hazard": "Reach the exit while avoiding hazards and staying away from them.",
    "hazard-lever": "Activate the lever and avoid hazards while reaching the exit.",
    "path-gold": "Find the fastest exit and collect as much gold as possible.",
    "path-gold-hazard": (
        "Find the fastest exit, collect as much gold as possible, and avoid hazards."
    ),
    "path-gold-hazard-lever": (
        "Find the fastest exit, collect as much gold as possible, avoid hazards, "
        "and activate the lever."
    ),
}


def prepare_dqn_policies(
    base_dir: str = "deeprl_runs_dqn_try6/8",
    minigrid_ids_path: str = "deeprl_runs_dqn_try6/minigrid_ids.json",
    canonical_count: int = 128,
    regressor_only_fraction: float = 0.6,
    index_dir: str = "faiss_index_dqn",
    data_dir: str = "data_rl",
    models_dir: str = "models_dqn",
    plots_dir: str = "plots",
    epochs: int = 50,
    seed: int = 42,
    show_progress: bool = True,
    reset: bool = True,
    skip_regressor: bool = False,
    normalize_regressor: bool = False,
    regressor_random_search: bool = False,
    regressor_random_search_iters: int = 25,
    regressor_random_search_cv: int = 10,
    regressor_random_search_seed: int = 42,
    prefilter: str = "v1",
):
    """Main preparation function for DQN policies."""
    from faiss_utils.setup_faiss_vdb import FaissVectorDB

    random.seed(seed)

    # Load minigrid IDs
    with open(minigrid_ids_path, "r") as f:
        minigrid_ids = json.load(f)
    minigrid_value_map = _build_minigrid_value_map(minigrid_ids)

    # Discover reward systems present in base_dir
    all_reward_systems = sorted(
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    print(f"Found reward systems: {all_reward_systems}")

    # ── Phase 1: Select snapshots and encode transitions ──────────────────
    print("=" * 80)
    print("Phase 1: Selecting snapshots and encoding transitions")
    print("=" * 80)

    # policy_key -> snapshot info dict
    all_snapshots: dict[str, dict] = {}
    # policy_key -> encoded transitions (shared across setups since model is shared)
    all_encoded: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for reward_system in all_reward_systems:
        rewards_csv = os.path.join(base_dir, reward_system, "episode_rewards.csv")
        if not os.path.exists(rewards_csv):
            print(f"Warning: {rewards_csv} not found, skipping {reward_system}")
            continue

        vdb_snapshots, regressor_snapshots = select_policy_snapshots(
            rewards_csv,
            regressor_only_fraction=regressor_only_fraction,
            seed=seed,
            prefilter=prefilter,
        )
        all_selected = vdb_snapshots + regressor_snapshots

        if not all_selected:
            print(f"Warning: no positive snapshots for {reward_system}")
            continue

        print(
            f"\n[{reward_system}] VDB: {len(vdb_snapshots)}, "
            f"regressor-only: {len(regressor_snapshots)}"
        )

        for snap in all_selected:
            timesteps = snap["timesteps"]
            label = snap["label"]
            policy_key = f"{reward_system}_{label}"

            episode_dir = os.path.join(
                base_dir, reward_system, "episodes", f"episode_{timesteps:06d}"
            )
            transitions_path = os.path.join(episode_dir, "eval_transitions.pkl")

            if not os.path.exists(transitions_path):
                print(f"  Warning: {transitions_path} not found, skipping")
                continue

            encoded = encode_transitions(
                transitions_path, minigrid_ids, minigrid_value_map
            )
            if not encoded:
                print(f"  Warning: no valid transitions for {policy_key}")
                continue

            all_encoded[policy_key] = encoded
            all_snapshots[policy_key] = {
                "transitions": encoded,
                "reward": snap["reward"],
                "reward_det": snap["reward_det"],
                "timesteps": timesteps,
                "include_in_vdb": snap["include_in_vdb"],
                "reward_system": reward_system,
                "label": label,
                "model_path": os.path.join(episode_dir, "model.zip"),
            }
            print(
                f"  {policy_key}: {len(encoded)} transitions, "
                f"reward={snap['reward']:.2f}, reward_det={snap['reward_det']:.2f}"
            )

    if not all_snapshots:
        print("No snapshots found. Aborting.")
        return

    # ── Phase 2: Create canonical states ──────────────────────────────────
    print("\n" + "=" * 80)
    print("Phase 2: Creating canonical states")
    print("=" * 80)

    os.makedirs(data_dir, exist_ok=True)
    canonical_states: dict[str, np.ndarray] = {}

    for setup_name, setup_rewards in SETUP_CONFIGS.items():
        cs = sample_canonical_states(all_encoded, setup_rewards, canonical_count)
        canonical_states[setup_name] = cs
        cs_path = os.path.join(data_dir, f"canonical_states_dqn_{setup_name}.npy")
        np.save(cs_path, cs)
        print(f"[{setup_name}] {len(cs)} canonical states -> {cs_path}")

    # ── Phase 3: Train successor models, populate VDBs, collect regressor data
    print("\n" + "=" * 80)
    print("Phase 3: Training successor models and populating VDBs")
    print("=" * 80)

    # Prepare VDB per setup
    vdbs: dict[str, object] = {}
    for setup_name in SETUP_CONFIGS:
        setup_dir = os.path.join(index_dir, setup_name)
        os.makedirs(setup_dir, exist_ok=True)
        idx_path = os.path.join(setup_dir, "policy_dqn.index")
        meta_path = os.path.join(setup_dir, "metadata_dqn.pkl")
        if reset:
            for p in (idx_path, meta_path):
                if os.path.exists(p):
                    os.remove(p)
        vdbs[setup_name] = FaissVectorDB(
            index_path=idx_path, metadata_path=meta_path
        )

    # Regressor training data per setup:
    # We store one embedding per (policy, setup) plus the reward.
    regressor_data: dict[str, dict[str, list]] = {
        setup_name: {
            "policy_embedding": [],
            "reward": [],
            "policy_target": [],
        }
        for setup_name in SETUP_CONFIGS
    }

    for policy_key in tqdm(
        sorted(all_snapshots.keys()),
        desc="Training successor models",
    ):
        snap = all_snapshots[policy_key]
        reward_system = snap["reward_system"]
        transitions = snap["transitions"]
        include_in_vdb = snap["include_in_vdb"]

        _, embeddings_by_variant = train_and_save_successor_model_variants(
            policy_key,
            transitions,
            canonical_states,
            epochs=epochs,
            show_progress=show_progress,
            policy_seed="dqn",
            model_cls=SuccessorFeatureModelDeep,
        )

        for setup_name, setup_rewards in SETUP_CONFIGS.items():
            if reward_system not in setup_rewards:
                continue

            embedding = embeddings_by_variant[setup_name]
            emb_list = (
                embedding.tolist()
                if not isinstance(embedding, list)
                else embedding
            )

            # Always add to regressor data
            regressor_reward = snap["reward_det"] if prefilter == "v2" else snap["reward"]
            regressor_data[setup_name]["policy_embedding"].append(emb_list)
            regressor_data[setup_name]["reward"].append(regressor_reward)
            regressor_data[setup_name]["policy_target"].append(reward_system)

            # Add to VDB only for the 5 quantile snapshots
            if include_in_vdb:
                desc = DESC_MAP.get(
                    reward_system,
                    f"Composite objective: {reward_system.replace('-', ', ')}.",
                )
                vdbs[setup_name].add_policy_from_kwargs(
                    policy_target=reward_system,
                    policy_seed="dqn",
                    policy_name=policy_key,
                    spec="dqn8",
                    description=desc,
                    reward=regressor_reward,
                    policy_embedding=embedding,
                    policy_embedding_base=embeddings_by_variant["base"],
                    policy_embedding_pair=embeddings_by_variant["pair"],
                    policy_embedding_trip=embeddings_by_variant["trip"],
                    q_table=None,
                    dag=None,
                    energy_consumption=None,
                    training_time_s=None,
                    model_path=snap["model_path"],
                )

    # Save VDBs
    for setup_name, vdb in vdbs.items():
        if vdb.index is not None:
            vdb.save()
            print(f"[{setup_name}] VDB saved")
        else:
            print(f"[{setup_name}] Warning: VDB empty, skipping save")

    # ── Phase 4: Save regressor data and train regressors ─────────────────
    print("\n" + "=" * 80)
    print("Phase 4: Training regressors")
    print("=" * 80)

    if skip_regressor:
        print("Skipping regressor training (--skip-regressor).")
        # Still save regressor data for later use
        for setup_name in SETUP_CONFIGS:
            data = regressor_data[setup_name]
            if not data["policy_embedding"]:
                continue
            json_path = os.path.join(
                data_dir, f"regressor_training_data_dqn_{setup_name}.json"
            )
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[{setup_name}] Regressor data saved to {json_path}")
        return

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for setup_name in SETUP_CONFIGS:
        data = regressor_data[setup_name]
        if not data["policy_embedding"]:
            print(f"[{setup_name}] Warning: no regressor data, skipping")
            continue

        json_path = os.path.join(
            data_dir, f"regressor_training_data_dqn_{setup_name}.json"
        )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[{setup_name}] Regressor data saved to {json_path}")

        model_path = os.path.join(
            models_dir, f"reward_regressor_dqn_{setup_name}.pkl"
        )
        plot_path = os.path.join(
            plots_dir, f"regression_plot_dqn_{setup_name}.jpeg"
        )

        train_regressor_variants(
            source_json_path=json_path,
            output_json_paths={setup_name: json_path},
            output_model_paths={setup_name: model_path},
            output_plot_paths={setup_name: plot_path},
            variants={setup_name: SETUP_CONFIGS[setup_name]},
            embedding_key_by_variant={setup_name: "policy_embedding"},
            normalize_embeddings=normalize_regressor,
            random_search=regressor_random_search,
            random_search_iters=regressor_random_search_iters,
            random_search_cv=regressor_random_search_cv,
            random_search_seed=regressor_random_search_seed,
        )
        print(f"[{setup_name}] Regressor trained -> {model_path}")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare pi2vec assets for DQN policies (successor + FAISS + regressor)."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="deeprl_runs_dqn_try6/8",
        help="Directory containing per-reward-system DQN policy folders",
    )
    parser.add_argument(
        "--minigrid-ids-path",
        type=str,
        default="deeprl_runs_dqn_try6/minigrid_ids.json",
        help="Path to minigrid_ids.json for state encoding",
    )
    parser.add_argument(
        "--canonical-count",
        type=int,
        default=128,
        help="Number of canonical states per setup",
    )
    parser.add_argument(
        "--regressor-only-fraction",
        type=float,
        default=0.6,
        help="Fraction of remaining positive snapshots used for regressor only",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="faiss_index_dqn",
        help="Base directory for FAISS index files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_rl",
        help="Directory for training data and canonical states",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models_dqn",
        help="Directory for regressor model outputs",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory for regressor plot outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for successor models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not remove existing FAISS index files before starting",
    )
    parser.add_argument(
        "--skip-regressor",
        action="store_true",
        help="Skip regressor training (save data only)",
    )
    parser.add_argument(
        "--normalize-regressor",
        action="store_true",
        help="Normalize embeddings before regressor fit",
    )
    parser.add_argument(
        "--regressor-random-search",
        action="store_true",
        help="Run RandomizedSearchCV for the regressor",
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
        "--prefilter",
        choices=["v1", "v2"],
        default="v1",
        help="Snapshot prefilter: v1=positive reward filter, v2=no positive filter",
    )
    args = parser.parse_args()

    prepare_dqn_policies(
        base_dir=args.base_dir,
        minigrid_ids_path=args.minigrid_ids_path,
        canonical_count=args.canonical_count,
        regressor_only_fraction=args.regressor_only_fraction,
        index_dir=args.index_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        plots_dir=args.plots_dir,
        epochs=args.epochs,
        seed=args.seed,
        reset=not args.no_reset,
        skip_regressor=args.skip_regressor,
        normalize_regressor=args.normalize_regressor,
        regressor_random_search=args.regressor_random_search,
        regressor_random_search_iters=args.regressor_random_search_iters,
        regressor_random_search_cv=args.regressor_random_search_cv,
        regressor_random_search_seed=args.regressor_random_search_seed,
        prefilter=args.prefilter,
    )
